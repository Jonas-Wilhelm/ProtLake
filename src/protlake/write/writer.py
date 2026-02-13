"""
General-purpose ProtlakeWriter for storing structure data (CIF/BCIF) with flexible metadata schemas.

This module provides a writer that:
- Stores BCIF data in shard files
- Optionally stores heavy metadata in JSON shards (msgpack + zstd compressed)
- Stores light metadata in a Delta table with a user-defined schema

The required core prolake fields (id, bcif_shard, bcif_off, etc.) are automatically added to the user schema.
"""

import os
import time
import random
import logging
import warnings
import atexit
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field

import msgpack
import zstandard as zstd
import pyarrow as pa
import pyarrow.dataset as ds

from deltalake import DeltaTable

from protlake.utils import ensure_dirs, get_protlake_dirs, cif_to_bcif_bytes, cif_bytes_to_bcif_bytes
from protlake.write.core import (
    ShardPackWriter,
    DeltaAppender,
    RetryConfig,
    LeaseMismatchRetry,
    load_delta_table_with_retries,
)

logger = logging.getLogger(__name__)


# --------------- Core schema fields (auto-merged) ---------------
def get_core_schema_fields(include_json: bool = True) -> List[pa.Field]:
    """
    Return the core schema fields that are automatically added to user schemas.
    
    These fields track the binary storage location of BCIF and optionally JSON data.
    """
    fields = [
        pa.field("id", pa.binary(32)),         # sha256 digest (binary, not hex)
        pa.field("id_hex", pa.string()),       # hex representation of the id
        pa.field("bcif_shard", pa.string()),
        pa.field("bcif_off", pa.int64()),
        pa.field("bcif_data_off", pa.int64()),
        pa.field("bcif_len", pa.int32()),
    ]
    if include_json:
        fields.extend([
            pa.field("json_shard", pa.string()),
            pa.field("json_off", pa.int64()),
            pa.field("json_data_off", pa.int64()),
            pa.field("json_len", pa.int32()),
        ])
    return fields


def build_full_schema(user_schema: pa.Schema, include_json: bool = True) -> pa.Schema:
    """
    Merge core fields with user-provided schema fields.
    
    Core fields are prepended; user fields that duplicate core field names are skipped.
    """
    core_fields = get_core_schema_fields(include_json=include_json)
    core_names = {f.name for f in core_fields}
    
    # Error if user tries to redefine a core field
    conflict_fields = [f.name for f in user_schema if f.name in core_names]
    if conflict_fields:
        raise ValueError(f"User schema fields {conflict_fields} conflict with reserved core field names.")
    
    return pa.schema(core_fields + list(user_schema))


# --------------- Configuration ---------------
@dataclass
class ProtlakeWriterConfig:
    """Configuration for ProtlakeWriter."""
    out_path: str
    user_schema: pa.Schema
    batch_size_metadata: int = 2500
    shard_size: int = 1 << 30  # 1 GB default
    bcif_shard_prefix: str = "bcif-pack"
    json_shard_prefix: str = "json-pack"
    claim_mode: bool = True
    claim_ttl: int = 60 * 30  # 30 minutes
    write_json_shards: bool = True
    retry_conf: RetryConfig = field(default_factory=RetryConfig)
    # CIF->BCIF conversion tolerances (used when accepting CIF input)
    rtol: float = 1e-6
    atol: float = 1e-4
    # Optional: custom serializer for heavy metadata before msgpack
    heavy_serializer: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    # Zstd compression level for heavy metadata
    zstd_level: int = 12


class ProtlakeWriter:
    """
    General-purpose writer for storing structure data with flexible metadata schemas.
    
    This writer handles:
    - BCIF shard packing (accepts CIF or BCIF bytes)
    - Optional JSON shard packing for heavy metadata (msgpack + zstd compressed)
    - Delta table appending for light metadata with user-defined schema
    
    Core fields (id, id_hex, bcif_shard, bcif_off, etc.) are automatically added
    to the user schema.
    
    Example usage:
        >>> import pyarrow as pa
        >>> user_schema = pa.schema([
        ...     pa.field("name", pa.string()),
        ...     pa.field("sample", pa.int32()),
        ...     pa.field("score", pa.float32()),
        ... ])
        >>> cfg = ProtlakeWriterConfig(out_path="/path/to/output", user_schema=user_schema)
        >>> writer = ProtlakeWriter(cfg)
        >>> writer.write_cif(
        ...     cif_bytes=cif_data,
        ...     light_metadata={"name": "test", "sample": 1, "score": 0.95},
        ...     heavy_metadata={"pae": [[1.0, 2.0], [2.0, 1.0]]},
        ... )
    """
    
    def __init__(self, cfg: ProtlakeWriterConfig):
        """
        Initialize the ProtlakeWriter.
        
        Args:
            cfg: Configuration for the writer.
        """
        self.cfg = cfg
        self.user_schema = cfg.user_schema
        self.full_schema = build_full_schema(self.user_schema, include_json=cfg.write_json_shards)
        
        self.shard_dir, self.delta_path = get_protlake_dirs(cfg.out_path)
        ensure_dirs([self.shard_dir, self.delta_path])
        
        # BCIF shard writer
        self.bcif_packer = ShardPackWriter(
            self.shard_dir,
            prefix=cfg.bcif_shard_prefix,
            max_bytes=cfg.shard_size,
            use_claims=cfg.claim_mode,
            claim_ttl=cfg.claim_ttl,
        )
        
        # Optional JSON shard writer
        self.json_packer: Optional[ShardPackWriter] = None
        if cfg.write_json_shards:
            self.json_packer = ShardPackWriter(
                self.shard_dir,
                prefix=cfg.json_shard_prefix,
                max_bytes=cfg.shard_size,
                use_claims=cfg.claim_mode,
                claim_ttl=cfg.claim_ttl,
            )
        
        # Delta table appender
        self.delta_appender = DeltaAppender(
            self.delta_path,
            self.full_schema,
            batch_size=cfg.batch_size_metadata,
            retry_config=cfg.retry_conf,
        )
        
        # Compression context (reusable)
        self._zstd_compressor = zstd.ZstdCompressor(level=cfg.zstd_level)
        
        # Suppress biotite warnings about missing auth_*_id attributes
        warnings.filterwarnings(
            "ignore",
            message="Attribute 'auth_.*_id' not found within 'atom_site' category",
            category=UserWarning,
            module="biotite.structure",
        )
        
        # Ensure final flush on exit
        atexit.register(self.finalize)
    
    def write_cif(
        self,
        cif_bytes: bytes,
        light_metadata: Dict[str, Any],
        heavy_metadata: Optional[Dict[str, Any]] = None,
        max_lease_retries: int = 2000,
    ) -> Dict[str, Any]:
        """
        Write a CIF structure with metadata to Protlake.
        
        The CIF bytes are converted to BCIF internally.
        
        Args:
            cif_bytes: Raw CIF file content as bytes.
            light_metadata: Dictionary of user-defined fields matching user_schema.
            heavy_metadata: Optional heavy metadata for JSON shard (if write_json_shards=True).
            max_lease_retries: Maximum retries on lease mismatch (concurrent write conflicts).
        
        Returns:
            Dictionary with 'id_hex' and storage location info.
        """
        bcif_bytes = cif_bytes_to_bcif_bytes(cif_bytes, rtol=self.cfg.rtol, atol=self.cfg.atol)
        return self.write_bcif(
            bcif_bytes=bcif_bytes,
            light_metadata=light_metadata,
            heavy_metadata=heavy_metadata,
            max_lease_retries=max_lease_retries,
        )
    
    def write_cif_file(
        self,
        cif_path: str,
        light_metadata: Dict[str, Any],
        heavy_metadata: Optional[Dict[str, Any]] = None,
        max_lease_retries: int = 2000,
    ) -> Dict[str, Any]:
        """
        Write a CIF file with metadata to Protlake.
        
        The CIF file is read and converted to BCIF internally.
        
        Args:
            cif_path: Path to the CIF file.
            light_metadata: Dictionary of user-defined fields matching user_schema.
            heavy_metadata: Optional heavy metadata for JSON shard (if write_json_shards=True).
            max_lease_retries: Maximum retries on lease mismatch (concurrent write conflicts).
        
        Returns:
            Dictionary with 'id_hex' and storage location info.
        """
        bcif_bytes = cif_to_bcif_bytes(cif_path, rtol=self.cfg.rtol, atol=self.cfg.atol)
        return self.write_bcif(
            bcif_bytes=bcif_bytes,
            light_metadata=light_metadata,
            heavy_metadata=heavy_metadata,
            max_lease_retries=max_lease_retries,
        )
    
    def write_bcif(
        self,
        bcif_bytes: bytes,
        light_metadata: Dict[str, Any],
        heavy_metadata: Optional[Dict[str, Any]] = None,
        max_lease_retries: int = 2000,
    ) -> Dict[str, Any]:
        """
        Write pre-converted BCIF bytes with metadata to Protlake.
        
        Args:
            bcif_bytes: BCIF (BinaryCIF) content as bytes.
            light_metadata: Dictionary of user-defined fields matching user_schema.
            heavy_metadata: Optional heavy metadata for JSON shard (if write_json_shards=True).
            max_lease_retries: Maximum retries on lease mismatch (concurrent write conflicts).
        
        Returns:
            Dictionary with 'id_hex' and storage location info.
        """
        # Prepare heavy metadata if needed
        json_pack_bytes_comp = None
        if self.cfg.write_json_shards and heavy_metadata is not None:
            serialized = heavy_metadata
            if self.cfg.heavy_serializer is not None:
                serialized = self.cfg.heavy_serializer(heavy_metadata)
            json_pack_bytes = msgpack.packb(serialized, use_bin_type=True, use_single_float=True)
            json_pack_bytes_comp = self._zstd_compressor.compress(json_pack_bytes)
        
        # Retry loop for lease mismatch scenarios
        bcif_pack = None
        json_pack = None
        
        for attempt in range(1, max_lease_retries + 1):
            try:
                # Pick shards
                bcif_shard_path = self.bcif_packer.choose_shard()
                if self.cfg.write_json_shards and json_pack_bytes_comp is not None:
                    json_shard_path = self.json_packer.choose_shard()
                    # Append JSON to shard
                    json_pack = self.json_packer.append(json_shard_path, json_pack_bytes_comp, rec_id=None)
                    logger.debug(
                        f"packed json id={json_pack['id_hex'][:12]}… "
                        f"shard={os.path.basename(json_pack['shard_path'])} "
                        f"off={json_pack['off']} len={json_pack['length']}"
                    )
                
                # Append BCIF to shard
                bcif_pack = self.bcif_packer.append(bcif_shard_path, bcif_bytes, rec_id=None)
                logger.debug(
                    f"packed bcif id={bcif_pack['id_hex'][:12]}… "
                    f"shard={os.path.basename(bcif_pack['shard_path'])} "
                    f"off={bcif_pack['off']} len={bcif_pack['length']}"
                )
                
                # Success
                break
                
            except LeaseMismatchRetry as e:
                if attempt >= max_lease_retries:
                    raise RuntimeError(
                        f"Failed to write after {max_lease_retries} lease retry attempts. "
                        f"All shards appear contested. Last error: {e}"
                    )
                backoff = min(0.1 * (1.5 ** min(attempt, 10)), 5.0) * random.uniform(0.8, 1.2)
                logger.warning(
                    f"Lease mismatch on attempt {attempt}/{max_lease_retries}, "
                    f"retrying in {backoff:.2f}s..."
                )
                time.sleep(backoff)
                continue
        
        # Build the row for Delta table
        core_data = {
            "id": bcif_pack["id_bytes"],
            "id_hex": bcif_pack["id_hex"],
            "bcif_shard": bcif_pack["shard_path"],
            "bcif_off": int(bcif_pack["off"]),
            "bcif_data_off": int(bcif_pack["data_off"]),
            "bcif_len": int(bcif_pack["length"]),
        }
        
        if self.cfg.write_json_shards:
            if json_pack is not None:
                core_data.update({
                    "json_shard": json_pack["shard_path"],
                    "json_off": int(json_pack["off"]),
                    "json_data_off": int(json_pack["data_off"]),
                    "json_len": int(json_pack["length"]),
                })
            else:
                # No heavy metadata provided
                core_data.update({
                    "json_shard": "",
                    "json_off": 0,
                    "json_data_off": 0,
                    "json_len": 0,
                })
        
        # Merge core data with light metadata
        delta_row = {**core_data, **light_metadata}
        self.delta_appender.add_row(delta_row)
        
        return {
            "id_hex": bcif_pack["id_hex"],
            "bcif_shard": bcif_pack["shard_path"],
            "bcif_off": bcif_pack["off"],
        }
    
    def flush(self) -> None:
        """Flush buffered rows to the Delta table."""
        self.delta_appender.flush()
    
    def finalize(self) -> None:
        """Flush all pending data. Called automatically on exit."""
        logger.info("Finalizing ProtlakeWriter")
        logger.info(f"  Flushing {self.delta_appender.row_count()} rows")
        self.delta_appender.flush()
    
    def row_count(self) -> int:
        """Return the number of buffered rows not yet flushed."""
        return self.delta_appender.row_count()
    
    # --------------- Query methods ---------------
    
    def check_exists(self, keys: Dict[str, Any]) -> bool:
        """
        Check if an entry matching the given key values exists.
        
        Args:
            keys: Dictionary of column-value pairs to match, e.g.:
                  {"name": "test", "sample": 1}
        
        Returns:
            True if at least one matching entry exists.
        """
        if not keys:
            raise ValueError("keys dict must not be empty")
        
        if not DeltaTable.is_deltatable(f"file://{os.path.abspath(self.delta_path)}"):
            return False
        
        # Build filter expression from dict
        filter_expr = None
        for col, val in keys.items():
            cond = ds.field(col) == val
            filter_expr = cond if filter_expr is None else (filter_expr & cond)
        
        dt = load_delta_table_with_retries(
            delta_path=self.delta_path,
            base_sleep=self.cfg.retry_conf.base_sleep,
            jitter=self.cfg.retry_conf.jitter,
            max_sleep=self.cfg.retry_conf.max_sleep,
            max_retries=self.cfg.retry_conf.max_retries,
        )
        
        pa_dataset = dt.to_pyarrow_dataset()
        # Use a minimal column to reduce I/O
        scanner = pa_dataset.scanner(filter=filter_expr, columns=["id_hex"])
        
        for batch in scanner.to_batches():
            if batch.num_rows > 0:
                return True
        return False
    
    def check_complete(
        self,
        expected_keys: List[Dict[str, Any]],
        key_columns: Optional[List[str]] = None,
    ) -> Tuple[bool, set, set]:
        """
        Check if all expected key combinations exist in the Delta table.
        
        Performs a single scan of the table (filtered by unique values of the first
        key column using `isin`) and compares against expected key combinations.
        Useful for verifying if a complete set of expected entries (e.g. multiple 
        inference samples from a prediction run) are in the protlake.
        
        Args:
            expected_keys: List of dicts, each containing key column values to check.
                           e.g., [{"name": "foo", "seed": 42, "sample_idx": 0}, ...]
            key_columns: Columns to use as composite keys. If None, inferred from
                         the keys of the first dict in expected_keys.
        
        Returns:
            Tuple of (all_complete, found_keys, missing_keys):
            - all_complete: True if all expected keys exist in the table
            - found_keys: Set of tuples representing found key combinations
            - missing_keys: Set of tuples representing missing key combinations
        
        Example:
            >>> expected = [
            ...     {"name": "foo", "seed": s, "sample_idx": i}
            ...     for s in [42, 123]
            ...     for i in range(5)
            ... ]
            >>> complete, found, missing = writer.check_complete(expected)
            >>> if not complete:
            ...     print(f"Missing {len(missing)} entries")
        """
        if not expected_keys:
            raise ValueError("expected_keys must be a non-empty list of dicts.")
        
        # Infer key columns from first dict if not provided
        if key_columns is None:
            key_columns = list(expected_keys[0].keys())
        
        # Build target set of tuples
        target_keys: set = {
            tuple(d[col] for col in key_columns)
            for d in expected_keys
        }
        
        # Early return if table doesn't exist
        if not DeltaTable.is_deltatable(f"file://{os.path.abspath(self.delta_path)}"):
            return False, set(), target_keys
        
        dt = load_delta_table_with_retries(
            delta_path=self.delta_path,
            base_sleep=self.cfg.retry_conf.base_sleep,
            jitter=self.cfg.retry_conf.jitter,
            max_sleep=self.cfg.retry_conf.max_sleep,
            max_retries=self.cfg.retry_conf.max_retries,
        )
        
        # Build isin filter on first key column for efficient pruning
        first_col = key_columns[0]
        unique_first_vals = list({d[first_col] for d in expected_keys})
        filter_expr = ds.field(first_col).isin(unique_first_vals)
        
        pa_dataset = dt.to_pyarrow_dataset()
        scanner = pa_dataset.scanner(filter=filter_expr, columns=key_columns)
        
        found_keys: set = set()
        remaining = len(target_keys)
        
        for batch in scanner.to_batches():
            # Extract columns as Python lists
            col_data = [batch.column(col).to_pylist() for col in key_columns]
            
            for row_vals in zip(*col_data):
                if row_vals in target_keys and row_vals not in found_keys:
                    found_keys.add(row_vals)
                    remaining -= 1
            
            if remaining == 0:
                break  # Early exit once all found
        
        missing_keys = target_keys - found_keys
        all_complete = len(missing_keys) == 0
        
        return all_complete, found_keys, missing_keys
    
    def query(
        self,
        filter_expr: Optional[ds.Expression] = None,
        columns: Optional[List[str]] = None,
    ) -> pa.Table:
        """
        Query entries from the Delta table.
        
        Args:
            filter_expr: Optional PyArrow dataset filter expression.
            columns: Optional list of columns to return. If None, returns all columns.
        
        Returns:
            PyArrow Table with matching entries.
        """
        if not DeltaTable.is_deltatable(f"file://{os.path.abspath(self.delta_path)}"):
            return pa.table({f.name: [] for f in self.full_schema}, schema=self.full_schema)
        
        dt = load_delta_table_with_retries(
            delta_path=self.delta_path,
            base_sleep=self.cfg.retry_conf.base_sleep,
            jitter=self.cfg.retry_conf.jitter,
            max_sleep=self.cfg.retry_conf.max_sleep,
            max_retries=self.cfg.retry_conf.max_retries,
        )
        
        pa_dataset = dt.to_pyarrow_dataset()
        scanner = pa_dataset.scanner(filter=filter_expr, columns=columns)
        return scanner.to_table()
    
    def delete(self, predicate: str) -> None:
        """
        Delete entries matching a predicate.
        
        Args:
            predicate: SQL-like predicate string, e.g. "name = 'test'"
        """
        if not os.path.exists(os.path.join(self.delta_path, "_delta_log")):
            logger.info("Delta table does not exist yet; nothing to delete.")
            return
        dt = DeltaTable(f"file://{os.path.abspath(self.delta_path)}")
        dt.delete(predicate)
