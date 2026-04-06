import logging
from math import e
import os
import time
from typing import Any, Dict, Optional, overload
from deltalake import DeltaTable, write_deltalake
from protlake.query import check_exists as delta_check_exists
from protlake.utils import (
    DeltaTable_nrow, 
    deltatable_maintenance,
    bcif_shard_to_mmCIF_file,
    bcif_shard_to_mmCIF_str,
    pread_bcif_to_atom_array,
    read_bytes_from_shard_fd,
)
from protlake.write.writer import get_core_schema_fields
from protlake.write.core import RetryConfig
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
import numpy as np
from biotite.structure import to_sequence

logger = logging.getLogger(__name__)

class ProtLake():
    def __init__(self, path, create=False, retry_conf: Optional[RetryConfig] = None):
        self.path = path
        self.delta_path = os.path.join(path, 'delta')
        self.shard_path = os.path.join(path, 'shards')
        self.retry_conf = retry_conf if retry_conf is not None else RetryConfig()
        self.n_row = None

        self.dt = None
        self.colnames = None
        self.shard_index_cache = None
        self.load(create=create)

    def load(self, create=False) -> bool:
        try:
            self.dt = DeltaTable(f"file://{os.path.abspath(self.delta_path)}")
            if create:
                print(f"ProtLake at {self.path} already exists, not creating a new one, despite create=True.")
        except Exception as e:
            if create:
                print(f"Creating new ProtLake at {self.path}")
                os.makedirs(self.delta_path, exist_ok=True)
                os.makedirs(self.shard_path, exist_ok=True)
                empty = pa.table(
                    {f.name: pa.array([], type=f.type) for f in get_core_schema_fields()}
                )
                write_deltalake(
                    f"file://{os.path.abspath(self.delta_path)}",
                    empty,
                    configuration={'delta.checkpointInterval': '500'}
                )
                self.dt = DeltaTable(f"file://{os.path.abspath(self.delta_path)}")
            else:
                raise RuntimeError(f"Could not load ProtLake at {self.path}: {e}")
        self.colnames = [f.name for f in self.dt.schema().fields]
        return True

    def nrow(self) -> int:
        self.load()
        self.n_row = DeltaTable_nrow(self.dt)
        return self.n_row

    def to_pandas(self) -> pd.DataFrame:
        return self.dt.to_pandas()

    def check_exists(self, keys: Dict[str, Any]) -> bool:
        return delta_check_exists(
            delta_path=self.delta_path,
            keys=keys,
            retry_config=self.retry_conf,
        )
    
    def maintenance(self, target_size = 1 << 28, max_concurrent_tasks=2, retention_hours=0, reload=False) -> bool:
        deltatable_maintenance(self.dt, target_size=target_size, max_concurrent_tasks=max_concurrent_tasks, retention_hours=retention_hours)
        if reload:
            self.load()
        return True

    def load_shard_index_cache(
            self, 
            columns: str | list[str] | None = None,
            filters: Optional[Dict[str, list[Any]]] = None
        ) -> bool:
        required = ["bcif_shard", "bcif_data_off", "bcif_len"]
        if isinstance(columns, str):
            columns = [columns]
        columns = list(columns or [])
        filters = filters or {}

        # include required columns, requested columns, and filter columns
        return_cols = set(required + columns)
        filter_cols = set(filters.keys())
        all_cols = return_cols.union(filter_cols)

        # check that all referenced cols are in the table and throw error if not
        missing = all_cols - set(self.colnames)
        if missing:
            raise ValueError(f"load_shard_index: Some requested columns are not in the ProtLake: {missing}")

        # build deltalake filter format:
        # {"name": ["a", "b"]} -> [("name", "in", ["a", "b"])]
        dl_filters = []
        for col, values in filters.items():
            if values is None or len(values) == 0 or not isinstance(values, (list, set, tuple)):
                raise ValueError(f"Filter values for column '{col}' must be a non-empty list, set, or tuple.")
            dl_filters.append((col, "in", list(values)))

        table = self.dt.to_pyarrow_table(
            columns=list(return_cols),
            filters=dl_filters if dl_filters else None,
        )

        self.shard_index_cache = table
        return True

    def validate_bcif_entries(
        self,
        batch_size: int = 10_000,
    ) -> list[bytes]:
        """
        Stream over the Delta table and return ids whose BCIF payload cannot be
        read from the shard or whose CRC check fails.

        Only the columns required for validation are projected from the Delta table.
        """
        self.load()
        self.nrow()

        dataset = self.dt.to_pyarrow_dataset()
        scanner = ds.Scanner.from_dataset(
            dataset,
            columns=["id", "bcif_shard", "bcif_data_off", "bcif_len"],
            batch_size=batch_size,
        )

        invalid_ids: list[bytes] = []
        processed_rows = 0
        start_time = time.perf_counter()
        fd_cache: dict[str, int] = {}

        try:
            for batch in scanner.to_batches():
                cols = batch.to_pydict()
                ids = cols["id"]
                shards = cols["bcif_shard"]
                offsets = cols["bcif_data_off"]
                lengths = cols["bcif_len"]

                for rec_id, shard, offset, length in zip(ids, shards, offsets, lengths):
                    processed_rows += 1

                    if shard is None or offset is None or length is None:
                        invalid_ids.append(rec_id)
                        continue

                    shard_path = os.path.join(self.shard_path, str(shard))
                    try:
                        fd = fd_cache.get(shard_path)
                        if fd is None:
                            fd = os.open(shard_path, os.O_RDONLY)
                            fd_cache[shard_path] = fd

                        read_bytes_from_shard_fd(fd, offset, length)
                    except Exception:
                        invalid_ids.append(rec_id)

                elapsed = time.perf_counter() - start_time
                rate = processed_rows / elapsed if elapsed > 0 else 0.0
                pct = (100.0 * processed_rows / self.n_row) if self.n_row else 0.0
                logger.info(
                    "Validated %s BCIF entries (%.1f %%), invalid=%s, rate=%.0f rows/s",
                    f"{processed_rows:,}",
                    pct,
                    f"{len(invalid_ids):,}",
                    rate,
                )
        finally: # Close any open file descriptors
            for fd in fd_cache.values():
                try:
                    os.close(fd)
                except OSError:
                    pass

        elapsed = time.perf_counter() - start_time
        rate = processed_rows / elapsed if elapsed > 0 else 0.0
        logger.info(
            "Finished BCIF validation: scanned %s rows, invalid=%s, elapsed=%.1fs, rate=%.0f rows/s",
            f"{processed_rows:,}",
            f"{len(invalid_ids):,}",
            elapsed,
            rate,
        )
        return invalid_ids

    # ------------- extract_cif -------------
    # ---------------------------------------
    @overload
    def extract_cif(
        self,
        bcif_shard: str | list[str],
        bcif_data_off: int | list[int],
        bcif_len: int | list[int],
        file_names: str | list[str],
        out_dir: str,
        overwrite: bool = False,
        as_str: bool = False,
    ) -> None: ...

    @overload
    def extract_cif(
        self,
        *,
        out_dir: str,
        overwrite: bool = False,
        as_str: bool = False,
        df: pd.DataFrame,
        file_name_cols: str | list[str] | None = None,
    ) -> None: ...

    def extract_cif(
        self,
        bcif_shard: Optional[str | list[str]] = None,
        bcif_data_off: Optional[int | list[int]] = None,
        bcif_len: Optional[int | list[int]] = None,
        file_names: Optional[str | list[str]] = None,
        file_name_cols: Optional[str | list[str]] = None,
        out_dir: str = None,
        overwrite: bool = False,
        as_str: bool = False,
        df: Optional[pd.DataFrame] = None,
    ) -> None | str | list[str]:
        
        if out_dir is None and not as_str:
            raise ValueError("out_dir must be specified.")
        
        if df is not None:
            # check that no other args are provided
            if any(arg is not None for arg in (bcif_shard, bcif_data_off, bcif_len, file_names)):
                raise ValueError("Pass either df OR explicit bcif_shard/bcif_data_off/bcif_len/file_names, not both.")
            
            # check if df has all the required columns
            required_cols = {'bcif_shard', 'bcif_data_off', 'bcif_len'}
            if not required_cols.issubset(df.columns):
                raise ValueError(f"df must contain the following columns: {required_cols}")
            
            bcif_shard = df['bcif_shard'].tolist()
            bcif_data_off = df['bcif_data_off'].tolist()
            bcif_len = df['bcif_len'].tolist()
            if file_name_cols is None:
                file_name_cols = 'name'
                print(
                    "Warning: extract_cif() defaulted file_name_cols to 'name'. "
                    "If 'name' is not unique, files may be overwritten or skipped. "
                    "Specify file_name_cols to avoid this."
                )
            if isinstance(file_name_cols, str):
                file_name_cols = [file_name_cols]
            missing_name_cols = set(file_name_cols) - set(df.columns)
            if missing_name_cols:
                raise ValueError(f"df is missing file name columns: {missing_name_cols}")
            file_names = (
                df[file_name_cols]
                .astype(str)
                .agg('_'.join, axis=1)
                .tolist()
            )
        else:
            if bcif_shard is None or bcif_data_off is None or bcif_len is None or file_names is None:
                raise ValueError("Either df or all of bcif_shard, bcif_data_off, bcif_len, and file_names must be provided.")
            if isinstance(bcif_shard, str):
                bcif_shard = [bcif_shard]
            if isinstance(bcif_data_off, int):
                bcif_data_off = [bcif_data_off]
            if isinstance(bcif_len, int):
                bcif_len = [bcif_len]
            if isinstance(file_names, str):
                file_names = [file_names]

        if file_name_cols is not None and df is None:
            raise ValueError("file_name_cols can only be used when df is provided.")

        if not as_str and file_names is None:
            raise ValueError("file_names must be provided explicitly or derived from df via file_name_cols.")

        if not as_str:
            os.makedirs(out_dir, exist_ok=True)
            for shard, offset, length, file_name in zip(bcif_shard, bcif_data_off, bcif_len, file_names):
                shard_path = os.path.join(self.shard_path, str(shard))
                if not file_name.endswith('.cif'):
                    file_name += '.cif'
                out_path = os.path.join(out_dir, file_name)
                if os.path.exists(out_path) and not overwrite:
                    print(f"File {out_path} exists, skipping...")
                    continue
                try:
                    bcif_shard_to_mmCIF_file(shard_path, offset, length, out_path)
                except Exception as e:
                    print(f"Error reading {shard_path} at offset {offset} with length {length}: {e}")
                    continue
        else:
            out_str_list = []
            for shard, offset, length in zip(bcif_shard, bcif_data_off, bcif_len):
                shard_path = os.path.join(self.shard_path, str(shard))
                try:
                    out_str = bcif_shard_to_mmCIF_str(shard_path, offset, length)
                    out_str_list.append(out_str)
                except Exception as e:
                    print(f"Error reading {shard_path} at offset {offset} with length {length}: {e}")
                    continue

        if as_str:
            # return list or if only one, return single string
            return out_str_list if len(out_str_list) > 1 else out_str_list[0]

    def to_fasta(
        self,
        path: str,
        name_cols: str | list[str] = 'name',
        sep: str = '_',
    ) -> None:
        self.load()

        if 'sequence' not in self.colnames:
            print("No 'sequence' column found. Exiting without writing FASTA.")
            return

        if isinstance(name_cols, str):
            name_cols = [name_cols]

        missing_name_cols = set(name_cols) - set(self.colnames)
        if missing_name_cols:
            raise ValueError(f"ProtLake is missing FASTA name columns: {missing_name_cols}")

        dataset = self.dt.to_pyarrow_dataset()
        columns = [*name_cols, 'sequence']
        table = dataset.to_table(columns=columns)
        df = table.to_pandas()

        names = (
            df[name_cols]
            .astype(str)
            .agg(sep.join, axis=1)
            .tolist()
        )
        sequences = df['sequence'].tolist()

        with open(path, 'w') as fasta_handle:
            for name, sequence in zip(names, sequences):
                if sequence is None:
                    continue
                fasta_handle.write(f">{name}\n{sequence}\n")

    # --------------- get_seq ---------------
    # ---------------------------------------
    @overload
    def get_seq(
        self,
        bcif_shard: str | list[str],
        bcif_off: int | list[int],
        bcif_len: int | list[int],
        name: str | list[str],
        df: None = None,
    ) -> str | list[str]: ...

    @overload
    def get_seq(
        self,
        bcif_shard: None = None,
        bcif_off: None = None,
        bcif_len: None = None,
        name: None = None,
        df: pd.DataFrame = ...,
        chains: list[str] = ['A'],
    ) -> str | list[str]: ...

    def get_seq(
        self,
        bcif_shard: Optional[str | list[str]] = None,
        bcif_off: Optional[int | list[int]] = None,
        bcif_len: Optional[int | list[int]] = None,
        name: Optional[str | list[str]] = None,
        df: Optional[pd.DataFrame] = None,
        chains: list[str] = ['A'],
    ) -> str | list[str]:
        
        if df is not None:
            if any(arg is not None for arg in (bcif_shard, bcif_off, bcif_len, name)):
                raise ValueError("Pass either df OR explicit bcif_shard/off/len/name, not both.")
            bcif_shard = df['bcif_shard'].tolist()
            bcif_off = df['bcif_data_off'].tolist()
            bcif_len = df['bcif_len'].tolist()
            name = df['name'].tolist()
        else:
            if bcif_shard is None or bcif_off is None or bcif_len is None or name is None:
                raise ValueError("Either df or all of bcif_shard, bcif_off, bcif_len, and name must be provided.")
            if isinstance(bcif_shard, str):
                bcif_shard = [bcif_shard]
            if isinstance(bcif_off, int):
                bcif_off = [bcif_off]
            if isinstance(bcif_len, int):
                bcif_len = [bcif_len]
            if isinstance(name, str):
                name = [name]

        seq_list = []
        for shard, offset, length, id in zip(bcif_shard, bcif_off, bcif_len, name):
            shard_path = os.path.join(self.shard_path, str(shard))
            try:
                aa = pread_bcif_to_atom_array(shard_path, offset, length)
                seqs, _ = to_sequence(aa[np.isin(aa.chain_id, chains) & ~aa.hetero])
                if len(seqs) > 1:
                    print(f"Warning: More than one chain found in {id}, returning first chain only.")
                seq_list.append(str(seqs[0]))
            except Exception as e:
                print(f"Error reading {shard_path} at offset {offset} with length {length}: {e}")
                seq_list.append(None)
                continue

        return seq_list if len(seq_list) > 1 else seq_list[0]

    def dedupe_by_name(self, name_col: str = 'name', batch_size: int = 100_000, dry_run: bool = False) -> None:
        """
        Remove duplicate entries based on the specified name column, keeping only the first occurrence.

        This function reads the Delta table in batches, identifies duplicates based on the name column,
        and removes all but the first occurrence of each duplicate entry. It uses a set to track seen names
        and a list to collect IDs of entries to remove. After processing all batches, it deletes the identified
        duplicate entries from the Delta table.

        Args:
            name_col (str): The name of the column to check for duplicates. Default is 'name'.
            batch_size (int): The number of rows to process in each batch. Default is 10,000.
            dry_run (bool): If True, do not actually delete duplicates, just log them. Default is False.
        """
        self.load()
        self.nrow()

        dataset = self.dt.to_pyarrow_dataset()
        scanner = ds.Scanner.from_dataset(
            dataset,
            columns=["id_hex", name_col],
            batch_size=batch_size,
        )

        seen_names = set()
        ids_to_remove = []
        processed_rows = 0
        start_time = time.perf_counter()

        for batch in scanner.to_batches():
            cols = batch.to_pydict()
            ids = cols["id_hex"]
            names = cols[name_col]

            for rec_id, name in zip(ids, names):
                processed_rows += 1
                if name in seen_names:
                    ids_to_remove.append(rec_id)
                else:
                    seen_names.add(name)
            
            elapsed = time.perf_counter() - start_time
            rate = processed_rows / elapsed if elapsed > 0 else 0.0
            pct = (100.0 * processed_rows / self.n_row) if self.n_row else 0.0
            logger.info(
                "Scanned %s rows for deduplication (%.1f %%), found %s duplicates so far, rate=%.0f rows/s",
                f"{processed_rows:,}",
                pct,
                f"{len(ids_to_remove):,}",
                rate,
            )

        if ids_to_remove:
            logger.info(f"Removing {len(ids_to_remove)} duplicate entries based on column '{name_col}'")
            if not dry_run:
                quoted = [f"'{v.replace(chr(39), chr(39)*2)}'" for v in ids_to_remove]
                self.dt.delete(f"id_hex IN ({', '.join(quoted)})")
                self.load()  # Reload to update state after deletion
            else:
                logger.info("Dry run enabled, not actually deleting duplicates.")
        else:
            logger.info(f"No duplicates found based on column '{name_col}'")

    def __repr__(self):
        if self.n_row is None:
            self.nrow()
        return f"ProtLake at {self.path} with {self.n_row:,} structures"
