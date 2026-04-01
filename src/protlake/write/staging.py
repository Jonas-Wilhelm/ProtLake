"""
Staging writer for ProtLake.

Writes PDB structures and metadata to a two-phase spool directory
(incoming → ready) so that a downstream ingestion process only ever
sees complete, fully-fsynced batches.

Directory layout under ``protlake_dir``::

    spool/
        incoming/<batch_id>/   # files being written (incomplete)
        ready/<batch_id>/      # atomically published (complete)
"""

import json
import logging
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from .writer import ProtlakeWriter

logger = logging.getLogger(__name__)


def ensure_spool_dirs(protlake_dir: str):
    """Create (if needed) and return the incoming and ready spool directories."""
    spool_dir = os.path.join(protlake_dir, "spool")
    incoming_dir = os.path.join(spool_dir, "incoming")
    ready_dir = os.path.join(spool_dir, "ready")
    for d in [spool_dir, incoming_dir, ready_dir]:
        os.makedirs(d, exist_ok=True)
    return incoming_dir, ready_dir


def _fsync_file(path: str) -> None:
    with open(path, "rb") as f:
        os.fsync(f.fileno())


def _fsync_dir(dir_path: str) -> None:
    fd = os.open(dir_path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class StagingWriter:
    """Accumulate PDB + metadata entries and publish them as one atomic batch.

    Each entry consists of:
    * a PDB string       → ``<group_id>.pdb``
    * a light_metadata dict (scores etc.) → ``<group_id>.json``
    * an optional heavy_metadata dict     → ``<group_id>.heavy.json``

    Usage::

        writer = StagingWriter("/data/protlake")
        writer.add(pdb_string, {"description": "design_1", "score": 0.95})
        writer.add(pdb_string2, {"description": "design_2", "score": 0.91},
                   heavy_metadata={"pae": [[1.0, 2.0], [2.0, 1.0]]})
        writer.publish()          # atomic rename incoming → ready
    """

    def __init__(self, protlake_dir: str):
        incoming_root, ready_root = ensure_spool_dirs(protlake_dir)
        self.batch_id = str(uuid.uuid4())
        self._incoming_dir = os.path.join(incoming_root, self.batch_id)
        self._ready_dir = os.path.join(ready_root, self.batch_id)
        os.makedirs(self._incoming_dir, exist_ok=False)

        self._written_paths: List[str] = []
        self._count = 0
        self._published = False

    # ------------------------------------------------------------------
    def add(
        self,
        pdb_string: str,
        light_metadata: Dict[str, Any],
        heavy_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Stage one PDB structure with its metadata.

        Args:
            pdb_string: PDB-format structure as a string.
            light_metadata: Flat dict of lightweight scores / descriptors.
            heavy_metadata: Optional dict of bulkier data (PAE matrices, etc.).

        Returns:
            The ``group_id`` assigned to this entry.
        """
        if self._published:
            raise RuntimeError("Batch already published; create a new StagingWriter.")

        group_id = f"group-{self._count:05d}"

        # PDB
        pdb_path = os.path.join(self._incoming_dir, f"{group_id}.pdb")
        with open(pdb_path, "w") as f:
            f.write(pdb_string)
        self._written_paths.append(pdb_path)

        # Light metadata
        json_path = os.path.join(self._incoming_dir, f"{group_id}.json")
        with open(json_path, "w") as f:
            json.dump(light_metadata, f, indent=4)
        self._written_paths.append(json_path)

        # Heavy metadata (optional)
        if heavy_metadata is not None:
            heavy_path = os.path.join(self._incoming_dir, f"{group_id}.heavy.json")
            with open(heavy_path, "w") as f:
                json.dump(heavy_metadata, f, indent=4)
            self._written_paths.append(heavy_path)

        self._count += 1
        return group_id

    @property
    def entry_count(self) -> int:
        return self._count

    # ------------------------------------------------------------------
    def publish(self) -> Optional[str]:
        """Flush, write a manifest, and atomically move the batch to ``ready/``.

        Returns:
            The path to the ready batch directory, or ``None`` if the batch
            was empty (the incoming directory is removed in that case).
        """
        if self._published:
            raise RuntimeError("Batch already published.")

        if self._count == 0:
            os.rmdir(self._incoming_dir)
            logger.info("Empty batch %s removed.", self.batch_id)
            self._published = True
            return None

        # Write manifest
        manifest_path = os.path.join(self._incoming_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(
                {
                    "batch_id": self.batch_id,
                    "entry_count": self._count,
                    "file_count": len(self._written_paths),
                    "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID", "N/A"),
                    "SLURM_ARRAY_TASK_ID": os.environ.get("SLURM_ARRAY_TASK_ID", "N/A"),
                    "hostname": os.uname().nodename,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                f,
                indent=4,
            )
        self._written_paths.append(manifest_path)

        # Fsync every file so all data is durable before the rename
        for path in self._written_paths:
            _fsync_file(path)

        # Atomic publish
        os.rename(self._incoming_dir, self._ready_dir)

        # Fsync the parent directory so the rename is durable
        _fsync_dir(os.path.dirname(self._ready_dir))

        self._published = True
        logger.info(
            "Published batch %s (%d entries) → %s",
            self.batch_id,
            self._count,
            self._ready_dir,
        )
        return self._ready_dir


class SpoolIngester:
    """Ingest published staging batches from ``spool/ready`` into a ProtlakeWriter."""

    def __init__(self, protlake_dir: str, writer: ProtlakeWriter, min_entries: int = 1):
        _, ready_root = ensure_spool_dirs(protlake_dir)
        self._ready_root = ready_root
        self.writer = writer
        self.min_entries = min_entries

    def _list_ready_batches(self) -> List[str]:
        batch_dirs = []
        for name in os.listdir(self._ready_root):
            path = os.path.join(self._ready_root, name)
            if os.path.isdir(path):
                batch_dirs.append(path)
        return sorted(batch_dirs)

    def _load_manifest(self, batch_dir: str) -> Dict[str, Any]:
        manifest_path = os.path.join(batch_dir, "manifest.json")
        with open(manifest_path, "r") as f:
            return json.load(f)

    def _total_ready_entries(self, batch_dirs: List[str]) -> int:
        total = 0
        for batch_dir in batch_dirs:
            total += int(self._load_manifest(batch_dir).get("entry_count", 0))
        return total

    def _iter_group_ids(self, batch_dir: str) -> List[str]:
        group_ids = []
        for filename in os.listdir(batch_dir):
            if filename.startswith("group-") and filename.endswith(".pdb"):
                group_ids.append(filename[:-4])
        return sorted(group_ids)

    def _ingest_batch(self, batch_dir: str) -> int:
        ingested = 0
        for group_id in self._iter_group_ids(batch_dir):
            pdb_path = os.path.join(batch_dir, f"{group_id}.pdb")
            json_path = os.path.join(batch_dir, f"{group_id}.json")
            heavy_path = os.path.join(batch_dir, f"{group_id}.heavy.json")

            with open(json_path, "r") as f:
                light_metadata = json.load(f)

            heavy_metadata = None
            if os.path.exists(heavy_path):
                with open(heavy_path, "r") as f:
                    heavy_metadata = json.load(f)

            self.writer.write_pdb_file(
                pdb_path=pdb_path,
                light_metadata=light_metadata,
                heavy_metadata=heavy_metadata,
            )
            ingested += 1

        self.writer.flush()
        shutil.rmtree(batch_dir)
        return ingested

    def run_once(self, run_maintenance=False) -> Dict[str, int]:
        batch_dirs = self._list_ready_batches()
        if not batch_dirs:
            return {"processed_batches": 0, "processed_entries": 0, "pending_entries": 0}

        pending_entries = self._total_ready_entries(batch_dirs)
        if pending_entries < self.min_entries:
            logger.info(
                "Skipping spool ingest: %d ready entries below min_entries=%d",
                pending_entries,
                self.min_entries,
            )
            return {"processed_batches": 0, "processed_entries": 0, "pending_entries": pending_entries}

        processed_batches = 0
        processed_entries = 0
        for batch_dir in batch_dirs:
            manifest = self._load_manifest(batch_dir)
            logger.info(
                "Ingesting staged batch %s with %s entries",
                manifest.get("batch_id", os.path.basename(batch_dir)),
                manifest.get("entry_count", "unknown"),
            )
            processed_entries += self._ingest_batch(batch_dir)
            processed_batches += 1

        if run_maintenance and processed_entries > 0:
            t0 = time.time()
            self.writer.maintenance()
            print(f"DeltaTable maintenance completed in {time.time() - t0:.1f} seconds.")

        return {
            "processed_batches": processed_batches,
            "processed_entries": processed_entries,
            "pending_entries": 0,
        }

    def run_loop(self, interval_seconds: int, run_maintenance=False) -> None:
        while True:
            result = self.run_once(run_maintenance=run_maintenance)
            logger.info(
                "Spool ingest sweep complete: %d batches, %d entries",
                result["processed_batches"],
                result["processed_entries"],
            )
            time.sleep(interval_seconds)
