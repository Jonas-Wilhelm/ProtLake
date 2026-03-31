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
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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