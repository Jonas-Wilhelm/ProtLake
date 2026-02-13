#!/usr/bin/env python
"""
Tests for ProtlakeWriter class.

Uses 6Y7A.cif as a base structure, adds random noise to coordinates,
and writes multiple samples to test the writer functionality.
"""

import os
import shutil
import pytest
import warnings
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx

from typing import List

# --------------- Test configuration ---------------

# Paths relative to this test file
TEST_DIR = Path(__file__).resolve().parent
STRUCTURES_DIR = TEST_DIR / "structures"
OUTPUT_DIR = TEST_DIR / "output" / "protlake_writer"
TEST_CIF = STRUCTURES_DIR / "6Y7A.cif"

# Test parameters
NUM_SAMPLES = 100
NOISE_STD = 0.1  # Angstroms
RANDOM_SEED = 42


# --------------- Fixtures ---------------

@pytest.fixture(scope="module")
def base_structure():
    """Load the base structure from 6Y7A.cif."""
    cif_file = pdbx.CIFFile.read(str(TEST_CIF))
    atoms = pdbx.get_structure(cif_file, model=1, extra_fields=[])
    # Get only ATOM records (not HETATM) for simplicity
    atoms = atoms[struc.filter_amino_acids(atoms)]
    return atoms


@pytest.fixture(scope="module")
def output_dir():
    """Create and clean the output directory."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yield OUTPUT_DIR
    # Cleanup after all tests (optional - comment out to inspect output)
    shutil.rmtree(OUTPUT_DIR)


@pytest.fixture(scope="module")
def user_schema():
    """Define a user schema for testing."""
    return pa.schema([
        pa.field("name", pa.string()),
        pa.field("sample_idx", pa.int32()),
        pa.field("noise_seed", pa.int32()),
        pa.field("rmsd_from_original", pa.float32()),
        pa.field("atom_count", pa.int32()),
    ])


@pytest.fixture(scope="module")
def protlake_writer(output_dir, user_schema):
    """Create a ProtlakeWriter instance for testing."""
    from protlake.write.writer import ProtlakeWriter, ProtlakeWriterConfig
    
    cfg = ProtlakeWriterConfig(
        out_path=str(output_dir / "protlake_output"),
        user_schema=user_schema,
        batch_size_metadata=25,  # Small batch for testing
        shard_size=10 * 1024 * 1024,  # 10 MB shards for testing
        write_json_shards=True,
        zstd_level=3,  # Lower compression for speed
    )
    
    writer = ProtlakeWriter(cfg)
    yield writer
    # Ensure final flush
    writer.finalize()


def structure_to_cif_bytes(atoms: struc.AtomArray) -> bytes:
    """Convert an AtomArray to CIF bytes."""
    import io
    cif_file = pdbx.CIFFile()
    pdbx.set_structure(cif_file, atoms, data_block="structure")
    
    # Write to string buffer
    buffer = io.StringIO()
    cif_file.write(buffer)
    return buffer.getvalue().encode("utf-8")


def add_coordinate_noise(atoms: struc.AtomArray, noise_std: float, rng: np.random.Generator) -> struc.AtomArray:
    """Add Gaussian noise to atom coordinates."""
    noisy_atoms = atoms.copy()
    noise = rng.normal(0, noise_std, size=atoms.coord.shape)
    noisy_atoms.coord = atoms.coord + noise
    return noisy_atoms


def calculate_rmsd(atoms1: struc.AtomArray, atoms2: struc.AtomArray) -> float:
    """Calculate RMSD between two structures."""
    diff = atoms1.coord - atoms2.coord
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))


def _parallel_worker_write(
    worker_id: int,
    out_path: str,
    user_schema_bytes: bytes,
    base_cif_bytes: bytes,
    n_atoms: int,
    samples_per_worker: int,
) -> List[str]:
    """Worker function that writes noised structures (module-level for pickling)."""
    import io
    import numpy as np
    import pyarrow as pa
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx
    from protlake.write.writer import ProtlakeWriter, ProtlakeWriterConfig

    # Deserialize schema
    reader = pa.ipc.open_stream(user_schema_bytes)
    schema = reader.schema
    reader.close()

    cfg = ProtlakeWriterConfig(
        out_path=out_path,
        user_schema=schema,
        batch_size_metadata=10,
        shard_size=10 * 1024 * 1024,  # 10 MB
        write_json_shards=True,
        claim_mode=True,
        claim_ttl=60,
    )

    writer = ProtlakeWriter(cfg)
    rng = np.random.default_rng(RANDOM_SEED + worker_id * 1000)
    written_ids = []

    # Parse base structure from CIF bytes
    cif_file = pdbx.CIFFile.read(io.StringIO(base_cif_bytes.decode("utf-8")))
    atoms = pdbx.get_structure(cif_file, model=1, extra_fields=[])
    atoms = atoms[struc.filter_amino_acids(atoms)]

    for i in range(samples_per_worker):
        # Add noise to coordinates
        noisy_atoms = atoms.copy()
        noise = rng.normal(0, NOISE_STD, size=atoms.coord.shape)
        noisy_atoms.coord = atoms.coord + noise

        # Calculate RMSD
        diff = atoms.coord - noisy_atoms.coord
        rmsd = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))

        # Convert to CIF bytes
        cif_file_out = pdbx.CIFFile()
        pdbx.set_structure(cif_file_out, noisy_atoms, data_block="structure")
        buffer = io.StringIO()
        cif_file_out.write(buffer)
        noisy_cif_bytes = buffer.getvalue().encode("utf-8")

        # Generate fake heavy metadata
        n_residues = len(set(noisy_atoms.res_id))
        fake_pae = rng.uniform(0, 30, size=(n_residues, n_residues)).tolist()

        result = writer.write_cif(
            cif_bytes=noisy_cif_bytes,
            light_metadata={
                "name": f"parallel_worker_{worker_id}",
                "sample_idx": worker_id * samples_per_worker + i,
                "noise_seed": RANDOM_SEED + worker_id * 1000 + i,
                "rmsd_from_original": float(rmsd),
                "atom_count": n_atoms,
            },
            heavy_metadata={
                "pae": fake_pae,
                "plddt": rng.uniform(50, 100, size=n_residues).tolist(),
            },
        )
        written_ids.append(result["id_hex"])

    writer.finalize()
    return written_ids


# --------------- Tests ---------------

class TestProtlakeWriterBasic:
    """Basic tests for ProtlakeWriter initialization and schema."""
    
    def test_schema_merge(self, user_schema):
        """Test that core fields are properly merged with user schema."""
        from protlake.write.writer import build_full_schema, get_core_schema_fields
        
        full_schema = build_full_schema(user_schema, include_json=True)
        core_fields = get_core_schema_fields(include_json=True)
        
        # Check core fields are present
        core_names = {f.name for f in core_fields}
        for name in core_names:
            assert name in full_schema.names, f"Core field {name} not in full schema"
        
        # Check user fields are present
        for field in user_schema:
            assert field.name in full_schema.names, f"User field {field.name} not in full schema"
    
    def test_schema_no_json(self, user_schema):
        """Test schema without JSON shard fields."""
        from protlake.write.writer import build_full_schema
        
        full_schema = build_full_schema(user_schema, include_json=False)
        
        assert "bcif_shard" in full_schema.names
        assert "json_shard" not in full_schema.names
    
    def test_writer_initialization(self, protlake_writer, output_dir):
        """Test that writer creates necessary directories."""
        assert (output_dir / "protlake_output" / "shards").exists()
        assert (output_dir / "protlake_output" / "delta").exists()


class TestProtlakeWriterWrite:
    """Tests for write operations."""
    
    def test_write_single_structure(self, protlake_writer, base_structure):
        """Test writing a single structure."""
        cif_bytes = structure_to_cif_bytes(base_structure)
        
        result = protlake_writer.write_cif(
            cif_bytes=cif_bytes,
            light_metadata={
                "name": "test_single",
                "sample_idx": 0,
                "noise_seed": 0,
                "rmsd_from_original": 0.0,
                "atom_count": len(base_structure),
            },
            heavy_metadata={"pae": [[1.0, 2.0], [2.0, 1.0]]},
        )
        
        assert "id_hex" in result
        assert len(result["id_hex"]) == 64  # sha256 hex
        assert "bcif_shard" in result
        assert "bcif_off" in result
    
    def test_write_multiple_noisy_structures(self, protlake_writer, base_structure):
        """Write 100 noisy variants of the base structure."""
        rng = np.random.default_rng(RANDOM_SEED)
        
        written_ids = []
        for i in range(NUM_SAMPLES):
            # Add noise
            noisy_structure = add_coordinate_noise(base_structure, NOISE_STD, rng)
            rmsd = calculate_rmsd(base_structure, noisy_structure)
            
            # Convert to CIF bytes
            cif_bytes = structure_to_cif_bytes(noisy_structure)
            
            # Generate fake heavy metadata (simulating PAE matrix)
            n_residues = len(set(noisy_structure.res_id))
            fake_pae = rng.uniform(0, 30, size=(n_residues, n_residues)).tolist()
            
            result = protlake_writer.write_cif(
                cif_bytes=cif_bytes,
                light_metadata={
                    "name": "6Y7A_noisy",
                    "sample_idx": i,
                    "noise_seed": RANDOM_SEED + i,
                    "rmsd_from_original": float(rmsd),
                    "atom_count": len(noisy_structure),
                },
                heavy_metadata={
                    "pae": fake_pae,
                    "plddt": rng.uniform(50, 100, size=n_residues).tolist(),
                },
            )
            
            written_ids.append(result["id_hex"])
        
        # Flush to ensure all data is written
        protlake_writer.flush()
        
        # Verify all IDs are unique (different noise = different structure = different hash)
        assert len(set(written_ids)) == NUM_SAMPLES, "Expected all structures to have unique IDs"
    
    def test_write_without_heavy_metadata(self, protlake_writer, base_structure):
        """Test writing without heavy metadata."""
        cif_bytes = structure_to_cif_bytes(base_structure)
        
        result = protlake_writer.write_cif(
            cif_bytes=cif_bytes,
            light_metadata={
                "name": "test_no_heavy",
                "sample_idx": 999,
                "noise_seed": 0,
                "rmsd_from_original": 0.0,
                "atom_count": len(base_structure),
            },
            heavy_metadata=None,
        )
        
        assert "id_hex" in result
        protlake_writer.flush()


class TestProtlakeWriterWriteParallel:
    """Test writing from multiple processes to check for race conditions."""

    NUM_PROCESSES = 16
    SAMPLES_PER_PROCESS = 25

    def test_parallel_writes(self, output_dir, user_schema, base_structure):
        """Spawn 16 processes that simultaneously write noised structures."""
        import multiprocessing as mp
        from functools import partial

        parallel_output = output_dir / "parallel_test"

        # Prepare base structure data for serialization to child processes
        cif_bytes_base = structure_to_cif_bytes(base_structure)
        n_atoms = len(base_structure)

        # Serialize schema for passing to child processes
        sink = pa.BufferOutputStream()
        writer_schema = pa.ipc.new_stream(sink, user_schema)
        writer_schema.close()
        schema_bytes = sink.getvalue().to_pybytes()

        # Spawn processes
        with mp.Pool(processes=self.NUM_PROCESSES) as pool:
            worker_fn = partial(
                _parallel_worker_write,
                out_path=str(parallel_output),
                user_schema_bytes=schema_bytes,
                base_cif_bytes=cif_bytes_base,
                n_atoms=n_atoms,
                samples_per_worker=self.SAMPLES_PER_PROCESS,
            )
            results = pool.map(worker_fn, range(self.NUM_PROCESSES))

        # Collect all written IDs
        all_ids = []
        for worker_ids in results:
            all_ids.extend(worker_ids)

        expected_total = self.NUM_PROCESSES * self.SAMPLES_PER_PROCESS

        # Verify all IDs are unique
        assert len(set(all_ids)) == expected_total, (
            f"Expected {expected_total} unique IDs, got {len(set(all_ids))}"
        )

        # Verify data in delta table
        from protlake.write.writer import ProtlakeWriter, ProtlakeWriterConfig

        cfg = ProtlakeWriterConfig(
            out_path=str(parallel_output),
            user_schema=user_schema,
            write_json_shards=True,
        )
        reader = ProtlakeWriter(cfg)

        table = reader.query()
        assert table.num_rows == expected_total, (
            f"Expected {expected_total} rows in delta table, got {table.num_rows}"
        )

        # Verify each worker's samples are present
        for worker_id in range(self.NUM_PROCESSES):
            filter_expr = ds.field("name") == f"parallel_worker_{worker_id}"
            worker_table = reader.query(filter_expr=filter_expr)
            assert worker_table.num_rows == self.SAMPLES_PER_PROCESS, (
                f"Worker {worker_id}: expected {self.SAMPLES_PER_PROCESS} rows, "
                f"got {worker_table.num_rows}"
            )


class TestProtlakeWriterQuery:
    """Tests for query operations."""
    
    def test_check_exists_found(self, protlake_writer):
        """Test check_exists returns True for existing entries."""
        protlake_writer.flush()
        
        # Query for the noisy samples we wrote
        exists = protlake_writer.check_exists({"name": "6Y7A_noisy"})
        
        assert exists is True
    
    def test_check_exists_not_found(self, protlake_writer):
        """Test check_exists returns False for non-existing entries."""
        exists = protlake_writer.check_exists({"name": "nonexistent_structure"})
        
        assert exists is False
    
    def test_query_all(self, protlake_writer):
        """Test querying all entries."""
        protlake_writer.flush()
        
        table = protlake_writer.query()
        
        # Should have at least the 100 noisy + 2 test structures
        assert table.num_rows >= NUM_SAMPLES + 2
    
    def test_query_with_filter(self, protlake_writer):
        """Test querying with a filter."""
        protlake_writer.flush()
        
        filter_expr = ds.field("name") == "6Y7A_noisy"
        table = protlake_writer.query(filter_expr=filter_expr)
        
        assert table.num_rows == NUM_SAMPLES
    
    def test_query_with_columns(self, protlake_writer):
        """Test querying specific columns."""
        protlake_writer.flush()
        
        columns = ["name", "sample_idx", "rmsd_from_original"]
        table = protlake_writer.query(columns=columns)
        
        assert set(table.column_names) == set(columns)
    
    def test_query_with_filter_and_columns(self, protlake_writer):
        """Test querying with filter and column selection."""
        protlake_writer.flush()
        
        filter_expr = (ds.field("name") == "6Y7A_noisy") & (ds.field("sample_idx") < 10)
        columns = ["name", "sample_idx", "rmsd_from_original"]
        table = protlake_writer.query(filter_expr=filter_expr, columns=columns)
        
        assert table.num_rows == 10
        assert set(table.column_names) == set(columns)
    
    def test_query_rmsd_values(self, protlake_writer):
        """Verify RMSD values are approximately as expected from noise."""
        protlake_writer.flush()
        
        filter_expr = ds.field("name") == "6Y7A_noisy"
        table = protlake_writer.query(filter_expr=filter_expr, columns=["rmsd_from_original"])
        
        rmsd_values = table.column("rmsd_from_original").to_pylist()
        mean_rmsd = np.mean(rmsd_values)
        
        # With 0.1 Å std noise, RMSD should be around 0.1 * sqrt(3) ≈ 0.17 Å
        expected_rmsd = NOISE_STD * np.sqrt(3)
        assert 0.05 < mean_rmsd < 0.5, f"Mean RMSD {mean_rmsd} outside expected range"


class TestProtlakeWriterCheckComplete:
    """Tests for check_complete method."""
    
    def test_check_complete_all_found(self, protlake_writer):
        """Test check_complete returns True when all expected keys exist."""
        protlake_writer.flush()
        
        # Build expected keys matching what we wrote (100 noisy samples)
        expected = [
            {"name": "6Y7A_noisy", "sample_idx": i, "noise_seed": RANDOM_SEED + i}
            for i in range(NUM_SAMPLES)
        ]
        
        complete, found, missing = protlake_writer.check_complete(expected)
        
        assert complete is True
        assert len(found) == NUM_SAMPLES
        assert len(missing) == 0
    
    def test_check_complete_partial_missing(self, protlake_writer):
        """Test check_complete detects missing entries."""
        protlake_writer.flush()
        
        # Include some keys that don't exist (sample_idx 100-109)
        expected = [
            {"name": "6Y7A_noisy", "sample_idx": i, "noise_seed": RANDOM_SEED + i}
            for i in range(95, 110)  # 95-99 exist, 100-109 don't
        ]
        
        complete, found, missing = protlake_writer.check_complete(expected)
        
        assert complete is False
        assert len(found) == 5  # samples 95-99
        assert len(missing) == 10  # samples 100-109
    
    def test_check_complete_all_missing(self, protlake_writer):
        """Test check_complete when no expected keys exist."""
        protlake_writer.flush()
        
        expected = [
            {"name": "nonexistent_structure", "sample_idx": i, "noise_seed": 0}
            for i in range(5)
        ]
        
        complete, found, missing = protlake_writer.check_complete(expected)
        
        assert complete is False
        assert len(found) == 0
        assert len(missing) == 5
    
    def test_check_complete_empty_expected(self, protlake_writer):
        """Test check_complete with empty expected list."""
        protlake_writer.flush()
        
        with pytest.raises(ValueError, match="expected_keys must be a non-empty list of dicts."):
            complete, found, missing = protlake_writer.check_complete([])
    
    def test_check_complete_explicit_key_columns(self, protlake_writer):
        """Test check_complete with explicit key_columns parameter."""
        protlake_writer.flush()
        
        # Use only name and sample_idx as keys
        expected = [
            {"name": "6Y7A_noisy", "sample_idx": i}
            for i in range(10)
        ]
        
        complete, found, missing = protlake_writer.check_complete(
            expected, 
            key_columns=["name", "sample_idx"]
        )
        
        assert complete is True
        assert len(found) == 10
    
    def test_check_complete_on_empty_table(self, output_dir, user_schema):
        """Test check_complete on non-existent delta table."""
        from protlake.write.writer import ProtlakeWriter, ProtlakeWriterConfig
        
        empty_path = output_dir / "check_complete_empty_test"
        cfg = ProtlakeWriterConfig(
            out_path=str(empty_path),
            user_schema=user_schema,
            write_json_shards=False,
        )
        
        writer = ProtlakeWriter(cfg)
        
        expected = [{"name": "test", "sample_idx": 0, "noise_seed": 0}]
        complete, found, missing = writer.check_complete(expected)
        
        assert complete is False
        assert len(found) == 0
        assert len(missing) == 1


class TestProtlakeWriterDelete:
    """Tests for delete operations."""
    
    def test_delete_entries(self, protlake_writer):
        """Test deleting entries by predicate."""
        protlake_writer.flush()
        
        # First verify entry exists
        assert protlake_writer.check_exists({"name": "test_single"})
        
        # Delete it
        protlake_writer.delete("name = 'test_single'")
        
        # Verify it's gone
        assert not protlake_writer.check_exists({"name": "test_single"})
    
    def test_delete_nonexistent(self, protlake_writer):
        """Test deleting non-existent entries doesn't error."""
        # Should not raise
        protlake_writer.delete("name = 'definitely_does_not_exist'")


class TestProtlakeWriterEdgeCases:
    """Edge case and error handling tests."""
    
    def test_row_count(self, protlake_writer, base_structure):
        """Test row_count reflects buffered rows."""
        # Flush first to start clean
        protlake_writer.flush()
        initial_count = protlake_writer.row_count()
        assert initial_count == 0
        
        # Write a structure
        cif_bytes = structure_to_cif_bytes(base_structure)
        protlake_writer.write_cif(
            cif_bytes=cif_bytes,
            light_metadata={
                "name": "test_row_count",
                "sample_idx": 0,
                "noise_seed": 0,
                "rmsd_from_original": 0.0,
                "atom_count": len(base_structure),
            },
        )
        
        assert protlake_writer.row_count() == 1
        
        protlake_writer.flush()
        assert protlake_writer.row_count() == 0
    
    def test_empty_query_before_any_writes(self, output_dir, user_schema):
        """Test query on empty/non-existent delta table."""
        from protlake.write.writer import ProtlakeWriter, ProtlakeWriterConfig
        
        empty_path = output_dir / "empty_test"
        cfg = ProtlakeWriterConfig(
            out_path=str(empty_path),
            user_schema=user_schema,
            write_json_shards=False,
        )
        
        writer = ProtlakeWriter(cfg)
        
        # Query empty table
        table = writer.query()
        assert table.num_rows == 0
        
        # Check exists on empty table
        assert not writer.check_exists({"name": "anything"})


class TestProtlakeWriterWithoutJsonShards:
    """Test writer configured without JSON shards."""
    
    def test_write_without_json_shards(self, output_dir, user_schema, base_structure):
        """Test writing when write_json_shards=False."""
        from protlake.write.writer import ProtlakeWriter, ProtlakeWriterConfig
        
        no_json_path = output_dir / "no_json_test"
        cfg = ProtlakeWriterConfig(
            out_path=str(no_json_path),
            user_schema=user_schema,
            write_json_shards=False,
        )
        
        writer = ProtlakeWriter(cfg)
        cif_bytes = structure_to_cif_bytes(base_structure)
        
        result = writer.write_cif(
            cif_bytes=cif_bytes,
            light_metadata={
                "name": "test_no_json",
                "sample_idx": 0,
                "noise_seed": 0,
                "rmsd_from_original": 0.0,
                "atom_count": len(base_structure),
            },
            heavy_metadata={"ignored": "data"},  # Should be ignored
        )
        
        assert "id_hex" in result
        writer.flush()
        
        # Verify schema doesn't have json fields
        assert "json_shard" not in writer.full_schema.names


# --------------- Run tests directly ---------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
