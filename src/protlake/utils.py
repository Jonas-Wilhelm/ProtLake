import os, io, zlib
from deltalake import DeltaTable
import pyarrow.parquet as pq
import numpy as np
import zstandard as zstd
import msgpack
import pyarrow as pa
import pyarrow.dataset as ds
from biotite.structure.atoms import coord
from biotite.structure.util import vector_dot
from biotite.structure.io.pdbx import (
    BinaryCIFFile, 
    get_structure, 
    set_structure, 
    CIFFile, 
    compress
)
from biotite.structure.io import (
    save_structure, 
    load_structure
)

def deltatable_maintenance(dt, target_size = 1 << 28, max_concurrent_tasks=2):
    dt.alter.set_table_properties({"delta.logRetentionDuration": "interval 0 days"})
    dt.optimize.z_order(["name"], target_size=target_size, max_concurrent_tasks=max_concurrent_tasks) # ~256 MB per file
    # if too slow, just do compact instread of z_order, idea is to keep the names together
    # dt.optimize.compact(target_size=target_size, max_concurrent_tasks=max_concurrent_tasks)
    dt.vacuum(retention_hours=0, enforce_retention_duration=False, dry_run=False)
    dt.cleanup_metadata()
    dt.create_checkpoint()

def get_protlake_dirs(out_path):
    shard_dir = os.path.join(out_path, "shards")
    delta_path = os.path.join(out_path, "delta")
    return shard_dir, delta_path

def ensure_dirs(list_of_dirs):
    for path in list_of_dirs:
        os.makedirs(path, exist_ok=True)

def DeltaTable_nrow(dt: DeltaTable):
    count = 0
    for f in dt.file_uris():
        count += pq.ParquetFile(f).metadata.num_rows
    return count

def _sq_euclidian(reference, subject):
    '''
    Squared Euclidian distance function from biotite
    '''
    reference_coord = coord(reference)
    subject_coord = coord(subject)
    if reference_coord.ndim != 2:
        raise TypeError(
            "Expected an AtomArray or an ndarray with shape (n,3) as reference"
        )
    dif = subject_coord - reference_coord
    return vector_dot(dif, dif)

def rmsd_sc_automorphic(reference, subject):
    """
    Compute the per-residue all-atom RMSD between two structures, accounting for
    symmetry in selected sidechain atoms (e.g., OD1/OD2 in ASP). Returns the overall 
    RMSD across all residues.
    """
    # TODO make this work with non complete residues (e.g. only last 3 atoms of PHE)
    SYMMETRY_MAP = {
        "ASP": [("OD1","OD2")],
        "GLU": [("OE1","OE2")],
        "PHE": [("CD1","CD2"), ("CE1","CE2")],
        "TYR": [("CD1","CD2"), ("CE1","CE2")],
        "ARG": [("NH1","NH2")],
        "LEU": [("CD1","CD2")],
        "VAL": [("CG1","CG2")],
    }
    chain = reference.chain_id.astype(str)
    res   = reference.res_id.astype(str)
    keys  = chain + ":" + res

    boundaries = np.where(keys[:-1] != keys[1:])[0] + 1
    groups = np.split(np.arange(reference.shape[0]), boundaries)

    all_sd = np.array([], dtype=np.float32)
    for atom_idx in groups:
        res_name = reference[atom_idx[0]].res_name
        sd = _sq_euclidian(reference[atom_idx], subject[atom_idx])
        msd = np.mean(sd, axis=-1)
        if res_name in SYMMETRY_MAP.keys():
            atom_idx_swapped = atom_idx.copy()
            atom_pairs = SYMMETRY_MAP[res_name]
            for atom_pair in atom_pairs:
                # check if both atoms in pair are present
                if not (np.any(reference[atom_idx].atom_name == atom_pair[0]) and np.any(reference[atom_idx].atom_name == atom_pair[1])):
                    continue
                # swap their indices
                idx_a = np.where(reference[atom_idx].atom_name == atom_pair[0])[0][0]
                idx_b = np.where(reference[atom_idx].atom_name == atom_pair[1])[0][0]
                atom_idx_swapped[idx_a], atom_idx_swapped[idx_b] = atom_idx_swapped[idx_b], atom_idx_swapped[idx_a]
            sd_flip = _sq_euclidian(reference[atom_idx], subject[atom_idx_swapped])
            msd_flip = np.mean(sd_flip, axis=-1)
            if msd_flip < msd:
                sd = sd_flip
                msd = msd_flip
        all_sd = np.append(all_sd, sd)

    return np.sqrt(np.mean(all_sd, axis=-1))

def _bcif_bytes_to_atom_array(bcif_data: bytes):
    f = io.BytesIO(bcif_data)
    bcif = BinaryCIFFile.read(f)
    return get_structure(bcif, extra_fields=['b_factor'], model=1)

def _bcif_bytes_to_mmCIF_file(bcif_data: bytes, path: str):
    f = io.BytesIO(bcif_data)
    bcif = BinaryCIFFile.read(f)
    atoms = get_structure(bcif, extra_fields=['b_factor'], model=1)
    cif = CIFFile()
    set_structure(cif, atoms)
    with open(path, 'w') as out:
        cif.write(out)
    return True

def pread_bytes(path, offset, length):
    fd = os.open(path, os.O_RDONLY)
    try:
        return os.pread(fd, length, offset)
    finally:
        os.close(fd)

def read_bytes_from_shard(path, data_off, data_len, check_crc=True):
    ''' Read bytes with CRC validation '''
    fd = os.open(path, os.O_RDONLY)
    try:
        # read payload
        payload = os.pread(fd, data_len, data_off)
        if not check_crc:
            return payload
        
        # read stored CRC
        crc_bytes = os.pread(fd, 4, data_off + data_len)
        crc_stored = int.from_bytes(crc_bytes, "big")

        # validate
        crc_calc = zlib.crc32(payload) & 0xFFFFFFFF
        if crc_calc != crc_stored:
            raise IOError("CRC mismatch")

        return payload
    finally:
        os.close(fd)

def pread_bcif_to_atom_array(path, offset, length):
    bcif_data = read_bytes_from_shard(path, offset, length)
    return _bcif_bytes_to_atom_array(bcif_data)

def bcif_shard_to_mmCIF_file(shard_path, offset, length, out_path):
    bcif_data = read_bytes_from_shard(shard_path, offset, length)
    return _bcif_bytes_to_mmCIF_file(bcif_data, out_path)

def pread_json_msgpack_to_dict(path, offset, length):
    json_msgpack_bytes_comp = read_bytes_from_shard(path, offset, length)
    decompressor = zstd.ZstdDecompressor()
    json_msgpack_bytes = decompressor.decompress(json_msgpack_bytes_comp)
    return msgpack.unpackb(json_msgpack_bytes, raw=False)

def cif_to_bcif_bytes(cif_path: str, rtol: float = 1e-6, atol: float = 1e-4) -> bytes:
    """ Converts a mmCIF file (e.g. from AF3) to binary CIF (bcif) format."""
    atom_array = load_structure(cif_path, extra_fields=['b_factor'])
    bcif = BinaryCIFFile()
    set_structure(bcif, atom_array)
    bcif = compress(bcif, rtol=rtol, atol=atol)
    buf = io.BytesIO()
    bcif.write(buf)
    return buf.getvalue()

def cif_bytes_to_bcif_bytes(cif_data: bytes, rtol: float = 1e-6, atol: float = 1e-4) -> bytes:
    """ Converts utf-8 encoded mmCIF bytes (e.g. from AF3) to binary CIF (bcif) format."""
    f = io.StringIO(cif_data.decode('utf-8'))
    cif = CIFFile.read(f)
    atom_array = get_structure(cif, extra_fields=['b_factor'], model=1)
    bcif = BinaryCIFFile()
    set_structure(bcif, atom_array)
    bcif = compress(bcif, rtol=rtol, atol=atol)
    buf = io.BytesIO()
    bcif.write(buf)
    return buf.getvalue()

# def get_row_from_deltalake_simple(dt, row_dict):
#     ''' Get a single row from a Delta Lake table matching the criteria in row_dict'''
#     dataset = dt.to_pyarrow_dataset()

#     # Filter expression
#     expr = None
#     for col, val in row_dict.items():
#         cond = ds.field(col) == pa.scalar(val)
#         expr = cond if expr is None else (expr & cond)

#     # Scan with predicate pushdown
#     table = dataset.to_table(filter=expr)

#     if table.num_rows == 0:
#         return None

#     # Since we expect exactly one row, take the first
#     return table.to_pylist()[0]

def get_row_from_deltalake(dt, row_dict):
    ''' Get a single row from a Delta Lake table matching the criteria in row_dict'''
    dataset = dt.to_pyarrow_dataset()

    # build dataset expression
    expr = None
    for col, val in row_dict.items():
        cond = ds.field(col) == pa.scalar(val)
        expr = cond if expr is None else (expr & cond)

    # Use a Scanner and stream batches; return the first matching row immediately
    scanner = ds.Scanner.from_dataset(dataset, filter=expr)
    for batch in scanner.to_batches():
        if batch.num_rows:
            pyd = batch.to_pydict()
            # return the first row from this batch as a dict
            return {c: pyd[c][0] for c in pyd}
    return None

def dump_cif_from_deltalake_row(dt, row_dict, out_path='dump.cif', shard_dir=None):
    ''' Dump a CIF file from a Delta Lake table given a row_dict to identify the row'''
    if shard_dir is None:
        file_uri = dt.file_uris()[0]
        shard_dir = os.path.join(os.path.dirname(os.path.dirname(file_uri)), 'shards')
    row = get_row_from_deltalake(dt, row_dict)
    if row is None:
        raise ValueError("No matching row found in Delta Lake table")

    atom_array = pread_bcif_to_atom_array(os.path.join(shard_dir, row['bcif_shard']), row['bcif_data_off'], row['bcif_len'])
    save_structure(out_path, atom_array)

    return True

def dump_random_cif_from_deltalake(dt, out_path='dump.cif', shard_dir=None):
    ''' Dump a random CIF file from a Delta Lake table'''
    if shard_dir is None:
        file_uri = dt.file_uris()[0]
        shard_dir = os.path.join(os.path.dirname(os.path.dirname(file_uri)), 'shards')
    dataset = dt.to_pyarrow_dataset()
    scanner = ds.Scanner.from_dataset(dataset, batch_size=1)
    for batch in scanner.to_batches():
        if batch.num_rows:
            pyd = batch.to_pydict()
            # take the first row from this batch as a dict
            row = {c: pyd[c][0] for c in pyd}
            atom_array = pread_bcif_to_atom_array(os.path.join(shard_dir, row['bcif_shard']), row['bcif_data_off'], row['bcif_len'])
            save_structure(out_path, atom_array)
            return True
    raise ValueError("No rows found in Delta Lake table")