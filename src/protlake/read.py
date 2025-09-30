import os, io, zlib
from biotite.structure.io.pdbx import BinaryCIFFile, get_structure, set_structure, CIFFile
from biotite.structure.io import save_structure
import zstandard as zstd
import msgpack
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc

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