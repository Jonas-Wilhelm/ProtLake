import os, io, shutil
from deltalake import DeltaTable
import pyarrow as pa
import pyarrow.compute as pc
from biotite.structure.io import load_structure
from biotite.structure.io.pdbx import BinaryCIFFile, CIFFile, set_structure, compress, get_structure
from utils import get_protlake_dirs, deltatable_maintenance
import pyarrow.parquet as pq
import zstandard as zstd
import msgpack

def pread_bytes(path, offset, length):
    fd = os.open(path, os.O_RDONLY)
    try:
        return os.pread(fd, length, offset)
    finally:
        os.close(fd)

def pread_bcif_to_atom_array(path, offset, length):
    bcif_data = pread_bytes(path, offset, length)
    return bcif_bytes_to_atom_array(bcif_data)

def pread_json_msgpack_to_dict(path, offset, length):
    json_msgpack_bytes_comp = pread_bytes(path, offset, length)
    decompressor = zstd.ZstdDecompressor()
    json_msgpack_bytes = decompressor.decompress(json_msgpack_bytes_comp)
    return msgpack.unpackb(json_msgpack_bytes, raw=False)

def bcif_bytes_to_atom_array(bcif_data: bytes):
    f = io.BytesIO(bcif_data)
    bcif = BinaryCIFFile.read(f)
    return get_structure(bcif, extra_fields=['b_factor'], model=1)

def get_DeltaTable_nrows(dt: DeltaTable):
    count = 0
    for f in dt.file_uris():
        count += pq.ParquetFile(f).metadata.num_rows
    return count


# def _lookup_core_row(name: str, seed: int, sample: int):
#     dt = DeltaTable(f"file://{os.path.abspath(delta_core_path)}")
#     tbl = dt.to_pyarrow_table(columns=["name","seed","sample","bcif_shard","bcif_off","bcif_len"])

#     # Build comparisons (ChunkedArray-safe)
#     m1 = pc.equal(tbl["name"], pa.scalar(name, type=pa.string()))
#     m2 = pc.equal(tbl["seed"], pa.scalar(seed, type=pa.int32()))
#     m3 = pc.equal(tbl["sample"], pa.scalar(sample, type=pa.int32()))

#     # Combine with pc.and_ (or pc.and_kleene for null-aware logic)
#     mask = pc.and_(m1, pc.and_(m2, m3))

#     # Some Arrow versions need a single chunk for Table.filter()
#     if isinstance(mask, pa.ChunkedArray):
#         mask = mask.combine_chunks()

#     sub = tbl.filter(mask)
#     if sub.num_rows == 0:
#         print("sub.num_rows == 0")
#         return None
#     if sub.num_rows > 1:
#         sub = sub.slice(0, 1)

#     pdict = sub.to_pydict()
#     return pdict["bcif_shard"][0], int(pdict["bcif_off"][0]), int(pdict["bcif_len"][0])

# def get_bcif_data_for(name: str, seed: int, sample: int, as_atom_array: bool=False):
#     row = _lookup_core_row(name, seed, sample)
#     if row is None:
#         return None
#     shard, off, ln = row
#     bcif_bytes = pread_bytes(shard, off, ln)
#     if as_atom_array:
#         return bcif_bytes_to_atom_array(bcif_bytes)
#     return bcif_bytes

