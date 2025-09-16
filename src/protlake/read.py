import os, io, zlib
from biotite.structure.io.pdbx import BinaryCIFFile, get_structure
import zstandard as zstd
import msgpack

def _bcif_bytes_to_atom_array(bcif_data: bytes):
    f = io.BytesIO(bcif_data)
    bcif = BinaryCIFFile.read(f)
    return get_structure(bcif, extra_fields=['b_factor'], model=1)

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

def pread_json_msgpack_to_dict(path, offset, length):
    json_msgpack_bytes_comp = read_bytes_from_shard(path, offset, length)
    decompressor = zstd.ZstdDecompressor()
    json_msgpack_bytes = decompressor.decompress(json_msgpack_bytes_comp)
    return msgpack.unpackb(json_msgpack_bytes, raw=False)

