#!/usr/bin/env python
import os, time, sys, shutil, zstandard as zstd, msgpack, json
from datetime import timedelta
import pyarrow as pa
from deltalake import DeltaTable
sys.path.append("/home/jonaswil/Software/ProtLake")
from protlake.utils import deltatable_maintenance, get_protlake_dirs
from protlake.af3.ingest import IngestConfig, AF3IngestPipeline
from protlake.read import pread_bytes, bcif_bytes_to_atom_array
from biotite.structure.io.pdbx import BinaryCIFFile, CIFFile, set_structure, compress, get_structure


TESTDATA_DIR = "tests/af3/testdata_af3_output"
OUT_DIR = "tests_out/af3/af3_to_protlake/protlake"
OUT_DIR_EXTRACT_TEST = "tests_out/af3/af3_to_protlake/extract_test"
BATCH_SIZE = 100

def main():
    # ----------- writing -----------
    start_time = time.time()

    input_dirs = [os.path.join(TESTDATA_DIR, d) for d in os.listdir(TESTDATA_DIR)]

    cfg = IngestConfig(out_path=OUT_DIR, batch_size_metadata=BATCH_SIZE, shard_size=1 << 30, verbose=False)
    AF3IngestPipeline(cfg).run(input_dirs)

    end_time = time.time()
    # print elapsed time in hour, minutes, seconds
    elapsed = end_time - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Processing took: {int(hours)}h {int(minutes)}m {int(seconds)}s")


    # ----------- maintenance -----------
    shard_path, delta_path = get_protlake_dirs(OUT_DIR)
    dt = DeltaTable(f"file://{os.path.abspath(delta_path)}")
    deltatable_maintenance(dt)

    mtnc_time = time.time()
    # print elapsed time in hour, minutes, seconds
    elapsed = mtnc_time - end_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Delta lake maintenance took: {int(hours)}h {int(minutes)}m {int(seconds)}s")


    # ----------- reading -----------
    df = dt.to_pandas()
    os.makedirs(OUT_DIR_EXTRACT_TEST, exist_ok=True)
    # generate list with random indices within number of rows in df_core
    import random
    random_indices = random.sample(range(len(df)), 5)
    random_indices.sort()

    for i in random_indices:
        aa = bcif_bytes_to_atom_array(pread_bytes(os.path.join(shard_path, df.iloc[i]["bcif_shard"]), df.iloc[i]["bcif_off"], df.iloc[i]["bcif_len"]))
        cif = CIFFile()
        set_structure(cif, aa)
        # write cif to a file called out_test_read.cif
        cif.write(os.path.join(OUT_DIR_EXTRACT_TEST, f"{df.iloc[i]['name']}__{df.iloc[i]['seed']}__{df.iloc[i]['sample']}__from_db.cif"))
        # copy the corresponding cif from af3_out_path to dir_test_read
        af3_cif_path = os.path.join(TESTDATA_DIR, df.iloc[i]["name"], f"seed-{df.iloc[i]['seed']}_sample-{df.iloc[i]['sample']}", f"{df.iloc[i]['name']}_seed-{df.iloc[i]['seed']}_sample-{df.iloc[i]['sample']}_model.cif")
        if os.path.exists(af3_cif_path):
            shutil.copy(af3_cif_path, os.path.join(OUT_DIR_EXTRACT_TEST, f"{df.iloc[i]['name']}__{df.iloc[i]['seed']}__{df.iloc[i]['sample']}__from_af3.cif"))
        
        conf = pread_bytes(os.path.join(shard_path, df.iloc[i]["json_shard"]), df.iloc[i]["json_off"], df.iloc[i]["json_len"])
        decompressor = zstd.ZstdDecompressor()
        raw = decompressor.decompress(conf)
        conf_json = msgpack.unpackb(raw, raw=False)
        with open(os.path.join(OUT_DIR_EXTRACT_TEST, f"{df.iloc[i]['name']}__{df.iloc[i]['seed']}__{df.iloc[i]['sample']}_conf__from_db.json"), "w") as f:
            json.dump(conf_json, f, indent=4)


if __name__ == "__main__":
    main()
