import os
from deltalake import DeltaTable
import pyarrow.parquet as pq

def deltatable_maintenance(dt, target_size = 1 << 28, max_concurrent_tasks=2):
    dt.alter.set_table_properties({"delta.logRetentionDuration": "interval 0 days"})
    dt.optimize.z_order(["name"], target_size=target_size, max_concurrent_tasks=max_concurrent_tasks) # ~256 MB per file
    # if too slow, just do compact instread of z_order, idea is to keep the names together
    # dt.optimize.compact(target_size=target_size, max_concurrent_tasks=max_concurrent_tasks)
    dt.vacuum(retention_hours=0, enforce_retention_duration=False, dry_run=False)
    dt.create_checkpoint()
    dt.cleanup_metadata()

def get_protlake_dirs(out_path):
    shard_dir = os.path.join(out_path, "shards")
    delta_path = os.path.join(out_path, "delta")
    return shard_dir, delta_path

def get_protlake_dirs_old(out_path):
    shard_dir = os.path.join(out_path, "shards")
    delta_core_path = os.path.join(out_path, "delta", "meta_core")
    delta_heavy_path = os.path.join(out_path, "delta", "meta_heavy")
    return shard_dir, delta_core_path, delta_heavy_path

def ensure_dirs(list_of_dirs):
    for path in list_of_dirs:
        os.makedirs(path, exist_ok=True)

def DeltaTable_nrow(dt: DeltaTable):
    count = 0
    for f in dt.file_uris():
        count += pq.ParquetFile(f).metadata.num_rows
    return count