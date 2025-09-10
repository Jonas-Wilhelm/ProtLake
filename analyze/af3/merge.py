import os, shutil
from deltalake import DeltaTable
from utils import deltatable_maintenance
import argparse
from utils import get_protlake_dirs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protlake-path", type=str, required=True,
                        help="path to the deltatable to merge into")
    parser.add_argument("--staging-path", type=str, required=True,
                        help="path to the staging deltatable")
    parser.add_argument("--dont-delete-staging-table", action="store_true", 
                        help="if set, the staging-path directory will not be deleted after merging")
    args = parser.parse_args()
    protlake_path = args.protlake_path
    staging_path = args.staging_path

    _, delta_path_main = get_protlake_dirs(protlake_path)

    dt = DeltaTable(f"file://{os.path.abspath(delta_path_main)}")

    dt_stage = DeltaTable(f"file://{os.path.abspath(staging_path)}")

    dt.alter.add_columns(
        dt_stage.schema().fields
    )

    staging_cols = [f.name for f in dt_stage.schema().fields]
    key_col = "id_hex"

    update_cols = {
        f"`{col}`": f"updates.`{col}`" 
        for col in staging_cols 
        if col != key_col
    }

    tbl_stage = dt_stage.to_pyarrow_table()

    dt.merge(
        tbl_stage,
        predicate="dt.id_hex = updates.id_hex",
        source_alias="updates",
        target_alias="dt"
    ).when_matched_update(updates=update_cols).execute()

    # do maintenance
    deltatable_maintenance(dt)

    # remove staging_path directory
    if not args.dont_delete_staging_table:
        shutil.rmtree(staging_path)

if __name__ == "__main__":
    main()

# predicate = "base.id = updates.id AND base.ingest_version <= updates.af3_analysis_source_version"