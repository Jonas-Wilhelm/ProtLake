#!/usr/bin/env python3

import os, time, sys, shutil
from deltalake import DeltaTable, write_deltalake
from deltalake.schema import Field, PrimitiveType, StructType
import pyarrow as pa
import pyarrow.compute as pc
import protlake
from ..utils import get_protlake_dirs, rmsd_sc_automorphic, DeltaTable_nrow, deltatable_maintenance
from ..read import pread_bcif_to_atom_array, pread_json_msgpack_to_dict
from biotite.structure import filter_peptide_backbone, superimpose, rmsd
from biotite.structure.io import load_structure, save_structure
from biotite.structure.info import standardize_order
from xxhash import xxh64
import numpy as np
import pandas as pd
import itertools
from pathlib import Path
import importlib.util
import argparse
from pathlib import Path

def _import_plugin_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def load_scorefxns(plugin_dir, parser=None, names=None):
    scorefxns = []
    for p in Path(plugin_dir).glob("*.py"):
        if p.name.startswith("_"): # for templates or deactivating 
            continue
        if names is not None:
            if p.stem not in names:
                continue
        m = _import_plugin_module(p)
        if hasattr(m, "score") and callable(m.score):
            scorefxns.append({
                "name": getattr(m, "scorefxn_name", m.__name__),
                "fn": m.score,
            })
            # add command line options to parser if plugin contains register_args
            if parser is not None and hasattr(m, "register_args"):
                m.register_args(parser)

    return scorefxns

def _to_pa_scalar(val):
    """Map a scalar Python/NumPy value to (pa_type, python_value) or return (None, None) if unsupported."""
    if val is None:
        # treat None as a null float (change if you prefer null string/int)
        return pa.int32(), None

    # numpy scalar handling
    if isinstance(val, (np.generic,)):
        if np.issubdtype(val.dtype, np.integer):
            return pa.int32(), int(val)
        if np.issubdtype(val.dtype, np.floating):
            return pa.float32(), float(val)
        if np.issubdtype(val.dtype, np.str_):
            return pa.string(), str(val)
        if np.issubdtype(val.dtype, np.bool):
            return pa.bool(), str(val)

    # plain python types
    if isinstance(val, int):
        return pa.int32(), int(val)
    if isinstance(val, float):
        return pa.float32(), float(val)
    if isinstance(val, str):
        return pa.string(), str(val)
    if isinstance(val, bool):
        return pa.bool_(), bool(val)

    # anything else (lists/arrays/objects) is unsupported here
    return None, None

def eval_scorefxns(scorefxns, aa_design, aa_af3, meta, confidences, sc_close_to_het_mask, CLI_args, staging_columns):
    for s in scorefxns:
        name = s["name"]
        fn = s["fn"]
        out = fn(aa_design, aa_af3, meta, confidences, sc_close_to_het_mask, CLI_args)
        if not isinstance(out, dict):
            print(f"[scorer:{name}] returned non-dict; skipping")
            continue

        for k, v in out.items():
            pa_type, pa_val = _to_pa_scalar(v)
            if pa_type is None:
                print(f"[scorer:{name}] key '{k}' returned unsupported type {type(v)}; skipping")
                continue

            staging_columns.append(k, pa_val, pa_type)

def _expand_mask_to_residues(mask, aa):
    """
    Expand an atom selection (bool mask or list/array of atom indices) to a boolean
    atom mask that selects *all atoms* of any residue (chain_id + res_id) that has
    >=1 atom in `mask`.
    """
    keys = np.char.add(aa.chain_id.astype(str), ":" + aa.res_id.astype(str))
    sel_keys = np.unique(keys[mask])
    return np.isin(keys, sel_keys)

def _filter_atoms_close_to_hetero(aa, dist = 7.5):
    if not np.any(aa.hetero):
        return np.zeros(aa.shape[0], dtype=bool)
    
    coords = aa.coord
    het_coords = coords[aa.hetero]
    dists = np.linalg.norm(coords[:, None, :] - het_coords[None, :, :], axis=2)
    near_mask = dists.min(axis=1) <= 7.5
    return (near_mask & ~aa.hetero)

def _standardize_order_not_hetero(aa):
    aa[~aa.hetero] = aa[standardize_order(aa[~aa.hetero])]
    return aa

class staging_col_dict:
    def __init__(self):
        self.main_dict = {"id_hex":   {"val": [], "type": pa.string()}}

    def append(self, key, val, type=pa.float32()):
        if key not in self.main_dict.keys():
            self.main_dict.update({key: {"val": [], "type": type}})

        self.main_dict[key]["val"].append(val)

    def get_dict(self):
        return self.main_dict
    
    def __repr__(self):
        return repr(self.main_dict)

def main():
    # bootstrap parser to parse custom score function directory
    default_scorefxn_dir = Path(protlake.__file__).resolve().parent / "af3" / "scorefxns"
    bootstrap_parser = argparse.ArgumentParser()
    bootstrap_parser.add_argument("--custom-scorefxn-dir", default=default_scorefxn_dir, help="Directory with score function plugins. default: protlake/af3/scorefxns")
    bootstrap_parser.add_argument("--scorefxns", type=str, nargs="+", default=None, help="Names of score functions to run (default: all in --custom-scorefxn-dir)")
    boot_args, remaining_argv = bootstrap_parser.parse_known_args()

    # main parser and load score functions
    parser = argparse.ArgumentParser(description="Run analysis")
    print("Loading score functions from:", boot_args.custom_scorefxn_dir)
    scorefxns = load_scorefxns(plugin_dir=Path(boot_args.custom_scorefxn_dir), parser=parser, names=boot_args.scorefxns)
    print("Loaded score functions:", [s["name"] for s in scorefxns])
    parser.add_argument("--protlake-path", type=str, required=True, help="Path to the Protlake directory to analyze")
    parser.add_argument("--snapshot-ver", type=int, required=True, help="DeltaLake snapshot version to use")
    parser.add_argument("--staging-path", type=str, required=True, help="Path to the staging directory, default: <protlake-path>/delta_staging_table")
    parser.add_argument("--design-dir", type=str, required=True, help="Path to the design directory")

    # parse arguments
    args = parser.parse_args(remaining_argv)
    main_protlake_path = args.protlake_path
    snapshot_ver = args.snapshot_ver
    design_dir = args.design_dir
    staging_path = args.staging_path

    # get slurm environment variables, if not present assume debug mode and set to single task
    if 'SLURM_ARRAY_TASK_COUNT' not in os.environ or 'SLURM_ARRAY_TASK_ID' not in os.environ:
        print("Warning: SLURM environment variables not found, assuming debug mode with single task")
        num_array_tasks = 1
        my_task_id = 0
    else:
        num_array_tasks = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
        my_task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])

    # connect to protlake
    shard_dir, delta_path = get_protlake_dirs(main_protlake_path)
    dt = DeltaTable(f"file://{os.path.abspath(delta_path)}", version=snapshot_ver)
    ds = dt.to_pyarrow_dataset()

    scanner = ds.scanner(
        # columns=["name"],       # project only what you need
        # filter=pc.field("col_c") < 9,              # pushdown what you can
        batch_size=1_000_000
    )

    # initialize staging columns dict
    staging_columns = staging_col_dict()
    
    # process each batch
    for batch in scanner.to_batches():
        start_time = time.time()
        names = batch["name"].to_numpy(zero_copy_only=False)
        order = np.argsort(names)
        s_names = names[order]
        breaks = np.flatnonzero(s_names[1:] != s_names[:-1]) + 1
        groups = np.split(order, breaks)                      # list of integer index arrays
        group_names = s_names[np.concatenate(([0], breaks))]  # name per group

        hashes = np.fromiter((xxh64(n).intdigest() for n in group_names), dtype=np.uint64)
        sel = (hashes % num_array_tasks) == my_task_id

        # time for hashing and selecting
        end_hash_time = time.time()
        print(f"Hashed and selected {sel.sum()} out of {len(sel)} groups in {end_hash_time - start_time:.3f} sec")

        for gi, keep in enumerate(sel):
            if not keep:
                continue
            name = group_names[gi]
            rows = groups[gi]
            # read in design as atom array
            design_path = os.path.join(design_dir, f"{name}.pdb")
            aa_design = load_structure(design_path)
            # remove hydrogens which are not in af3 output anyway
            aa_design = aa_design[aa_design.element != "H"]
            aa_design = _standardize_order_not_hetero(aa_design)
            # create bool selection masks
            CA_mask = (~aa_design.hetero & (aa_design.atom_name == "CA"))
            bb_mask = filter_peptide_backbone(aa_design)
            close_to_het_mask = _expand_mask_to_residues(
                _filter_atoms_close_to_hetero(aa_design, 7.5), 
                aa_design
            )
            sc_close_to_het_mask = (close_to_het_mask & ~bb_mask)
            for row in rows:
                # ------------ import structure and meta data ------------
                meta = batch.slice(row, 1).to_pylist()[0]
                
                aa_af3 = pread_bcif_to_atom_array(os.path.join(shard_dir, meta["bcif_shard"]), meta["bcif_off"], meta["bcif_len"])
                if aa_af3[~aa_af3.hetero].shape[0] != aa_design[~aa_design.hetero].shape[0]:
                    print(f"Warning: Different number of (non H) atoms in in design model and af3 prediction for {name}, sample {meta['sample']}, seed {meta['seed']}")
                    continue
                if not np.all(aa_af3.atom_name[~aa_af3.hetero] == aa_design.atom_name[~aa_design.hetero]):
                    print(f"Warning: Missmatch of atom names in in design model and af3 prediction for {name}, sample {meta['sample']}, seed {meta['seed']}")
                    continue

                aa_af3, _ = superimpose(aa_design, aa_af3, atom_mask=CA_mask)
                
                confidences = pread_json_msgpack_to_dict(os.path.join(shard_dir, meta["json_shard"]), meta["json_off"], meta["json_len"])
                # convert all numeric lists to numpy arrays
                confidences = {
                    k: np.asarray(v) if isinstance(v, list) and all(isinstance(x, (int, float, list)) for x in v) else v for k, v in confidences.items()
                }
                
                # ------------ calculate base metrics ------------
                staging_columns.append("id_hex", meta["id_hex"], pa.string())

                pLDDT_CA = np.mean(confidences["atom_plddts"][bb_mask], dtype=np.float32)
                staging_columns.append("pLDDT_CA", pLDDT_CA)

                pLDDT_AA = np.mean(confidences["atom_plddts"], dtype=np.float32)
                staging_columns.append("pLDDT_AA", pLDDT_AA)
                
                RMSD_CA = rmsd(aa_design[CA_mask], aa_af3[CA_mask])
                staging_columns.append("RMSD_CA", RMSD_CA)

                RMSD_AA = rmsd_sc_automorphic(aa_design[~aa_design.hetero], aa_af3[~aa_af3.hetero])
                staging_columns.append("RMSD_AA", RMSD_AA)

                if any(aa_af3.hetero):
                    pLDDT_SC_around_lig = np.mean(confidences["atom_plddts"][sc_close_to_het_mask], dtype=np.float32)
                    staging_columns.append("pLDDT_SC_around_lig", pLDDT_SC_around_lig)

                    RMSD_SC_around_lig = rmsd_sc_automorphic(aa_design[sc_close_to_het_mask], aa_af3[sc_close_to_het_mask])
                    staging_columns.append("RMSD_CS_around_lig", RMSD_SC_around_lig)

                    pLDDT_LIGs = {k: np.mean(confidences["atom_plddts"][aa_af3.res_name == k], dtype=np.float32) for k in np.unique(aa_af3[aa_af3.hetero].res_name)}
                    for key, val in pLDDT_LIGs.items():
                        staging_columns.append(f"pLDDT_LIG_{key}", val)
                

                # ------------ calculate user defiend metrics ------------
                eval_scorefxns(scorefxns, aa_design, aa_af3, meta, confidences, sc_close_to_het_mask, args, staging_columns)

        end_time = time.time()
        n_models = len(staging_columns.get_dict()['id_hex']['val'])
        print(f"Processed batch with {n_models} models in {end_time - start_time:.3f} sec")

        staging_table = pa.table(
            {
                **{name: pa.array(col["val"], type=col["type"]) for name, col in staging_columns.get_dict().items()},
                **{"af3_analysis_source_version": pa.array(np.full(len(staging_columns.get_dict()["id_hex"]["val"]), snapshot_ver, dtype=np.int64))}
            }
        )

        write_deltalake(staging_path, staging_table, mode="append")

if __name__ == "__main__":
    main()