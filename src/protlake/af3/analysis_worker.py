#!/usr/bin/env -S PYTHONUNBUFFERED=1 python

import os, time, sys
from deltalake import DeltaTable, write_deltalake
import pyarrow as pa
import pyarrow.compute as pc
import protlake
from protlake.utils import (
    get_protlake_dirs, 
    rmsd_sc_automorphic, 
    pread_bcif_to_atom_array, 
    pread_json_msgpack_to_dict
)
from biotite.structure import filter_peptide_backbone, superimpose, rmsd, AtomArray
from biotite.structure.io import load_structure, save_structure
from biotite.structure.info import standardize_order
from xxhash import xxh64
import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util
import argparse
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ScoreFunctionInput:
    aa_design: AtomArray
    aa_af3: AtomArray
    meta: dict
    confidences: dict
    sc_close_to_het_mask: np.ndarray
    CLI_args: argparse.Namespace

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
                "module": m,
            })
            print(f"Loaded score function plugin: {getattr(m, 'scorefxn_name', m.__name__)}")
            # add command line options to parser if plugin contains register_args
            if parser is not None and hasattr(m, "register_args"):
                m.register_args(parser)
        else:
            raise ValueError(f"Plugin module {p} does not contain a callable 'score' function")

    return scorefxns

def execute_score_function_initializers(scorefxns, args):
    for s in scorefxns:
        m = s["module"]
        if hasattr(m, "init") and callable(m.init):
            print(f"Executing init() for plugin {s['name']}")
            m.init(args)

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
        if np.issubdtype(val.dtype, np.bool_):
            return pa.bool_(), bool(val)

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

def eval_scorefxns(scorefxns, sfx_input, staging_columns):
    for s in scorefxns:
        name = s["name"]
        fn = s["fn"]
        out = fn(sfx_input)
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
    """
    Return a boolean mask selecting all non-hetero atoms that are within `dist` Angstroms
    of any hetero atom.
    """
    if not np.any(aa.hetero):
        return np.zeros(aa.shape[0], dtype=bool)
    
    coords = aa.coord
    het_coords = coords[aa.hetero]
    dists = np.linalg.norm(coords[:, None, :] - het_coords[None, :, :], axis=2)
    near_mask = dists.min(axis=1) <= dist
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
    

def replace_ligand_res_names(replacement_strings, aa_af3):
    for spec in replacement_strings:
        old_resname, new_resname = spec.split("=")
        aa_af3.res_name[aa_af3.res_name == old_resname] = new_resname


def replace_ligand_atom_names(replacement_strings, aa_af3):
    for spec in replacement_strings:
        resname_pair, atom_string = spec.split("=")
        resname_pair = tuple(resname_pair.split(":"))
        atom_pairs = [tuple(s.split(":")) for s in atom_string.split(",")]
        res_mask = (aa_af3.res_name == resname_pair[0])
        for atom_pair in atom_pairs:
            aa_af3.atom_name[res_mask & (aa_af3.atom_name == atom_pair[0])] = atom_pair[1]

        aa_af3.res_name[res_mask] = resname_pair[1]


def af3_array_setup(aa_af3, replace_lig_res_names, replace_lig_atom_names, OXT_present_design, confidences):
    if not OXT_present_design:
        OXT_mask = (aa_af3.atom_name == "OXT") & ~aa_af3.hetero
        aa_af3 = aa_af3[~OXT_mask]

    if not OXT_present_design and np.any(OXT_mask):
                    # remove OXT from confidence metrics as well
        confidences['atom_chain_ids'] = confidences['atom_chain_ids'][~OXT_mask]
        confidences['atom_plddts'] = confidences['atom_plddts'][~OXT_mask]

    if replace_lig_res_names is not None:
        replace_ligand_res_names(replace_lig_res_names, aa_af3)

    if replace_lig_atom_names is not None:
        replace_ligand_atom_names(replace_lig_atom_names, aa_af3)
    return aa_af3

def main():
    # bootstrap parser to parse custom score function directory
    default_scorefxn_dir = Path(protlake.__file__).resolve().parent / "af3" / "scorefxns"
    bootstrap_parser = argparse.ArgumentParser()
    bootstrap_parser.add_argument("--custom-scorefxn-dir", default=default_scorefxn_dir, help="Directory with score function plugins. default: protlake/af3/scorefxns")
    bootstrap_parser.add_argument("--scorefxns", type=str, nargs="+", default=None, 
                                  help="Names of score functions to run. " \
                                    "Default: all in --custom-scorefxn-dir")
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
    parser.add_argument("--ncaa", type=str, nargs='+', required=False, help="List of NCAA 3-letter CCD codes.")
    parser.add_argument("--replace_lig_res_names", type=str, required=False, nargs="+", 
                        help="Residue names to replace in the AF3 output. " \
                             "Format: OLD_RESNAME1=NEW_RESNAME1 OLD_RESNAME2=NEW_RESNAME2")
    parser.add_argument("--replace_lig_atom_names", type=str, required=False, nargs="+", 
                        help="Similar to --replace_lig_res_names but also enables atom name replacements. " \
                            "Format:  OLD_RESNAME1:NEW_RESNAME1=OLD_ATOM1:NEW_ATOM1,OLD_ATOM2:NEW_ATOM2. " \
                            "Example: LIG_B:LIG=C1:C5,C2:C6")

    # parse arguments
    args = parser.parse_args(remaining_argv)
    execute_score_function_initializers(scorefxns, args)
    main_protlake_path = args.protlake_path
    snapshot_ver = args.snapshot_ver
    design_dir = args.design_dir
    staging_path = args.staging_path
    replace_lig_res_names = args.replace_lig_res_names
    replace_lig_atom_names = args.replace_lig_atom_names

    if replace_lig_res_names is not None and replace_lig_atom_names is not None:
        raise ValueError("Cannot use both --replace_lig_res_names and --replace_lig_atom_names at the same time")

    # get slurm environment variables, if not present assume local mode and set to single task
    if 'SLURM_ARRAY_TASK_COUNT' not in os.environ or 'SLURM_ARRAY_TASK_ID' not in os.environ:
        print("Warning: SLURM environment variables not found, assuming local mode with single task")
        num_array_tasks = 1
        my_task_id = 0
    else:
        num_array_tasks = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
        my_task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
        print(f"Running in SLURM mode with {num_array_tasks} array tasks, this is task {my_task_id}")

    # connect to protlake
    shard_dir, delta_path = get_protlake_dirs(main_protlake_path)
    dt = DeltaTable(f"file://{os.path.abspath(delta_path)}", version=snapshot_ver)
    ds = dt.to_pyarrow_dataset()

    scanner = ds.scanner(
        # columns=["name"],       # project only what you need
        # filter=pc.field("col_c") < 9,              # pushdown what you can
        batch_size=100_000
    )
    
    # process each batch
    for batch in scanner.to_batches():    
        # initialize staging columns dict
        staging_columns = staging_col_dict()
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
            # if ncaa specified, set those residues to non-hetero
            if args.ncaa:
                for resn in args.ncaa:
                    aa_design.hetero[aa_design.res_name == resn] = False

            aa_design = _standardize_order_not_hetero(aa_design) # TODO also do for af3 output?
            OXT_present_design = np.any(aa_design.atom_name == "OXT")
            # create bool selection masks
            aa_design_prot = aa_design[~aa_design.hetero]
            CA_mask = (aa_design_prot.atom_name == "CA")
            bb_mask = filter_peptide_backbone(aa_design_prot)
            close_to_het_mask = _expand_mask_to_residues(
                _filter_atoms_close_to_hetero(aa_design, 7.5), 
                aa_design
            )
            sc_close_to_het_mask = (close_to_het_mask[~aa_design.hetero] & ~bb_mask)
            for row in rows:
                # ------------ import structure and meta data ------------
                meta = batch.slice(row, 1).to_pylist()[0]

                # STRUCTURE
                try:
                    aa_af3 = pread_bcif_to_atom_array(os.path.join(shard_dir, meta["bcif_shard"]), meta["bcif_data_off"], meta["bcif_len"])
                except Exception as e:
                    print(f"Error reading AF3 structure for {name}, sample {meta['sample']}, seed {meta['seed']}: {e}")
                    continue

                # CONFIDENCES
                try:
                    confidences = pread_json_msgpack_to_dict(os.path.join(shard_dir, meta["json_shard"]), meta["json_data_off"], meta["json_len"])
                except Exception as e:
                    print(f"Error reading confidences for {name}, sample {meta['sample']}, seed {meta['seed']}: {e}")
                    continue
                # convert lists to numpy arrays
                confidences = {
                    k: np.asarray(v) if isinstance(v, list) and all(isinstance(x, (int, float, list, str)) for x in v) else v for k, v in confidences.items()
                }
                
                # if ncaa specified, set those residues to non-hetero
                if args.ncaa:
                    for resn in args.ncaa:
                        aa_af3.hetero[aa_af3.res_name == resn] = False

                # remove OXT from AF3 output if not present in design model
                aa_af3 = af3_array_setup(aa_af3, replace_lig_res_names, replace_lig_atom_names, OXT_present_design, confidences)
                
                # check that design and af3 have same atoms in same order for non-hetero atoms
                aa_af3_prot = aa_af3[~aa_af3.hetero]
                if aa_af3_prot.shape[0] != aa_design_prot.shape[0]:
                    print(f"Warning: Different number of (non H) atoms in in design model and af3 prediction for {name}, sample {meta['sample']}, seed {meta['seed']}")
                    continue
                if not np.all(aa_af3_prot.atom_name == aa_design_prot.atom_name):
                    print(f"Warning: Missmatch of atom names in in design model and af3 prediction for {name}, sample {meta['sample']}, seed {meta['seed']}")
                    continue

                # first calculate transformation to align AF3 to design based on non-hetero CA atoms (to allow ligand differences)
                _, transformation = superimpose(
                    aa_design_prot, 
                    aa_af3_prot, 
                    atom_mask=CA_mask
                )
                # apply transformation to whole AF3 structure
                aa_af3 = transformation.apply(aa_af3)
                aa_af3_prot = aa_af3[~aa_af3.hetero] # also update protein-only view

                # ------------ calculate base metrics ------------
                staging_columns.append("id_hex", meta["id_hex"], pa.string())

                pLDDT_CA = np.mean(confidences["atom_plddts"][aa_af3.atom_name == "CA"], dtype=np.float32)
                staging_columns.append("pLDDT_CA", pLDDT_CA)

                pLDDT_AA = np.mean(confidences["atom_plddts"], dtype=np.float32)
                staging_columns.append("pLDDT_AA", pLDDT_AA)
                
                RMSD_CA = rmsd(aa_design_prot[CA_mask], aa_af3_prot[CA_mask])
                staging_columns.append("RMSD_CA", RMSD_CA)

                RMSD_AA = rmsd_sc_automorphic(aa_design_prot, aa_af3_prot)
                staging_columns.append("RMSD_AA", RMSD_AA)

                if any(aa_af3.hetero):
                    pLDDT_SC_around_lig = np.mean(confidences["atom_plddts"][~aa_af3.hetero][sc_close_to_het_mask], dtype=np.float32)
                    staging_columns.append("pLDDT_SC_around_lig", pLDDT_SC_around_lig)
                    
                    # before calculating RMSD_SC_around_lig, we align on that region
                    aa_af3_temp, _ = superimpose(aa_design_prot, aa_af3_prot, atom_mask=sc_close_to_het_mask)
                    RMSD_SC_around_lig = rmsd_sc_automorphic(aa_design_prot[sc_close_to_het_mask], aa_af3_temp[~aa_af3_temp.hetero][sc_close_to_het_mask])
                    staging_columns.append("RMSD_SC_around_lig", RMSD_SC_around_lig)
                    del aa_af3_temp
                    
                    pLDDT_LIGs = {k: np.mean(confidences["atom_plddts"][aa_af3.res_name == k], dtype=np.float32) for k in np.unique(aa_af3[aa_af3.hetero].res_name)}
                    for key, val in pLDDT_LIGs.items():
                        staging_columns.append(f"pLDDT_LIG_{key}", val)
                
                # ------------ calculate user defiend metrics ------------
                # TODO: pass one input object instead of separate arguments
                sfx_input = ScoreFunctionInput(
                    aa_design=aa_design,
                    aa_af3=aa_af3,
                    meta=meta,
                    confidences=confidences,
                    sc_close_to_het_mask=sc_close_to_het_mask,
                    CLI_args=args
                )

                eval_scorefxns(scorefxns, sfx_input, staging_columns)

        end_time = time.time()
        n_models = len(staging_columns.get_dict()['id_hex']['val'])
        print(f"Processed batch with {n_models} models in {end_time - start_time:.3f} sec")

        staging_table = pa.table(
            {
                **{name: pa.array(col["val"], type=col["type"]) for name, col in staging_columns.get_dict().items()},
                **{"af3_analysis_source_version": pa.array(np.full(len(staging_columns.get_dict()["id_hex"]["val"]), snapshot_ver, dtype=np.int64))}
            }
        )

        write_deltalake(staging_path, staging_table, mode="append", configuration={'delta.checkpointInterval': '500'})

if __name__ == "__main__":
    main()