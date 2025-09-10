import numpy as np
import itertools
from biotite.structure import rmsd

scorefxn_name = "lig_rmsd"
description = "<>"

def _resname_mask(aa, res_name):
    return (aa.res_name == res_name).astype(bool)

def _select_and_sort_aa(aa, res_name, atom_names):
    aa_sel = aa[(_resname_mask(aa, res_name) & np.isin(aa.atom_name, atom_names))]
    return aa_sel[np.argsort(aa_sel.atom_name)]

def _chain_resi_mask(aa, chain, resi):
    return (aa.chain_id == str(chain)) & (aa.res_id == int(resi))

def _unique_chain_resi_pairs(aa):
    """Return a (N,2) array of unique (chain_id, res_id) pairs."""
    # if aa.shape[0] == 0:
    #     return np.empty((0, 2), dtype=object)
    pairs = np.stack([aa.chain_id, aa.res_id], axis=1)
    unique = np.unique(pairs, axis=0)
    return np.atleast_2d(unique)

def _update_best_rmsd(current_best, reference, subject):
    r = rmsd(reference, subject)
    if np.isnan(current_best) or r < current_best:
        return r
    return current_best

def score(aa_design, aa_af3, meta, confidences, sc_close_to_het_mask, CLI_args):

    atom_names_rmsd = {s.split('$')[0]: np.array(s.split('$')[1].split(':'), dtype='<U6') for s in CLI_args.atom_names_rmsd}
    chem_eq_atoms = {s.split('$')[0]: np.array([x.split(':') for x in s.split('$')[1].split('_')]) for s in CLI_args.chem_eq_atoms}

    LIG_rmsds = {}

    for key, atom_names in atom_names_rmsd.items():
        resname_af3, resname_design = key.split(":")
        af3_rsmd_atoms = _select_and_sort_aa(aa_af3, resname_af3, atom_names)
        if af3_rsmd_atoms.shape[0] == 0:
            print(f"Warning: no atoms found for {resname_af3} in af3 model {meta['name']}, sample {meta['sample']}, seed {meta['seed']}")
            continue
        af3_chain_resi_pairs = _unique_chain_resi_pairs(af3_rsmd_atoms)


        design_rsmd_atoms = _select_and_sort_aa(aa_design, resname_design, atom_names)
        if design_rsmd_atoms.shape[0] == 0:
            print(f"Warning: no atoms found for {resname_design} in design model {meta['name']}, sample {meta['sample']}, seed {meta['seed']}")
            continue
        design_chain_resi_pairs = _unique_chain_resi_pairs(design_rsmd_atoms)

        lig_rmsd = np.nan

        # iterate over all combinations of ligands with same residue name
        for af3_lig, design_lig in itertools.product(af3_chain_resi_pairs, design_chain_resi_pairs):
            # print("AF3:", af3_lig, " <->  DESIGN:", design_lig)

            mask_design = _chain_resi_mask(design_rsmd_atoms, design_lig[0] ,design_lig[1])
            mask_af3 = _chain_resi_mask(af3_rsmd_atoms, af3_lig[0] ,af3_lig[1])
            if np.sum(mask_design) != np.sum(mask_af3):
                print(f"Warning: different number of atoms for design {design_lig} ({np.sum(mask_design)}) and af3 {af3_lig} ({np.sum(mask_af3)}) in model {meta['name']}, sample {meta['sample']}, seed {meta['seed']}")
                continue
            design_sel = design_rsmd_atoms[mask_design].copy() # copy might not be needed, but to be sure changing positions does not affect anything else
            af3_sel = af3_rsmd_atoms[mask_af3].copy()
            lig_rmsd = _update_best_rmsd(lig_rmsd, design_sel, af3_sel)
            if key in chem_eq_atoms:
                # flip that eqivalent atoms and see if rmsd improves
                for tuple in chem_eq_atoms[key]:
                    # switch positions of atoms 
                    i, j = np.where(af3_sel.atom_name == tuple[0])[0][0], \
                           np.where(af3_sel.atom_name == tuple[1])[0][0]
                    tmp = af3_sel[i].copy()     # make a copy of atom i
                    af3_sel[i] = af3_sel[j]     # assign atom j into position i
                    af3_sel[j] = tmp            # put original i into j
                lig_rmsd = _update_best_rmsd(lig_rmsd, design_sel, af3_sel)

        LIG_rmsds[f"RMSD_lig_{key}"] = lig_rmsd

    return LIG_rmsds

def register_args(parser):
    parser.add_argument("--atom_names_rmsd", type=str, required=False, nargs='+',
                        help="Names of ligand atoms (in the design structure) that should be used for ligand rmsd calculation \
                              Format: AF3_LIG:DESIGN_LIG-ATOM1:ATOM2:...")
    parser.add_argument("--chem_eq_atoms", type=str, required=False, nargs='+',
                        help="Chemically equivalent atom pairs to account for when calculating ligand RMSDs. \
                              (e.g. the non-substituted C atoms in para-hydroxybenzoic acid) \
                              Format: AF3_LIG:DESIGN_LIG-ATOM1:ATOM2_ATOM3:ATOM4-ATOM5:ATOM6...")