from protlake.af3.analysis_worker import ScoreFunctionInput
import numpy as np
import itertools
from biotite.structure import rmsd

scorefxn_name = "lig_rmsd"
description = "Calculates ligand RMSD between designed structure and AF3 prediction. Supports multiple ligands and chemically equivalent atoms."

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
        return r, True
    return current_best, False

def score(sfx_input: ScoreFunctionInput):

    aa_design = sfx_input.aa_design.copy() # copy to avoid modifying original
    aa_af3 = sfx_input.aa_af3.copy()
    meta = sfx_input.meta
    CLI_args = sfx_input.CLI_args

    if CLI_args.atom_names_rmsd is not None:
        atom_names_rmsd = {s.split('$')[0]: np.array(s.split('$')[1].split(':'), dtype='<U6') for s in CLI_args.atom_names_rmsd}
    else:
        # fill atom_names_rmsd with every ligand and every atom name found in the af3 structure 
        # (requires ligand and atom names to be identical in design model and af3 prediction)
        hetero_resnames = np.unique(aa_af3[aa_af3.hetero].res_name)
        atom_names_rmsd = {}
        for resname in hetero_resnames:
            atom_names = np.unique(aa_af3[_resname_mask(aa_af3, resname)].atom_name)
            atom_names_rmsd[f"{resname}:{resname}"] = atom_names
        
    if CLI_args.chem_eq_atoms is not None:
        chem_eq_atoms = {s.split('$')[0]: np.array([x.split(':') for x in s.split('$')[1].split('_')]) for s in CLI_args.chem_eq_atoms}
    else:
        chem_eq_atoms = {}

    LIG_rmsds = {}

    for key, atom_names in atom_names_rmsd.items():
        resname_af3, resname_design = key.split(":")
        af3_rsmd_atoms = _select_and_sort_aa(aa_af3, resname_af3, atom_names)
        if af3_rsmd_atoms.shape[0] == 0:
            print(f"Warning (lig_rmsd): no atoms found for {resname_af3} in af3 model {meta['name']}, sample {meta['sample']}, seed {meta['seed']}")
            continue
        af3_chain_resi_pairs = _unique_chain_resi_pairs(af3_rsmd_atoms)


        design_rsmd_atoms = _select_and_sort_aa(aa_design, resname_design, atom_names)
        if design_rsmd_atoms.shape[0] == 0:
            print(f"Warning (lig_rmsd): no atoms found for {resname_design} in design model {meta['name']}, sample {meta['sample']}, seed {meta['seed']}")
            continue
        design_chain_resi_pairs = _unique_chain_resi_pairs(design_rsmd_atoms)

        lig_rmsd = np.nan

        # iterate over all combinations of ligands with same residue name
        for af3_lig, design_lig in itertools.product(af3_chain_resi_pairs, design_chain_resi_pairs):
            # print("AF3:", af3_lig, " <->  DESIGN:", design_lig)

            mask_design = _chain_resi_mask(design_rsmd_atoms, design_lig[0] ,design_lig[1])
            mask_af3 = _chain_resi_mask(af3_rsmd_atoms, af3_lig[0] ,af3_lig[1])
            design_sel = design_rsmd_atoms[mask_design]
            af3_sel = af3_rsmd_atoms[mask_af3]
            design_names = design_sel.atom_name
            af3_names = af3_sel.atom_name
            # check if both selections have the same atoms
            if not np.array_equal(np.sort(design_names), np.sort(af3_names)):
                if CLI_args.lig_rmsd_allow_partial_matches:
                    common = np.intersect1d(design_names, af3_names)
                    if common.shape[0] == 0:
                        print(f"Warning (lig_rmsd): no common atoms found for design {design_lig} and af3 {af3_lig} in model {meta['name']}, sample {meta['sample']}, seed {meta['seed']}")
                        continue
                    design_sel = design_sel[np.isin(design_sel.atom_name, common)]
                    af3_sel = af3_sel[np.isin(af3_sel.atom_name, common)]
                    if not np.array_equal(design_sel.atom_name, af3_sel.atom_name):
                        print(f"Warning (lig_rmsd): atom order mismatch after matching for design {design_lig} and af3 {af3_lig} in model {meta['name']}, sample {meta['sample']}, seed {meta['seed']}")
                        continue
                    print(f"Info (lig_rmsd): partial atom match for design {design_lig} ({design_names}) and af3 {af3_lig} ({af3_names}) in model {meta['name']}, sample {meta['sample']}, seed {meta['seed']}. Using common atoms: {common}")
                else:
                    print(f"Warning (lig_rmsd): different atom sets for design {design_lig} ({design_names}) and af3 {af3_lig} ({af3_names}) in model {meta['name']}, sample {meta['sample']}, seed {meta['seed']}")
                    print("  To allow partial matches, enable --lig_rmsd_allow_partial_matches")
                    continue
            
            design_sel_c = design_sel.copy() # copy might not be needed, but to be sure changing positions does not affect anything else
            af3_sel_c = af3_sel.copy()
            lig_rmsd_curr_permut = np.nan
            lig_rmsd_curr_permut, _ = _update_best_rmsd(lig_rmsd_curr_permut, design_sel_c, af3_sel_c)
            if key in chem_eq_atoms:
                # flip equivalent atoms and see if rmsd improves
                for a, b in chem_eq_atoms[key]:
                    # switch positions of atoms 
                    i, j = np.where(af3_sel_c.atom_name == a)[0][0], \
                           np.where(af3_sel_c.atom_name == b)[0][0]
                    tmp_i = af3_sel_c[i].copy()     # make a copy of atom i
                    af3_sel_c[i] = af3_sel_c[j]     # assign atom j into position i
                    af3_sel_c[j] = tmp_i            # put original i into j
                    lig_rmsd_curr_permut, improved = _update_best_rmsd(lig_rmsd_curr_permut, design_sel_c, af3_sel_c) 
                    if improved: # keep the swapped version if it improved
                        continue  
                    else: # swap back
                        af3_sel_c[j] = af3_sel_c[i] # put swapped i back into j
                        af3_sel_c[i] = tmp_i        # put original i back into i

            lig_rmsd, _ = _update_best_rmsd(lig_rmsd, design_sel_c, af3_sel_c)

        LIG_rmsds[f"RMSD_lig_{key}"] = lig_rmsd

    return LIG_rmsds

def register_args(parser):
    parser.add_argument("--atom_names_rmsd", type=str, required=False, nargs='+',
                        help="Names of ligand atoms (in the design structure) that should be used for ligand rmsd calculation \
                              Format: AF3_LIG:DESIGN_LIG$ATOM1:ATOM2:...")
    parser.add_argument("--chem_eq_atoms", type=str, required=False, nargs='+',
                        help="Chemically equivalent atom pairs to account for when calculating ligand RMSDs. \
                              (e.g. the non-substituted C atoms in para-hydroxybenzoic acid) \
                              Format: AF3_LIG:DESIGN_LIG$ATOM1:ATOM2_ATOM3:ATOM4_ATOM5:ATOM6...")
    parser.add_argument("--lig_rmsd_allow_partial_matches", action="store_true",
                        help="If set, ligands with missing atoms in the af3 structure will still be considered for RMSD calculation. \
                              The RMSD will then be calculated over the available atoms only.")