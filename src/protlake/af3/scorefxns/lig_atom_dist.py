import numpy as np

scorefxn_name = "lig_atom_dist"
description = "Calculates distances between specified ligand atom pairs. Supports multiple ligands."

def score(aa_design, aa_af3, meta, confidences, sc_close_to_het_mask, CLI_args):
    
    out = {}

    for ligand_atom_pair in CLI_args.ligand_atom_distances:
        left, right = ligand_atom_pair.split('_')
        lig1, atom1 = left.split(':')
        lig2, atom2 = right.split(':')
        coord_1 = aa_af3[(aa_af3.res_name == lig1) & (aa_af3.atom_name == atom1)].coord
        assert coord_1.shape[0] == 1, \
            f"more than one atom found for ligand name {lig1}, atom {atom1}"
        coord_2 = aa_af3[(aa_af3.res_name == lig2) & (aa_af3.atom_name == atom2)].coord
        assert coord_2.shape[0] == 1, \
            f"more than one atom found for ligand name {lig2}, atom {atom2}"
        d = np.linalg.norm(coord_1 - coord_2)

        out[f"lig_atom_dist_{lig1}.{atom1}_{lig2}.{atom2}"] = d
    
    return out
    

def register_args(parser):
    parser.add_argument("--ligand_atom_distances", type=str, required=False, nargs='+',
                        help="Pairs of ligand atoms to calculate distances for. \
                              Example: MY-AKN:C1_MY-AZD:N3 MY-AKN:C2_MY-AZD:N5")