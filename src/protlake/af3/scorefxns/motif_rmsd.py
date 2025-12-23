import numpy as np
from protlake.utils import rmsd_sc_automorphic
from biotite.structure import rmsd, superimpose
from protlake.af3.analysis_worker import ScoreFunctionInput

scorefxn_name = "motif_rmsd"
description = "Calculates RMSD over specified motif residues between designed structure and AF3 prediction. Supports optional superimposition on motif residues."

def _expand_contig(contig_str):
    """
    Expand a contig string like "A10,B12-30,A35" into a list of (chain_id, res_id) tuples.
    Also parses contigs without chain IDs, e.g. "10,12-30,35" (assumes chain A).
    Returns a list of tuples (chain_id, res_id), sorted by chain_id and res_id.
    """
    residues = set()
    for part in contig_str.split(','):
        if '-' in part:
            if part[0].isalpha():
                # Parse chain ID and range
                chain_id = part[0]
                range_part = part[1:]
            else:
                # Set default chain ID if not provided
                chain_id = 'A'
                range_part = part
            start, end = map(int, range_part.split('-'))
            for res_id in range(start, end + 1):
                residues.add((chain_id, res_id))
        else:
            if part[0].isalpha():
                # Parse chain ID and residue number
                chain_id = part[0]
                res_id = int(part[1:])
            else:
                # Set default chain ID if not provided
                chain_id = 'A'
                res_id = int(part)
            residues.add((chain_id, res_id))
    return sorted(residues, key=lambda x: (x[0], x[1]))

def score(sfx_input: ScoreFunctionInput):
    aa_design = sfx_input.aa_design.copy() # copy to avoid modifying original
    aa_af3 = sfx_input.aa_af3.copy()
    CLI_args = sfx_input.CLI_args
    motif_residues = _expand_contig(CLI_args.motif_residues_contig)

    contig_mask = np.zeros(len(aa_design), dtype=bool)
    for chain_id, res_id in motif_residues:
        contig_mask |= (aa_design.chain_id == chain_id) & (aa_design.res_id == res_id)

    CA_mask = (~aa_design.hetero & (aa_design.atom_name == "CA"))

    if CLI_args.superimpose_on_motif:
        aa_af3_superimposed, _ = superimpose(aa_af3, aa_design, atom_mask=(contig_mask & CA_mask))
    else:
        aa_af3_superimposed = aa_af3

    motif_aa_rmsd = rmsd_sc_automorphic(aa_design[contig_mask], aa_af3_superimposed[contig_mask])
    motif_CA_rmsd = rmsd(aa_design[contig_mask & CA_mask], aa_af3_superimposed[contig_mask & CA_mask])

    return {
        "motif_rmsd_AA": motif_aa_rmsd,
        "motif_rmsd_CA": motif_CA_rmsd,
    }

def register_args(parser):
    parser.add_argument("--motif-residues-contig", type=str, required=True, 
                        help="Residues that should be used for motif rmsd calculation, can either include chain IDs (e.g. A10,A12-30,A35) \
                            or not (e.g. 10,12-30,35). If no chain IDs are given, chain A will be used.")
    parser.add_argument("--superimpose-on-motif", action="store_true",
                        help="If set, the AF3 structure will be aligned to the design structure based on the motif residue CA atoms before calculating the RMSD.")
