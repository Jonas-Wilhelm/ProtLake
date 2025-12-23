from protlake.af3.analysis_worker import ScoreFunctionInput
from protlake.utils import rmsd_sc_automorphic
from biotite.structure import superimpose
import sqlite3
import msgpack
import numpy as np

scorefxn_name = "diff_motif_rmsd"
description = "Calculates RMSD over specified motif residues between designed structure and AF3 prediction, using a diffused motif map."

_conn = None

def init(CLI_args):
    global _conn
    _conn = sqlite3.connect(CLI_args.diff_motif_map_db)


def _parse_chain_resid(chain_resno):
    chain_id = chain_resno[0]
    res_id = int(chain_resno[1:])
    return chain_id, res_id


def _load_diffused_index_map(db_conn, name):
    cur = db_conn.cursor()

    cur.execute(
        "SELECT data FROM maps WHERE name = ?",
        (name,),
    )
    row = cur.fetchone()

    if row is None:
        return None

    return msgpack.unpackb(row[0], raw=False)


def score(sfx_input: ScoreFunctionInput):
    aa_design = sfx_input.aa_design.copy() # copy to avoid modifying original
    aa_af3 = sfx_input.aa_af3.copy()
    # only protein atoms
    aa_design = aa_design[~aa_design.hetero]
    aa_af3 = aa_af3[~aa_af3.hetero]
    name = sfx_input.meta['name']
    n_parts = sfx_input.CLI_args.diff_motif_map_db_name_split
    if n_parts is not None:
        diff_name = '_'.join(name.split('_')[:n_parts])
    else:
        diff_name = name

    index_map = _load_diffused_index_map(_conn, diff_name)
    if index_map is None:
        raise ValueError(f"No diffused motif map found for structure name '{diff_name}' in database.")
    
    if sfx_input.CLI_args.diff_motif_align:
        motif_set = {_parse_chain_resid(v) for v in index_map.values()}
        motif_mask = [(c, r) in motif_set for c, r in zip(aa_design.chain_id, aa_design.res_id)]
        aa_af3, _ = superimpose(aa_af3, aa_design, atom_mask=motif_mask)
    
    out = {}
    for k, v in index_map.items():
        chain, res = _parse_chain_resid(v)
        rmsd = rmsd_sc_automorphic(
            aa_design[(aa_design.chain_id == chain) & (aa_design.res_id == res)],
            aa_af3[(aa_af3.chain_id == chain) & (aa_af3.res_id == res)]
        )
        out[f"diff_motif_rmsd_{k}"] = rmsd

    rmsds = list(out.values())
    out["diff_motif_rmsd_overall"] = np.sqrt(sum(r*r for r in rmsds) / len(rmsds))

    return out


def register_args(parser):
    parser.add_argument("--diff-motif-map-db", type=str, required=True,
                        help="Path to SQLite database containing mapping of motif residues between diffusion input and design / af3 prediction.")
    parser.add_argument("--diff-motif-map-db-name-split", type=int, required=False,
                        help="Number of parts of the structure name (split by '_') to use as key to look up in the diff motif map database.")
    parser.add_argument("--diff-motif-align", action='store_true', required=False,
                        help="If set, align the two structures on the motif residues before calculating the RMSD.")
