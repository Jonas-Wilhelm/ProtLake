import os
from deltalake import DeltaTable
import pyarrow.parquet as pq
import numpy as np
from biotite.structure.atoms import coord
from biotite.structure.util import vector_dot

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

def ensure_dirs(list_of_dirs):
    for path in list_of_dirs:
        os.makedirs(path, exist_ok=True)

def DeltaTable_nrow(dt: DeltaTable):
    count = 0
    for f in dt.file_uris():
        count += pq.ParquetFile(f).metadata.num_rows
    return count

def _sq_euclidian(reference, subject):
    '''
    Squared Euclidian distance function from biotite
    '''
    reference_coord = coord(reference)
    subject_coord = coord(subject)
    if reference_coord.ndim != 2:
        raise TypeError(
            "Expected an AtomArray or an ndarray with shape (n,3) as reference"
        )
    dif = subject_coord - reference_coord
    return vector_dot(dif, dif)

def rmsd_sc_automorphic(reference, subject):
    """
    Compute the per-residue all-atom RMSD between two structures, accounting for
    symmetry in selected sidechain atoms (e.g., OD1/OD2 in ASP). Returns the overall 
    RMSD across all residues.
    """
    # TODO make this work with non complete residues (e.g. only last 3 atoms of PHE)
    SYMMETRY_MAP = {
        "ASP": [("OD1","OD2")],
        "GLU": [("OE1","OE2")],
        "PHE": [("CD1","CD2"), ("CE1","CE2")],
        "TYR": [("CD1","CD2"), ("CE1","CE2")],
        "ARG": [("NH1","NH2")],
        "LEU": [("CD1","CD2")],
        "VAL": [("CG1","CG2")],
    }
    chain = reference.chain_id.astype(str)
    res   = reference.res_id.astype(str)
    keys  = chain + ":" + res

    boundaries = np.where(keys[:-1] != keys[1:])[0] + 1
    groups = np.split(np.arange(reference.shape[0]), boundaries)

    all_sd = np.array([], dtype=np.float32)
    for atom_idx in groups:
        res_name = reference[atom_idx[0]].res_name
        sd = _sq_euclidian(reference[atom_idx], subject[atom_idx])
        msd = np.mean(sd, axis=-1)
        if res_name in SYMMETRY_MAP.keys():
            atom_idx_swapped = atom_idx.copy()
            atom_pairs = SYMMETRY_MAP[res_name]
            for atom_pair in atom_pairs:
                # check if both atoms in pair are present
                if not (np.any(reference[atom_idx].atom_name == atom_pair[0]) and np.any(reference[atom_idx].atom_name == atom_pair[1])):
                    continue
                # swap their indices
                idx_a = np.where(reference[atom_idx].atom_name == atom_pair[0])[0][0]
                idx_b = np.where(reference[atom_idx].atom_name == atom_pair[1])[0][0]
                atom_idx_swapped[idx_a], atom_idx_swapped[idx_b] = atom_idx_swapped[idx_b], atom_idx_swapped[idx_a]
            sd_flip = _sq_euclidian(reference[atom_idx], subject[atom_idx_swapped])
            msd_flip = np.mean(sd_flip, axis=-1)
            if msd_flip < msd:
                sd = sd_flip
                msd = msd_flip
        all_sd = np.append(all_sd, sd)

    return np.sqrt(np.mean(all_sd, axis=-1))