import os
from typing import overload, Optional
from deltalake import DeltaTable
from protlake.utils import (
    DeltaTable_nrow, 
    deltatable_maintenance
)
from protlake.read import (
    bcif_shard_to_mmCIF_file,
    pread_bcif_to_atom_array
)
import pandas as pd
import numpy as np
from biotite.structure import to_sequence

class ProtLake():
    def __init__(self, path):
        self.path = path
        self.delta_path = os.path.join(path, 'delta')
        self.shard_path = os.path.join(path, 'shards')

        self.dt = None
        self.load()

    def load(self) -> bool:
        self.dt = DeltaTable(f"file://{os.path.abspath(self.delta_path)}")
        self.nrow = DeltaTable_nrow(self.dt)
        self.colnames = [f.name for f in self.dt.schema().fields]
        return True
    
    def to_pandas(self) -> pd.DataFrame:
        return self.dt.to_pandas()
    
    def maintenance(self, target_size = 1 << 28, max_concurrent_tasks=2, reload=False) -> bool:
        deltatable_maintenance(self.dt, target_size=target_size, max_concurrent_tasks=max_concurrent_tasks)
        if reload:
            self.load()
        return True

    # ------------- extract_cif -------------
    # ---------------------------------------
    @overload
    def extract_cif(
        self,
        bcif_shard: str | list[str],
        bcif_off: int | list[int],
        bcif_len: int | list[int],
        file_names: str | list[str],
        out_dir: str,
        overwrite: bool = False,
    ) -> None: ...

    @overload
    def extract_cif(
        self,
        *,
        out_dir: str,
        overwrite: bool = False,
        df: pd.DataFrame,
        append_seed_sample: bool = True,
    ) -> None: ...

    def extract_cif(
        self,
        bcif_shard: Optional[str | list[str]] = None,
        bcif_off: Optional[int | list[int]] = None,
        bcif_len: Optional[int | list[int]] = None,
        file_names: Optional[str | list[str]] = None,
        out_dir: str = None,
        overwrite: bool = False,
        df: Optional[pd.DataFrame] = None,
        append_seed_sample: bool = True,
    ) -> None:
        
        if out_dir is None:
            raise ValueError("out_dir must be specified.")
        
        if df is not None:
            # check that no other args are provided
            if any(arg is not None for arg in (bcif_shard, bcif_off, bcif_len, file_names)):
                raise ValueError("Pass either df OR explicit bcif_shard/off/len/file_names, not both.")
            
            # check if df has all the required columns
            required_cols = {'bcif_shard', 'bcif_data_off', 'bcif_len', 'name'}
            if append_seed_sample:
                required_cols.update({'seed', 'sample'})
            if not required_cols.issubset(df.columns):
                raise ValueError(f"df must contain the following columns: {required_cols}")
            
            bcif_shard = df['bcif_shard'].tolist()
            bcif_off = df['bcif_data_off'].tolist()
            bcif_len = df['bcif_len'].tolist()
            if append_seed_sample:
                file_names = [f"{name}_{seed}_{sample}" for name, seed, sample in zip(df['name'], df['seed'], df['sample'])]
            else:
                file_names = df['name'].tolist()
        else:
            if bcif_shard is None or bcif_off is None or bcif_len is None or file_names is None:
                raise ValueError("Either df or all of bcif_shard, bcif_off, bcif_len, and file_names must be provided.")
            if isinstance(bcif_shard, str):
                bcif_shard = [bcif_shard]
            if isinstance(bcif_off, int):
                bcif_off = [bcif_off]
            if isinstance(bcif_len, int):
                bcif_len = [bcif_len]
            if isinstance(file_names, str):
                file_names = [file_names]

        os.makedirs(out_dir, exist_ok=True)

        for shard, offset, length, file_name in zip(bcif_shard, bcif_off, bcif_len, file_names):
            shard_path = os.path.join(self.shard_path, str(shard))
            if not file_name.endswith('.cif'):
                file_name += '.cif'
            out_path = os.path.join(out_dir, file_name)
            if os.path.exists(out_path) and not overwrite:
                print(f"File {out_path} exists, skipping...")
                continue
            try:
                bcif_shard_to_mmCIF_file(shard_path, offset, length, out_path)
            except Exception as e:
                print(f"Error reading {shard_path} at offset {offset} with length {length}: {e}")
                continue

    # --------------- get_seq ---------------
    # ---------------------------------------
    @overload
    def get_seq(
        self,
        bcif_shard: str | list[str],
        bcif_off: int | list[int],
        bcif_len: int | list[int],
        name: str | list[str],
        df: None = None,
    ) -> str | list[str]: ...

    @overload
    def get_seq(
        self,
        bcif_shard: None = None,
        bcif_off: None = None,
        bcif_len: None = None,
        name: None = None,
        df: pd.DataFrame = ...,
        chains: list[str] = ['A'],
    ) -> str | list[str]: ...

    def get_seq(
        self,
        bcif_shard: Optional[str | list[str]] = None,
        bcif_off: Optional[int | list[int]] = None,
        bcif_len: Optional[int | list[int]] = None,
        name: Optional[str | list[str]] = None,
        df: Optional[pd.DataFrame] = None,
        chains: list[str] = ['A'],
    ) -> str | list[str]:
        
        if df is not None:
            if any(arg is not None for arg in (bcif_shard, bcif_off, bcif_len, name)):
                raise ValueError("Pass either df OR explicit bcif_shard/off/len/name, not both.")
            bcif_shard = df['bcif_shard'].tolist()
            bcif_off = df['bcif_data_off'].tolist()
            bcif_len = df['bcif_len'].tolist()
            name = df['name'].tolist()
        else:
            if bcif_shard is None or bcif_off is None or bcif_len is None or name is None:
                raise ValueError("Either df or all of bcif_shard, bcif_off, bcif_len, and name must be provided.")
            if isinstance(bcif_shard, str):
                bcif_shard = [bcif_shard]
            if isinstance(bcif_off, int):
                bcif_off = [bcif_off]
            if isinstance(bcif_len, int):
                bcif_len = [bcif_len]
            if isinstance(name, str):
                name = [name]

        seq_list = []
        for shard, offset, length, id in zip(bcif_shard, bcif_off, bcif_len, name):
            shard_path = os.path.join(self.shard_path, str(shard))
            try:
                aa = pread_bcif_to_atom_array(shard_path, offset, length)
                seqs, _ = to_sequence(aa[np.isin(aa.chain_id, chains) & ~aa.hetero])
                if len(seqs) > 1:
                    print(f"Warning: More than one chain found in {id}, returning first chain only.")
                seq_list.append(str(seqs[0]))
            except Exception as e:
                print(f"Error reading {shard_path} at offset {offset} with length {length}: {e}")
                continue

        return seq_list if len(seq_list) > 1 else seq_list[0]

    def __repr__(self):
        return f"ProtLake at {self.path} with {self.nrow:,} structures"