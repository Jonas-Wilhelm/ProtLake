import os
from typing import overload, Optional
from deltalake import DeltaTable
from protlake.utils import DeltaTable_nrow
from protlake.read import (
    bcif_shard_to_mmCIF_file
)
import pandas as pd

class ProtLake():
    def __init__(self, path):
        self.path = path
        self.delta_path = os.path.join(path, 'delta')
        self.shard_path = os.path.join(path, 'shards')

        self.dt = None
        self.load()

    def load(self):
        self.dt = DeltaTable(f"file://{os.path.abspath(self.delta_path)}")
        self.nrow = DeltaTable_nrow(self.dt)
        self.colnames = [f.name for f in self.dt.schema().fields]
        return True
    
    def to_pandas(self):
        return self.dt.to_pandas()
    
    @overload
    def extract_cif(
        self,
        bcif_shard: str | list[str],
        bcif_off: int | list[int],
        bcif_len: int | list[int],
        file_names: str | list[str],
        out_dir: str,
        overwrite: bool = False,
        df: None = None,
    ) -> None: ...

    @overload
    def extract_cif(
        self,
        bcif_shard: None = None,
        bcif_off: None = None,
        bcif_len: None = None,
        file_names: None = None,
        out_dir: str = ...,
        overwrite: bool = False,
        df: pd.DataFrame = ...,
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
    ) -> None:
        
        if out_dir is None:
            raise ValueError("out_dir must be specified.")
        
        if df is not None:
            if any(arg is not None for arg in (bcif_shard, bcif_off, bcif_len, file_names)):
                raise ValueError("Pass either df OR explicit bcif_shard/off/len/file_names, not both.")
            bcif_shard = df['bcif_shard'].tolist()
            bcif_off = df['bcif_data_off'].tolist()
            bcif_len = df['bcif_len'].tolist()
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

    def __repr__(self):
        return f"ProtLake at {self.path} with {self.nrow:,} structures"