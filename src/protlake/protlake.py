import os
from typing import Any, Dict, Optional, overload
from deltalake import DeltaTable, write_deltalake
from protlake.query import check_exists as delta_check_exists
from protlake.utils import (
    DeltaTable_nrow, 
    deltatable_maintenance,
    bcif_shard_to_mmCIF_file,
    bcif_shard_to_mmCIF_str,
    pread_bcif_to_atom_array
)
from protlake.af3.ingest import CORE_SCHEMA
from protlake.write.core import RetryConfig
import pyarrow as pa
import pandas as pd
import numpy as np
from biotite.structure import to_sequence

class ProtLake():
    def __init__(self, path, create=False, retry_conf: Optional[RetryConfig] = None):
        self.path = path
        self.delta_path = os.path.join(path, 'delta')
        self.shard_path = os.path.join(path, 'shards')
        self.retry_conf = retry_conf if retry_conf is not None else RetryConfig()
        self.n_row = None

        self.dt = None
        self.load(create=create)

    def load(self, create=False) -> bool:
        try:
            self.dt = DeltaTable(f"file://{os.path.abspath(self.delta_path)}")
            if create:
                print(f"ProtLake at {self.path} already exists, not creating a new one, despite create=True.")
        except Exception as e:
            if create:
                print(f"Creating new ProtLake at {self.path}")
                os.makedirs(self.delta_path, exist_ok=True)
                os.makedirs(self.shard_path, exist_ok=True)
                empty = pa.table(
                    {f.name: pa.array([], type=f.type) for f in CORE_SCHEMA}
                )
                write_deltalake(
                    f"file://{os.path.abspath(self.delta_path)}",
                    empty,
                    configuration={'delta.checkpointInterval': '500'}
                )
                self.dt = DeltaTable(f"file://{os.path.abspath(self.delta_path)}")
            else:
                raise RuntimeError(f"Could not load ProtLake at {self.path}: {e}")
        self.colnames = [f.name for f in self.dt.schema().fields]
        return True

    def nrow(self) -> int:
        self.load()
        self.n_row = DeltaTable_nrow(self.dt)
        return self.n_row

    def to_pandas(self) -> pd.DataFrame:
        return self.dt.to_pandas()

    def check_exists(self, keys: Dict[str, Any]) -> bool:
        return delta_check_exists(
            delta_path=self.delta_path,
            keys=keys,
            retry_config=self.retry_conf,
        )
    
    def maintenance(self, target_size = 1 << 28, max_concurrent_tasks=2, retention_hours=0, reload=False) -> bool:
        deltatable_maintenance(self.dt, target_size=target_size, max_concurrent_tasks=max_concurrent_tasks, retention_hours=retention_hours)
        if reload:
            self.load()
        return True

    # ------------- extract_cif -------------
    # ---------------------------------------
    @overload
    def extract_cif(
        self,
        bcif_shard: str | list[str],
        bcif_data_off: int | list[int],
        bcif_len: int | list[int],
        file_names: str | list[str],
        out_dir: str,
        overwrite: bool = False,
        as_str: bool = False,
    ) -> None: ...

    @overload
    def extract_cif(
        self,
        *,
        out_dir: str,
        overwrite: bool = False,
        as_str: bool = False,
        df: pd.DataFrame,
        file_name_cols: str | list[str] | None = None,
    ) -> None: ...

    def extract_cif(
        self,
        bcif_shard: Optional[str | list[str]] = None,
        bcif_data_off: Optional[int | list[int]] = None,
        bcif_len: Optional[int | list[int]] = None,
        file_names: Optional[str | list[str]] = None,
        file_name_cols: Optional[str | list[str]] = None,
        out_dir: str = None,
        overwrite: bool = False,
        as_str: bool = False,
        df: Optional[pd.DataFrame] = None,
    ) -> None | str | list[str]:
        
        if out_dir is None and not as_str:
            raise ValueError("out_dir must be specified.")
        
        if df is not None:
            # check that no other args are provided
            if any(arg is not None for arg in (bcif_shard, bcif_data_off, bcif_len, file_names)):
                raise ValueError("Pass either df OR explicit bcif_shard/bcif_data_off/bcif_len/file_names, not both.")
            
            # check if df has all the required columns
            required_cols = {'bcif_shard', 'bcif_data_off', 'bcif_len'}
            if not required_cols.issubset(df.columns):
                raise ValueError(f"df must contain the following columns: {required_cols}")
            
            bcif_shard = df['bcif_shard'].tolist()
            bcif_data_off = df['bcif_data_off'].tolist()
            bcif_len = df['bcif_len'].tolist()
            if file_name_cols is None:
                file_name_cols = 'name'
                print(
                    "Warning: extract_cif() defaulted file_name_cols to 'name'. "
                    "If 'name' is not unique, files may be overwritten or skipped. "
                    "Specify file_name_cols to avoid this."
                )
            if isinstance(file_name_cols, str):
                file_name_cols = [file_name_cols]
            missing_name_cols = set(file_name_cols) - set(df.columns)
            if missing_name_cols:
                raise ValueError(f"df is missing file name columns: {missing_name_cols}")
            file_names = (
                df[file_name_cols]
                .astype(str)
                .agg('_'.join, axis=1)
                .tolist()
            )
        else:
            if bcif_shard is None or bcif_data_off is None or bcif_len is None or file_names is None:
                raise ValueError("Either df or all of bcif_shard, bcif_data_off, bcif_len, and file_names must be provided.")
            if isinstance(bcif_shard, str):
                bcif_shard = [bcif_shard]
            if isinstance(bcif_data_off, int):
                bcif_data_off = [bcif_data_off]
            if isinstance(bcif_len, int):
                bcif_len = [bcif_len]
            if isinstance(file_names, str):
                file_names = [file_names]

        if file_name_cols is not None and df is None:
            raise ValueError("file_name_cols can only be used when df is provided.")

        if not as_str and file_names is None:
            raise ValueError("file_names must be provided explicitly or derived from df via file_name_cols.")

        if not as_str:
            os.makedirs(out_dir, exist_ok=True)
            for shard, offset, length, file_name in zip(bcif_shard, bcif_data_off, bcif_len, file_names):
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
        else:
            out_str_list = []
            for shard, offset, length in zip(bcif_shard, bcif_data_off, bcif_len):
                shard_path = os.path.join(self.shard_path, str(shard))
                try:
                    out_str = bcif_shard_to_mmCIF_str(shard_path, offset, length)
                    out_str_list.append(out_str)
                except Exception as e:
                    print(f"Error reading {shard_path} at offset {offset} with length {length}: {e}")
                    continue

        if as_str:
            # return list or if only one, return single string
            return out_str_list if len(out_str_list) > 1 else out_str_list[0]

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
                seq_list.append(None)
                continue

        return seq_list if len(seq_list) > 1 else seq_list[0]

    def __repr__(self):
        if self.n_row is None:
            self.nrow()
        return f"ProtLake at {self.path} with {self.n_row:,} structures"
