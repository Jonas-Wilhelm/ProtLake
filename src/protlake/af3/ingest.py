import os, io, json, time, hashlib, zlib, fcntl, msgpack, atexit, shutil, uuid, random, warnings, atexit, logging
from typing import Iterator, Optional, TypedDict, Dict, Any, List, Tuple
import zstandard as zstd
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds

from socket import gethostname
from dataclasses import dataclass, field

from protlake.write.core import (
    ShardPackWriter,
    DeltaAppender,
    RetryConfig,
    LeaseMismatchRetry,
    load_delta_table_with_retries,
)

logger = logging.getLogger(__name__)

from deltalake import DeltaTable
from deltalake.exceptions import DeltaError

from protlake.utils import ensure_dirs, get_protlake_dirs, cif_to_bcif_bytes, cif_bytes_to_bcif_bytes, is_retryable_delta_error

# --------------- Arrow schemas ---------------
list_f32 = pa.list_(pa.float32())
lol_f32  = pa.list_(list_f32)
lol_i16  = pa.list_(pa.list_(pa.int16()))
list_str = pa.list_(pa.string())
list_i32 = pa.list_(pa.int32())

CORE_SCHEMA = pa.schema([
    pa.field("id", pa.binary(32)),         # sha256 digest (binary, not hex)
    pa.field("id_hex", pa.string()),       # hex representation of the id
    pa.field("name", pa.string()),
    pa.field("sample", pa.int32()),
    pa.field("seed", pa.int32()),
    pa.field("bcif_shard", pa.string()),
    pa.field("bcif_off", pa.int64()),
    pa.field("bcif_data_off", pa.int64()),
    pa.field("bcif_len", pa.int32()),
    pa.field("json_shard", pa.string()),
    pa.field("json_off", pa.int64()),
    pa.field("json_data_off", pa.int64()),
    pa.field("json_len", pa.int32()),
    pa.field("chain_iptm", list_f32),
    pa.field("chain_pair_iptm", lol_f32),
    pa.field("chain_pair_pae_min", lol_f32),
    pa.field("chain_ptm", list_f32),
    pa.field("fraction_disordered", pa.float32()),
    pa.field("has_clash", pa.float32()),
    pa.field("iptm", pa.float32()),
    pa.field("ptm", pa.float32()),
    pa.field("ranking_score", pa.float32()),
])

# --------------- Typed rows ---------------
class CoreRow(TypedDict, total=False):
    id: bytes
    id_hex: str
    name: str
    sample: int
    seed: int
    bcif_shard: str
    bcif_off: int
    bcif_data_off: int
    bcif_len: int
    json_shard: str
    json_off: int
    json_data_off: int
    json_len: int
    chain_iptm: List[float]
    chain_pair_iptm: List[List[float]]
    chain_pair_pae_min: List[List[float]]
    chain_ptm: List[float]
    fraction_disordered: float
    has_clash: float
    iptm: float
    ptm: float
    ranking_score: float

# --------------- Parsing helpers ---------------
def parse_seed_sample(subdir_name: str) -> Tuple[int, int]:
    """
    Expect patterns like 'seed-123_sample-0' or similar.
    """
    seed, sample = 0, 0
    parts = subdir_name.split("_")
    for p in parts:
        if p.startswith("seed-"):
            seed = int(p.replace("seed-", ""))
        if p.startswith("sample-"):
            sample = int(p.replace("sample-", ""))
    return seed, sample

# --------------- Delta error handling ---------------
@dataclass
class RetryConfig:
    max_retries: int = 150
    base_sleep: float = 0.1
    max_sleep: float = 5.5
    jitter: float = 0.2

def load_delta_table_with_retries(delta_path, base_sleep, jitter, max_sleep, max_retries):
    """Load a DeltaTable with retries on known concurrency errors."""
    delay = base_sleep
    for attempt in range(1, max_retries + 1):
        try:
            dt = DeltaTable(f"file://{os.path.abspath(delta_path)}")
            break  # success
        except DeltaError as e:
            if attempt < max_retries and is_retryable_delta_error(e):
                sleep_for = delay * (1 + jitter * random.uniform(-1, 1))
                logger.warning(f"Loading deltatable failed with retryable error (attempt {attempt}/{max_retries}). Retrying in {sleep_for:.2f}s... error: {e}")
                time.sleep(sleep_for)
                delay = min(max_sleep, delay * 1.5)
                continue
            # not retryable or out of retries
            logger.error(f"Loading deltatable failed permanently after {attempt} attempts.")
            raise
    return dt

# --------------- JSON helpers ---------------
def _serialize_af3_StructureConfidence_dict(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [_serialize_af3_StructureConfidence_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _serialize_af3_StructureConfidence_dict(v) for k, v in obj.items()}
    return obj

def _round_af3_StructureConfidence_dict(d):
    """
    Round floats in AF3 StructureConfidence dict to 1 or 2 decimal places.
    Drastically reduces compressed size about 5 fold.
    """
    def round_recursive(x, digits):
        if isinstance(x, float):
            return round(x, digits)
        elif isinstance(x, list):
            return [round_recursive(i, digits) for i in x]
        else:
            return x

    for k, v in d.items():
        digits = 1 if k == "pae" else 2
        d[k] = round_recursive(v, digits)

    return d

# --------------- Metadata loader ---------------
class MetadataLoader:
    """Loads *_summary_confidences.json and *_confidences.json and normalizes shapes."""
    def load_light(self, dir_path: str, name: str, subdir: str) -> Dict[str, Any]:
        path = os.path.join(dir_path, subdir, f"{name}_{subdir}_summary_confidences.json")
        if not os.path.exists(path): return {}
        with open(path, "r") as f:
            data = json.load(f)
        # light payload is expected to match CORE_SCHEMA keys
        return data

    def load_heavy(self, dir_path: str, name: str, subdir: str) -> Dict[str, Any]:
        path = os.path.join(dir_path, subdir, f"{name}_{subdir}_confidences.json")
        if not os.path.exists(path): return {}
        with open(path, "r") as f:
            data = json.load(f)

        return data

# --------------- Scanner ---------------
class WorkItem(TypedDict):
    run_name: str
    subdir: str
    seed: int
    sample: int
    cif_path: str
    src_dir: str

class RunScanner:
    """Yields WorkItem with resolved CIF path for each AF3 model subdir."""
    def iter_items(self, input_dir: str) -> Iterator[WorkItem]:
        run_name = os.path.basename(os.path.normpath(input_dir))
        for e in os.scandir(input_dir):
            if not e.is_dir():
                continue
            subdir = e.name
            cif_path        = os.path.join(input_dir, subdir, f"{run_name}_{subdir}_model.cif")
            heavy_json_path = os.path.join(input_dir, subdir, f"{run_name}_{subdir}_confidences.json")
            light_json_path = os.path.join(input_dir, subdir, f"{run_name}_{subdir}_summary_confidences.json")
            if not os.path.exists(cif_path):
                logger.warning(f"Missing CIF file {cif_path}, skipping")
                continue
            if not os.path.exists(light_json_path):
                logger.warning(f"Missing light JSON file {light_json_path}, skipping")
                continue
            if not os.path.exists(heavy_json_path):
                logger.warning(f"Missing heavy JSON file {heavy_json_path}, skipping")
                continue
            seed, sample = parse_seed_sample(subdir)
            yield WorkItem(
                run_name=run_name,
                subdir=subdir,
                seed=seed,
                sample=sample,
                cif_path=cif_path,
                src_dir=os.path.join(input_dir, subdir),
            )

# --------------- Config & pipeline ---------------

@dataclass
class IngestConfig:
    out_path: str
    batch_size_metadata: int = 2500
    shard_size: int = 1 << 30
    rtol: float = 1e-6
    atol: float = 1e-4
    bcif_shard_prefix: str = "bcif-pack"
    json_shard_prefix: str = "json-pack"
    claim_mode: bool = False
    claim_ttl: int = 60 * 30
    dont_ingest_if_exists: bool = True
    delete_input_after_ingest: bool = False
    write_json_shards: bool = True
    retry_conf: RetryConfig = field(default_factory=RetryConfig)

class AF3IngestPipeline:
    def __init__(self, cfg: IngestConfig):
        self.cfg = cfg
        self.shard_dir, self.delta_path = get_protlake_dirs(cfg.out_path)
        ensure_dirs([self.shard_dir, self.delta_path])

        self.bcif_packer = ShardPackWriter(self.shard_dir, prefix=cfg.bcif_shard_prefix, max_bytes=cfg.shard_size,
                                          use_claims=cfg.claim_mode, claim_ttl=cfg.claim_ttl)
        if cfg.write_json_shards:
            self.json_packer = ShardPackWriter(self.shard_dir, prefix=cfg.json_shard_prefix, max_bytes=cfg.shard_size,
                                          use_claims=cfg.claim_mode, claim_ttl=cfg.claim_ttl)
        self.delta_appender = DeltaAppender(self.delta_path, CORE_SCHEMA, batch_size=cfg.batch_size_metadata, retry_config=cfg.retry_conf)
        self.loader = MetadataLoader()
        self.scanner = RunScanner()

        warnings.filterwarnings(
            "ignore", 
            message="Attribute 'auth_.*_id' not found within 'atom_site' category", 
            category=UserWarning, 
            module="biotite.structure"
        )

    def run(self, input_dirs: List[str]) -> None:
        if self.cfg.dont_ingest_if_exists and os.path.exists(os.path.join(self.delta_path, "_delta_log")):
            # Skip already ingested runs
            logger.info("Checking for already ingested runs...")
            check_start = time.time()
            dt = DeltaTable(f"file://{os.path.abspath(self.delta_path)}")
            existing_tbl = dt.to_pyarrow_table(columns=["name"])
            existing_names = set(existing_tbl.column("name").to_pylist())
            non_existing_input_dirs = [d for d in input_dirs if os.path.basename(os.path.normpath(d)) not in existing_names]
            check_end = time.time()
            logger.info(f"Checked {len(input_dirs)} input dirs against {len(existing_names)} existing names in {check_end - check_start:.2f} seconds.")
            if not non_existing_input_dirs:
                logger.info(f"All {len(input_dirs)} input runs already ingested! Exiting.")
                return
            else:
                logger.info(f"Skipping {len(input_dirs) - len(non_existing_input_dirs)} already ingested runs.")
                skipped = [os.path.basename(os.path.normpath(d)) for d in input_dirs if os.path.basename(os.path.normpath(d)) in existing_names]
                logger.debug(f"The following runs were skipped: {', '.join(skipped)}")
                input_dirs = non_existing_input_dirs

        for input_dir in input_dirs:
            bcif_shard_path_for_run = self.bcif_packer.choose_shard()  # guarantees same shard per input_dir
            if self.cfg.write_json_shards:
                json_shard_path_for_run = self.json_packer.choose_shard()  # guarantees same shard per input_dir

            items_found = False
            for item in self.scanner.iter_items(input_dir):
                items_found = True
                bcif_bytes = cif_to_bcif_bytes(item["cif_path"], rtol=self.cfg.rtol, atol=self.cfg.atol)

                # Load metadata
                light = self.loader.load_light(input_dir, item["run_name"], item["subdir"])

                if self.cfg.write_json_shards:
                    heavy = self.loader.load_heavy(input_dir, item["run_name"], item["subdir"])
                    json_pack_bytes = msgpack.packb(heavy, use_bin_type=True)
                    compressor = zstd.ZstdCompressor(level=12)
                    json_pack_bytes_comp = compressor.compress(json_pack_bytes)
                    # Append json to shard
                    json_pack = self.json_packer.append(json_shard_path_for_run, json_pack_bytes_comp, rec_id=None)
                    logger.debug(f"packed id={json_pack['id_hex'][:12]}…  shard={os.path.basename(json_pack['shard_path'])} off={json_pack['off']} len={json_pack['length']}  src={item['src_dir']}")

                # Append cif to shard
                bcif_pack = self.bcif_packer.append(bcif_shard_path_for_run, bcif_bytes, rec_id=None)
                logger.debug(f"packed id={bcif_pack['id_hex'][:12]}…  shard={os.path.basename(bcif_pack['shard_path'])} off={bcif_pack['off']} len={bcif_pack['length']}  src={item['src_dir']}")

                # Build rows
                common = dict(
                    id=bcif_pack["id_bytes"],
                    id_hex=bcif_pack["id_hex"],
                    name=item["run_name"],
                    sample=int(item["sample"]),
                    seed=int(item["seed"]),
                    bcif_shard=bcif_pack["shard_path"],
                    bcif_off=int(bcif_pack["off"]),
                    bcif_data_off=int(bcif_pack["data_off"]),
                    bcif_len=int(bcif_pack["length"]),
                )
                if self.cfg.write_json_shards:
                    common.update(
                        json_shard=json_pack["shard_path"],
                        json_off=int(json_pack["off"]),
                        json_data_off=int(json_pack["data_off"]),
                        json_len=int(json_pack["length"]),
                    )
                else:
                    common.update(json_shard="", json_off=0, json_data_off=0, json_len=0)

                delta_row: CoreRow = {**common, **{k: light.get(k) for k in [f.name for f in CORE_SCHEMA] if k in light}}
                self.delta_appender.add_row(delta_row)

            if not items_found:
                logger.warning(f"No valid subdirs with CIFs found under {input_dir}")
            # Delete the input_dir recursively to save space
            if self.cfg.delete_input_after_ingest:
                try:
                    shutil.rmtree(input_dir)
                    logger.debug(f"Deleted input directory {input_dir} to save space.")
                except Exception as e:
                    logger.warning(f"Failed to delete input directory {input_dir}: {e}")
                    
        # Final flush
        self.delta_appender.flush()

class AF3ProtlakeWriter:
    def __init__(self, cfg: IngestConfig):
        self.cfg = cfg
        self.shard_dir, self.delta_path = get_protlake_dirs(cfg.out_path)
        ensure_dirs([self.shard_dir, self.delta_path])

        self.bcif_packer = ShardPackWriter(
            self.shard_dir, 
            prefix=cfg.bcif_shard_prefix, 
            max_bytes=cfg.shard_size,
            use_claims=cfg.claim_mode, 
            claim_ttl=cfg.claim_ttl,
        )

        if cfg.write_json_shards:
            self.json_packer = ShardPackWriter(
                self.shard_dir, 
                prefix=cfg.json_shard_prefix, 
                max_bytes=cfg.shard_size,
                use_claims=cfg.claim_mode, 
                claim_ttl=cfg.claim_ttl,
            )

        self.delta_appender = DeltaAppender(self.delta_path, CORE_SCHEMA, batch_size=cfg.batch_size_metadata, retry_config=cfg.retry_conf)
        self.loader = MetadataLoader()
        self.scanner = RunScanner()

        warnings.filterwarnings(
            "ignore", 
            message="Attribute 'auth_.*_id' not found within 'atom_site' category", 
            category=UserWarning, 
            module="biotite.structure"
        )

        # Ensure final flush on exit
        atexit.register(self.finalize)

    def check_exists(self, name: str, seed: int, sample_idx: int) -> bool:
        if not DeltaTable.is_deltatable(f"file://{os.path.abspath(self.delta_path)}"):
            logger.info("Delta table does not exist yet; no entries found.")
            return False
        
        dt = load_delta_table_with_retries(
            delta_path = self.delta_path,
            base_sleep = self.cfg.retry_conf.base_sleep,
            jitter = self.cfg.retry_conf.jitter,
            max_sleep = self.cfg.retry_conf.max_sleep,
            max_retries = self.cfg.retry_conf.max_retries,
        )

        pa_dataset = dt.to_pyarrow_dataset()
        cond = (ds.field("name") == name) & (ds.field("seed") == seed) & (ds.field("sample") == sample_idx)
        scanner = pa_dataset.scanner(filter=cond, columns=["name", "seed", "sample"])

        for batch in scanner.to_batches():
            if batch.num_rows > 0:
                return True

        return False

    # def get_existing_outputs(self, expected_outputs: List[Tuple[str, Tuple[int], Tuple[int]]]) -> set[str]:
    #     """Given a list of (name, seeds, sample_idx) tuples, return the names that have complete outputs."""
    #     if not DeltaTable.is_deltatable(f"file://{os.path.abspath(self.delta_path)}"):
    #         raise RuntimeError("Delta table does not exist yet. Create (empty) protlake first.")

    #     dt = load_delta_table_with_retries(
    #         delta_path = self.delta_path,
    #         base_sleep = self.cfg.retry_conf.base_sleep,
    #         jitter = self.cfg.retry_conf.jitter,
    #         max_sleep = self.cfg.retry_conf.max_sleep,
    #         max_retries = self.cfg.retry_conf.max_retries,
    #     )

    #     pa_dataset = dt.to_pyarrow_dataset()
    #     existing_names = set()
    #     for name, seeds, sample_idx in expected_outputs:
    #         expr = ds.field("name") == name
    #         scanner = pa_dataset.scanner(filter=expr, columns=["seed", "sample"])

    #         target = {(s, si) for s in seeds for si in sample_idx}
    #         found = set()

    #         for batch in scanner.to_batches():
    #             seeds_batch = batch.column("seed").to_pylist()
    #             samples_batch = batch.column("sample").to_pylist()
    #             for s_val, samp_val in zip(seeds_batch, samples_batch):
    #                 tup = (s_val, samp_val)
    #                 if tup in target:
    #                     found.add(tup)
    #             if len(found) == len(target):   # early stop as soon as we've found all
    #                 break
            
    #         if len(found) == len(target):
    #             print(f"Name={name}: all {len(target)} entries found.")
    #             existing_names.add(name)
    #         else:
    #             if len(found) == 0:
    #                 print(f"Name={name}: no entries found.")
    #             else:
    #                 missing = sorted(target - found)
    #                 print(f"Name={name}: only {len(found)}/{len(target)} entries found. Missing: {missing}")

    #     return existing_names

    def get_existing_outputs(self, expected_outputs: List[Tuple[str, Tuple[int], Tuple[int]]], error_on_not_exist: bool = False) -> set[str]:
        """Given a list of (name, seeds, sample_idx) tuples, return the names that have complete outputs."""
        # TODO add option to delete incomplete entries
        if not DeltaTable.is_deltatable(f"file://{os.path.abspath(self.delta_path)}"):
            if error_on_not_exist:
                raise FileNotFoundError("Delta table does not exist yet. Create (empty) protlake first.")
            else: # no table -> no entries found -> return empty set
                logger.info("Delta table does not exist yet; no entries found.")
                return set()

        dt = load_delta_table_with_retries(
            delta_path = self.delta_path,
            base_sleep = self.cfg.retry_conf.base_sleep,
            jitter = self.cfg.retry_conf.jitter,
            max_sleep = self.cfg.retry_conf.max_sleep,
            max_retries = self.cfg.retry_conf.max_retries,
        )

        targets = {
            name: {(s, si) for s in seeds for si in sample_idx}
            for name, seeds, sample_idx in expected_outputs
        }

        found = {name: set() for name in targets}
        remaining = sum(len(t) for t in targets.values())

        pa_dataset = dt.to_pyarrow_dataset()
        names = list(targets.keys())

        scanner = pa_dataset.scanner(filter=ds.field("name").isin(names), columns=["name", "seed", "sample"])
        for batch in scanner.to_batches():
            name_batch = batch.column("name").to_pylist()
            seed_batch = batch.column("seed").to_pylist()
            sample_batch = batch.column("sample").to_pylist()

            for nm, s_val, samp_val in zip(name_batch, seed_batch, sample_batch):
                if nm in targets:
                    tup = (s_val, samp_val)
                    if tup in targets[nm] and tup not in found[nm]:
                        found[nm].add(tup)
                        remaining -= 1

            if remaining == 0:
                break  # early exit once we've found every target tuple

        existing_names = set()
        for name, target_set in targets.items():
            if found[name] == target_set: # complete
                logger.info(f"Name={name}: all {len(target_set)} entries found.")
                existing_names.add(name)
            else: # incomplete
                if not found[name]: # none found
                    logger.info(f"Name={name}: no entries found.")
                else: # partial found
                    missing = sorted(target_set - found[name])
                    logger.info(f"Name={name}: only {len(found[name])}/{len(target_set)} entries found. Missing: {missing}")

        return existing_names


    def check_exists_complete(self, name: str, seeds: tuple[int], sample_idx: tuple[int]) -> Tuple[bool, Optional[str]]:
        if not os.path.exists(os.path.join(self.delta_path, "_delta_log")):
            logger.info("Delta table does not exist yet; no entries found.")
            return False, 'all_missing'
        
        dt = load_delta_table_with_retries(
            delta_path = self.delta_path,
            base_sleep = self.cfg.retry_conf.base_sleep,
            jitter = self.cfg.retry_conf.jitter,
            max_sleep = self.cfg.retry_conf.max_sleep,
            max_retries = self.cfg.retry_conf.max_retries,
        )

        pa_dataset = dt.to_pyarrow_dataset()

        expr = ds.field("name") == name
        scanner = pa_dataset.scanner(filter=expr, columns=["seed", "sample"])

        target = {(s, si) for s in seeds for si in sample_idx}
        found = set()

        for batch in scanner.to_batches():
            seeds_batch = batch.column("seed").to_pylist()
            samples_batch = batch.column("sample").to_pylist()
            for s_val, samp_val in zip(seeds_batch, samples_batch):
                tup = (s_val, samp_val)
                if tup in target:
                    found.add(tup)
            if len(found) == len(target):   # early stop as soon as we've found all combos
                break
        
        missing = sorted(target - found)
        if len(missing) == 0:
            return True, None
        if len(missing) == len(target):
            return False, 'all_missing'
        else:
            logger.info(f"Found only {len(found)}/{len(target)} entries for name={name}. Missing: {missing}")
            return False, 'partial_missing'
    
    def remove_entries(self, name: str) -> None:
        if not os.path.exists(os.path.join(self.delta_path, "_delta_log")):
            logger.info("Delta table does not exist yet; no entries found.")
            return
        dt = DeltaTable(f"file://{os.path.abspath(self.delta_path)}")
        dt.delete(f"name = '{name}'")

    def write(
        self,
        cif: bytes,
        summary_confidences: dict,
        confidences: dict,
        name: str,
        sample_idx: int,
        seed: int,
        max_lease_retries: int = 2000,
    ) -> None:
        # Prepare data once (expensive operations)
        bcif_bytes = cif_bytes_to_bcif_bytes(cif, rtol=self.cfg.rtol, atol=self.cfg.atol)
        
        json_pack_bytes_comp = None
        if self.cfg.write_json_shards:
            json_pack_bytes = msgpack.packb(
                _round_af3_StructureConfidence_dict(
                    _serialize_af3_StructureConfidence_dict(
                        confidences
                    )
                ),
                use_bin_type=True,
                use_single_float=True
            )
            compressor = zstd.ZstdCompressor(level=12)
            json_pack_bytes_comp = compressor.compress(json_pack_bytes)

        # Retry loop for lease mismatch scenarios (e.g., after SLURM preemption)
        for attempt in range(1, max_lease_retries + 1):
            try:
                # pick shards for this run
                bcif_shard_path = self.bcif_packer.choose_shard()
                if self.cfg.write_json_shards:
                    json_shard_path = self.json_packer.choose_shard()

                if self.cfg.write_json_shards:
                    # Append json to shard
                    json_pack = self.json_packer.append(json_shard_path, json_pack_bytes_comp, rec_id=None)
                    logger.debug(f"packed id={json_pack['id_hex'][:12]}…  shard={os.path.basename(json_pack['shard_path'])} off={json_pack['off']} len={json_pack['length']}")

                # Append cif to shard
                bcif_pack = self.bcif_packer.append(bcif_shard_path, bcif_bytes, rec_id=None)
                logger.debug(f"packed id={bcif_pack['id_hex'][:12]}…  shard={os.path.basename(bcif_pack['shard_path'])} off={bcif_pack['off']} len={bcif_pack['length']}")
                
                # Success - break out of retry loop
                break
                
            except LeaseMismatchRetry as e:
                if attempt >= max_lease_retries:
                    raise RuntimeError(
                        f"Failed to write after {max_lease_retries} lease retry attempts. "
                        f"All shards appear contested. Last error: {e}"
                    )
                # Log and retry with exponential backoff
                backoff = min(0.1 * (1.5 ** min(attempt, 10)), 5.0) * random.uniform(0.8, 1.2)
                logger.warning(
                    f"Lease mismatch on attempt {attempt}/{max_lease_retries}, "
                    f"retrying in {backoff:.2f}s..."
                )
                time.sleep(backoff)
                continue

        # Build rows
        common = dict(
            id=bcif_pack["id_bytes"],
            id_hex=bcif_pack["id_hex"],
            name=name,
            sample=sample_idx,
            seed=seed,
            bcif_shard=bcif_pack["shard_path"],
            bcif_off=int(bcif_pack["off"]),
            bcif_data_off=int(bcif_pack["data_off"]),
            bcif_len=int(bcif_pack["length"]),
        )
        if self.cfg.write_json_shards:
            common.update(
                json_shard=json_pack["shard_path"],
                json_off=int(json_pack["off"]),
                json_data_off=int(json_pack["data_off"]),
                json_len=int(json_pack["length"]),
            )
        else:
            common.update(json_shard="", json_off=0, json_data_off=0, json_len=0)

        summary_confidences = _serialize_af3_StructureConfidence_dict(summary_confidences)
        delta_row: CoreRow = {**common, **{k: summary_confidences.get(k) for k in [f.name for f in CORE_SCHEMA] if k in summary_confidences}}
        self.delta_appender.add_row(delta_row)
    
    def finalize(self) -> None:
        logger.info("Finalizing AF3ProtlakeWriter")
        logger.info(f"  Flushing {self.delta_appender.row_count()} rows")
        self.delta_appender.flush()
