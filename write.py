import os, io, json, time, hashlib, zlib, sys, fcntl, msgpack, zstandard as zstd, atexit, shutil
from socket import gethostname
from dataclasses import dataclass
from typing import Iterator, Optional, TypedDict, Dict, Any, List, Tuple
import numpy as np
import pyarrow as pa
from deltalake import write_deltalake
from biotite.structure.io import load_structure
from biotite.structure.io.pdbx import BinaryCIFFile, set_structure, compress

from utils import ensure_dirs, get_protlake_dirs

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
    pa.field("bcif_len", pa.int32()),
    pa.field("json_shard", pa.string()),
    pa.field("json_off", pa.int64()),
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
    bcif_len: int
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

# --------------- CIF codec ---------------
def cif_to_bcif_bytes(cif_path: str, rtol: float = 1e-6, atol: float = 1e-4) -> bytes:
    atom_array = load_structure(cif_path, extra_fields=['b_factor'])
    bcif = BinaryCIFFile()
    set_structure(bcif, atom_array)
    bcif = compress(bcif, rtol=rtol, atol=atol)
    buf = io.BytesIO()
    bcif.write(buf)
    return buf.getvalue()

# --------------- small container format (PACK) ---------------
MAGIC   = b"PACK"
VERSION = 1

class PackRecord(TypedDict):
    id_hex: str
    id_bytes: bytes
    off: int
    length: int
    shard_path: str

class ShardPackWriter:
    """Shard pack writer. Claim-based exclusive ownership across processes is optional.

    Arguments:
        shard_dir: directory to hold .pack files
        prefix: shard filename prefix (e.g. "bcif-pack")
        max_bytes: target max size per shard
        use_claims: if True, use mkdir-based claim directories to avoid multiple processes
                    writing to the same shard. 
        claim_ttl: when using claims, consider a claim stale after this many seconds
    """
    def __init__(self, shard_dir: str, prefix: str = "bcif-pack", max_bytes: int = 1 << 30,
                 use_claims: bool = False, claim_ttl: int = 300):
        self.shard_dir = shard_dir
        self.prefix = prefix
        self.max_bytes = max_bytes
        self.use_claims = use_claims
        self.claim_ttl = claim_ttl
        self._current_shard: Optional[str] = None

        # only keep claim bookkeeping if user enabled claims
        if self.use_claims:
            self._owned_claims: set[str] = set()
            atexit.register(self._release_all_owned_claims)

    # ---------- helpers for claim mode ----------
    def _claim_dir(self, shard_path: str) -> str:
        return f"{shard_path}.claim"

    def _write_claim_meta(self, claim_dir: str) -> None:
        if not self.use_claims:
            return
        meta = {
            "hostname": gethostname(), 
            "pid": os.getpid(), 
            "ts": time.time(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "N/A"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "N/A"),
        }
        with open(os.path.join(claim_dir, "owner.json"), "w") as f:
            json.dump(meta, f)

    def _read_claim_meta(self, claim_dir: str) -> Optional[Dict[str, Any]]:
        if not self.use_claims:
            return None
        try:
            with open(os.path.join(claim_dir, "owner.json"), "r") as f:
                return json.load(f)
        except Exception:
            return None

    def _is_claim_stale(self, claim_dir: str) -> bool:
        if not self.use_claims:
            return False
        try:
            mtime = os.path.getmtime(claim_dir)
        except FileNotFoundError:
            return True
        return (time.time() - mtime) > self.claim_ttl

    def _try_mkdir_claim(self, claim_dir: str) -> bool:
        if not self.use_claims:
            return False
        try:
            os.mkdir(claim_dir)
        except FileExistsError:
            return False
        # created -> write metadata
        try:
            self._write_claim_meta(claim_dir)
        except Exception:
            pass
        self._owned_claims.add(claim_dir)
        return True

    def _steal_stale_claim(self, claim_dir: str, shard_path: str) -> bool:
        if not self.use_claims:
            return False
        if not os.path.exists(claim_dir):
            return False
        if not self._is_claim_stale(claim_dir):
            return False
        try:
            shutil.rmtree(claim_dir)
        except Exception:
            return False
        
        # After removing the claim, check shard size: if it's full, don't steal
        if os.path.exists(shard_path) and os.path.getsize(shard_path) >= self.max_bytes:
            # Another process finished the shard while this claim was stale — don't steal
            return False

        return self._try_mkdir_claim(claim_dir)

    def release_shard(self, shard_path: str) -> None:
        """Remove our claim on shard_path. No-op when claims are disabled."""
        if not self.use_claims:
            return
        claim_dir = self._claim_dir(shard_path)
        if claim_dir not in self._owned_claims:
            return
        try:
            shutil.rmtree(claim_dir)
        except Exception:
            pass
        self._owned_claims.discard(claim_dir)

    def _release_all_owned_claims(self) -> None:
        if not self.use_claims:
            return
        for claim_dir in list(self._owned_claims):
            try:
                shutil.rmtree(claim_dir)
            except Exception:
                pass
            self._owned_claims.discard(claim_dir)

    # ---------- shard selection ----------
    def choose_shard(self) -> str:
        """Return a shard path. If use_claims is True, claim it atomically.
        """
        # Reuse current shard if still valid
        if self._current_shard is not None:
            shard = self._current_shard
            claim_dir = self._claim_dir(shard)
            # Non-claim mode: just check size
            if not self.use_claims:
                if not os.path.exists(shard) or os.path.getsize(shard) < self.max_bytes:
                    return shard
                else:
                    self._current_shard = None
            else:
                # claim-mode: ensure we still own the claim and shard not full
                if (os.path.exists(claim_dir) and claim_dir in self._owned_claims
                    and (not os.path.exists(shard) or os.path.getsize(shard) < self.max_bytes)):
                    return shard
                # if claim was lost or shard is full -> drop current_shard and continue to pick a new one
                self._current_shard = None
        
        # No current shard: find a new one
        if not self.use_claims:
            # pick first non-existent or non-full shard
            i = 0
            while True:
                shard = os.path.join(self.shard_dir, f"{self.prefix}-{i:06d}.pack")
                if not os.path.exists(shard):
                    # don't set current_shard here until we actually create it in append,
                    fd = os.open(shard, os.O_CREAT | os.O_RDWR, 0o644)
                    os.close(fd)
                    self._current_shard = shard
                    return shard
                if os.path.getsize(shard) < self.max_bytes:
                    self._current_shard = shard
                    return shard
                i += 1

        # claim-mode behaviour
        i = 0
        while True:
            shard = os.path.join(self.shard_dir, f"{self.prefix}-{i:06d}.pack")
            claim_dir = self._claim_dir(shard)

            # If a claim exists, consider stealing if stale
            if os.path.exists(claim_dir):
                if self._is_claim_stale(claim_dir):
                    if self._steal_stale_claim(claim_dir, shard):
                        self._owned_claims.add(claim_dir)
                        # we now own the claim; ensure shard exists and return it
                        fd = os.open(shard, os.O_CREAT | os.O_RDWR, 0o644)
                        os.close(fd)
                        self._current_shard = shard
                        return shard
                    else:
                        # failed to steal (race); skip to next
                        i += 1
                        continue
                else:
                    # someone else owns an active claim; skip
                    i += 1
                    continue

            # No claim exists. If shard doesn't exist or is not full, try to create the claim
            if not os.path.exists(shard) or os.path.getsize(shard) < self.max_bytes:
                if self._try_mkdir_claim(claim_dir):
                    self._owned_claims.add(claim_dir)
                    # we own it; create/touch shard file
                    fd = os.open(shard, os.O_CREAT | os.O_RDWR, 0o644)
                    os.close(fd)
                    self._current_shard = shard
                    return shard
                else:
                    # lost race creating claim, try next
                    i += 1
                    continue

            # shard exists and is full -> next
            i += 1

    # ---------- append ----------
    def append(self, shard_path: str, payload: bytes, rec_id: Optional[bytes] = None) -> PackRecord:
        """Append a PACK record. If use_claims=True, we verify ownership and release the
           claim when the shard fills. Otherwise behaves the same as before.
        """
        if rec_id is None:
            rec_id = hashlib.sha256(payload).digest()
        id_hex = rec_id.hex()

        id_len = len(rec_id)
        data_len = len(payload)
        hdr = (MAGIC + bytes([VERSION]) +
               id_len.to_bytes(2, "big") +
               data_len.to_bytes(4, "big"))
        crc = (zlib.crc32(payload) & 0xFFFFFFFF).to_bytes(4, "big")
        record = b"".join([hdr, rec_id, payload, crc])

        claim_dir = self._claim_dir(shard_path) if self.use_claims else None
        if self.use_claims and os.path.exists(claim_dir) and claim_dir not in self._owned_claims:
            # Defensive: another process owns it
            raise RuntimeError(f"Shard {shard_path} is claimed by another process ({claim_dir}).")

        fd = os.open(shard_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            off = os.lseek(fd, 0, os.SEEK_END)
            os.pwrite(fd, record, off)
            data_off = off + len(hdr) + id_len
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

        # If this write made the shard reach or exceed max_bytes, release the claim (if used)
        if self.use_claims:
            try:
                if os.path.getsize(shard_path) >= self.max_bytes:
                    if claim_dir in self._owned_claims:
                        self.release_shard(shard_path)
            except Exception:
                pass

        return {"id_hex": id_hex, "id_bytes": rec_id, "off": data_off, "length": data_len, "shard_path": os.path.basename(shard_path)}

# --------------- Delta appenders ---------------
class DeltaAppender:
    """Buffer rows per schema and flush to a Delta table via deltalake."""
    def __init__(self, table_path: str, schema: pa.Schema, batch_size: int = 2500):
        self.table_uri = f"file://{os.path.abspath(table_path)}"
        self.schema = schema
        self.batch_size = batch_size
        self.buf: Dict[str, List[Any]] = {f.name: [] for f in schema}

    def add_row(self, row: Dict[str, Any]) -> None:
        # Strictly adhere to schema fields; missing fields become None
        for f in self.schema:
            self.buf[f.name].append(row.get(f.name))
        if len(self.buf[self.schema[0].name]) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.buf[self.schema[0].name]:
            return
        tbl = pa.Table.from_pydict(self.buf, schema=self.schema)
        write_deltalake(self.table_uri, tbl, mode="append")
        for k in self.buf:
            self.buf[k].clear()

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
                print(f"WARNING: Missing CIF file {cif_path}, skipping")
                continue
            if not os.path.exists(light_json_path):
                print(f"WARNING: Missing light JSON file {light_json_path}, skipping")
                continue
            if not os.path.exists(heavy_json_path):
                print(f"WARNING: Missing heavy JSON file {heavy_json_path}, skipping")
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
    verbose: bool = False
    rtol: float = 1e-6
    atol: float = 1e-4
    bcif_shard_prefix: str = "bcif-pack"
    json_shard_prefix: str = "json-pack"
    claim_mode: bool = False
    claim_ttl: int = 300

class AF3IngestPipeline:
    def __init__(self, cfg: IngestConfig):
        self.cfg = cfg
        self.shard_dir, self.delta_path = get_protlake_dirs(cfg.out_path)
        ensure_dirs([self.shard_dir, self.delta_path])

        self.bcif_packer = ShardPackWriter(self.shard_dir, prefix=cfg.bcif_shard_prefix, max_bytes=cfg.shard_size,
                                          use_claims=cfg.claim_mode, claim_ttl=cfg.claim_ttl)
        self.json_packer = ShardPackWriter(self.shard_dir, prefix=cfg.json_shard_prefix, max_bytes=cfg.shard_size,
                                          use_claims=cfg.claim_mode, claim_ttl=cfg.claim_ttl)
        self.delta_appender = DeltaAppender(self.delta_path, CORE_SCHEMA, batch_size=cfg.batch_size_metadata)
        self.loader = MetadataLoader()
        self.scanner = RunScanner()

    def run(self, input_dirs: List[str]) -> None:
        for input_dir in input_dirs:
            bcif_shard_path_for_run = self.bcif_packer.choose_shard()  # guarantees same shard per input_dir
            json_shard_path_for_run = self.json_packer.choose_shard()  # guarantees same shard per input_dir
            if self.cfg.verbose:
                print(f"[run] input_dir={input_dir} -> bcif_shard={os.path.basename(bcif_shard_path_for_run)}, json_shard={os.path.basename(json_shard_path_for_run)}")

            items_found = False
            for item in self.scanner.iter_items(input_dir):
                items_found = True
                bcif_bytes = cif_to_bcif_bytes(item["cif_path"], rtol=self.cfg.rtol, atol=self.cfg.atol)

                # Append to shard
                bcif_pack = self.bcif_packer.append(bcif_shard_path_for_run, bcif_bytes, rec_id=None)
                if self.cfg.verbose:
                    print(f"packed id={bcif_pack['id_hex'][:12]}…  shard={os.path.basename(bcif_pack['shard_path'])} off={bcif_pack['off']} len={bcif_pack['length']}  src={item['src_dir']}")

                heavy = self.loader.load_heavy(input_dir, item["run_name"], item["subdir"])
                json_pack_bytes = msgpack.packb(heavy, use_bin_type=True)
                compressor = zstd.ZstdCompressor(level=3)
                json_pack_bytes_comp = compressor.compress(json_pack_bytes)  

                json_pack = self.json_packer.append(json_shard_path_for_run, json_pack_bytes_comp, rec_id=None)
                if self.cfg.verbose:
                    print(f"packed id={json_pack['id_hex'][:12]}…  shard={os.path.basename(json_pack['shard_path'])} off={json_pack['off']} len={json_pack['length']}  src={item['src_dir']}")

                # Load metadata
                light = self.loader.load_light(input_dir, item["run_name"], item["subdir"])

                # Build rows
                common = dict(
                    id=bcif_pack["id_bytes"],
                    id_hex=bcif_pack["id_hex"],
                    name=item["run_name"],
                    sample=int(item["sample"]),
                    seed=int(item["seed"]),
                    bcif_shard=bcif_pack["shard_path"],
                    bcif_off=int(bcif_pack["off"]),
                    bcif_len=int(bcif_pack["length"]),
                    json_shard=json_pack["shard_path"],
                    json_off=int(json_pack["off"]),
                    json_len=int(json_pack["length"]),
                )

                delta_row: CoreRow = {**common, **{k: light.get(k) for k in [f.name for f in CORE_SCHEMA] if k in light}}
                self.delta_appender.add_row(delta_row)

            if not items_found:
                print(f"WARNING: No valid subdirs with CIFs found under {input_dir}")

        # Final flush
        self.delta_appender.flush()

