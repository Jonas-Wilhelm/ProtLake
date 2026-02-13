import os, json, time, hashlib, zlib, fcntl, atexit, shutil, uuid, random, atexit, logging
from typing import Optional, TypedDict, Dict, Any, List, Tuple
import pyarrow as pa

from socket import gethostname
from dataclasses import dataclass

from deltalake import write_deltalake, DeltaTable
from deltalake.exceptions import DeltaError
from protlake.utils import is_retryable_delta_error

logger = logging.getLogger(__name__)

# --------------- Custom exceptions ---------------
class LeaseMismatchRetry(Exception):
    """Raised when a lease mismatch is detected, signaling the caller should retry with a new shard."""
    pass

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
        self.shard_dir = os.path.realpath(shard_dir)
        self.prefix = prefix
        self.max_bytes = max_bytes
        self.use_claims = use_claims
        self.claim_ttl = claim_ttl
        self._current_shard: Optional[str] = None
        self._process_id = f"{gethostname()}:{os.getpid()}"  # for debug output

        # only keep claim bookkeeping if user enabled claims
        if self.use_claims:
            self._owned_claims: Dict[str, str] = {}
            atexit.register(self._release_all_owned_claims)

    # ---------- helpers for claim mode ----------
    def _claim_dir(self, shard_path: str) -> str:
        return f"{shard_path}.claim"

    def _write_claim_meta(self, claim_dir: str, lease_id: str) -> None:
        if not self.use_claims:
            return
        meta = {
            "hostname": gethostname(),
            "pid": os.getpid(),
            "ts": time.time(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "N/A"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "N/A"),
            "lease_id": lease_id,
        }
        tmp = os.path.join(claim_dir, "owner.json.tmp")
        final = os.path.join(claim_dir, "owner.json")
        with open(tmp, "w") as f:
            json.dump(meta, f)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp, final) # atomic in contrast to regular write
        # fsync the directory so other clients see the new entry more quickly
        try:
            dirfd = os.open(claim_dir, os.O_RDONLY)
            try:
                os.fsync(dirfd)
            finally:
                os.close(dirfd)
        except Exception:
            pass

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
        now = time.time()
        age = now - mtime
        is_stale = age > self.claim_ttl
        logger.debug(f"[{self._process_id}] _is_claim_stale: {os.path.basename(claim_dir)} age={age:.1f}s ttl={self.claim_ttl}s stale={is_stale}")
        return is_stale

    def _try_mkdir_claim(self, claim_dir: str) -> bool:
        if not self.use_claims:
            return False
        try:
            os.mkdir(claim_dir)
            logger.debug(f"[{self._process_id}] _try_mkdir_claim: created {os.path.basename(claim_dir)}")
        except FileExistsError:
            logger.debug(f"[{self._process_id}] _try_mkdir_claim: {os.path.basename(claim_dir)} already exists")
            return False
        # created -> write metadata
        # Generate a new lease token
        lease_id = uuid.uuid4().hex
        try:
            self._write_claim_meta(claim_dir, lease_id)
        except Exception as e:
            # If we can't write metadata reliably, back out of the claim
            logger.debug(f"[{self._process_id}] _try_mkdir_claim: write_claim_meta failed: {e}")
            try:
                shutil.rmtree(claim_dir)
            except Exception:
                pass
            return False

        # Track our lease string in-process
        self._owned_claims[claim_dir] = lease_id
        logger.debug(f"[{self._process_id}] _try_mkdir_claim: SUCCESS - claimed {os.path.basename(claim_dir)}, lease_id={lease_id[:12]}...")
        return True

    def _steal_stale_claim(self, claim_dir: str, shard_path: str) -> bool:
        if not self.use_claims:
            return False
        if not os.path.exists(claim_dir):
            return False
        
        # Read existing claim metadata before checking staleness
        old_meta = self._read_claim_meta(claim_dir)
        logger.debug(f"[{self._process_id}] _steal_stale_claim: attempting to steal {os.path.basename(claim_dir)}, old_meta={old_meta}")
        
        if not self._is_claim_stale(claim_dir):
            logger.debug(f"[{self._process_id}] _steal_stale_claim: {os.path.basename(claim_dir)} not stale (first check), aborting")
            return False
        # To avoid race conditions, sleep a small random amount before proceeding
        time.sleep(random.uniform(0.25, 1.5))
        # Check again if still stale
        if not os.path.exists(claim_dir):
            logger.debug(f"[{self._process_id}] _steal_stale_claim: {os.path.basename(claim_dir)} disappeared after sleep")
            return False
        if not self._is_claim_stale(claim_dir):
            logger.debug(f"[{self._process_id}] _steal_stale_claim: {os.path.basename(claim_dir)} no longer stale (second check), aborting")
            return False
        
        # If the shard is already full, don't steal
        if os.path.exists(shard_path) and os.path.getsize(shard_path) >= self.max_bytes:
            logger.debug(f"[{self._process_id}] _steal_stale_claim: shard {os.path.basename(shard_path)} is full, skipping")
            return False

        stale_name = claim_dir + f".stale.{os.getpid()}.{int(time.time())}"
        # move claim_dir to a temporary name (atomic on same filesystem)
        try:
            os.rename(claim_dir, stale_name)
            logger.debug(f"[{self._process_id}] _steal_stale_claim: renamed {os.path.basename(claim_dir)} -> {os.path.basename(stale_name)}")
        except Exception as e:
            # rename failed (race or NFS weirdness) -> give up for now
            logger.debug(f"[{self._process_id}] _steal_stale_claim: rename failed: {e}")
            return False
        try:
            os.mkdir(claim_dir)
        except Exception as e:
            # if someone beat us to it, try to clean up stale_name then give up
            logger.debug(f"[{self._process_id}] _steal_stale_claim: mkdir failed after rename: {e}")
            try:
                shutil.rmtree(stale_name)
            except Exception:
                pass
            return False
        # write metadata into our newly-created claim dir
        lease_id = uuid.uuid4().hex
        try:
            self._write_claim_meta(claim_dir, lease_id)
        except Exception as e:
            # failed to write metadata -> back out of claim
            logger.debug(f"[{self._process_id}] _steal_stale_claim: write_claim_meta failed: {e}")
            try:
                shutil.rmtree(claim_dir)
            except Exception:
                pass
            try:
                shutil.rmtree(stale_name)
            except Exception:
                pass
            return False
        # finally, clean up the moved stale dir
        try:
            shutil.rmtree(stale_name)
        except Exception:
            pass
        self._owned_claims[claim_dir] = lease_id
        logger.debug(f"[{self._process_id}] _steal_stale_claim: SUCCESS - stole {os.path.basename(claim_dir)}, new lease_id={lease_id[:12]}...")
        return True

    def _verify_claim_ownership(self, claim_dir: str) -> bool:
        """Poll owner.json a few times to ensure our lease is visible (NFS may delay visibility)."""
        if not self.use_claims:
            return True
        for _ in range(5):
            time.sleep(random.uniform(0.05, 0.1))
            meta = self._read_claim_meta(claim_dir)
            lease_id = self._owned_claims.get(claim_dir)
            if lease_id and meta and meta.get("lease_id") == lease_id:
                return True
        logger.debug(f"[{self._process_id}] _verify_claim_ownership: failed to verify ownership of {os.path.basename(claim_dir)}")
        return False

    def release_shard(self, shard_path: str) -> None:
        """Remove our claim on shard_path if (and only if) our lease still matches on disk."""
        if not self.use_claims:
            return
        claim_dir = self._claim_dir(shard_path)
        lease_id = self._owned_claims.get(claim_dir)
        if lease_id is None:
            return

        meta = self._read_claim_meta(claim_dir)
        # Only delete if we still hold the authoritative lease
        if meta is not None and meta.get("lease_id") != lease_id:
            self._owned_claims.pop(claim_dir, None)
            return
        # still own it -> remove
        try:
            shutil.rmtree(claim_dir)
        except Exception:
            pass
        self._owned_claims.pop(claim_dir, None)

    def _release_all_owned_claims(self) -> None:
        if not self.use_claims:
            return
        # Only remove claims that still carry our lease_id
        for claim_dir, lease_id in list(self._owned_claims.items()):
            meta = self._read_claim_meta(claim_dir)
            if meta is not None and meta.get("lease_id") != lease_id:
                # Lost ownership; don't tear down someone else's claim
                self._owned_claims.pop(claim_dir, None)
                continue
            try:
                shutil.rmtree(claim_dir)
            except Exception:
                pass
            self._owned_claims.pop(claim_dir, None)

    # ---------- shard selection ----------
    def choose_shard(self) -> str:
        """Return a shard path. If use_claims is True, claim it atomically using a lease token."""
        # Reuse current shard if still valid
        if self._current_shard is not None:
            shard = os.path.realpath(self._current_shard)
            claim_dir = self._claim_dir(shard)
            # Non-claim mode: just check size
            if not self.use_claims:
                if not os.path.exists(shard) or os.path.getsize(shard) < self.max_bytes:
                    self._current_shard = shard
                    return shard
                else:
                    self._current_shard = None
            else:
                # Validate our lease is still the one on disk and shard isn't full
                lease_id = self._owned_claims.get(claim_dir)
                claim_exists = os.path.exists(claim_dir)
                meta = self._read_claim_meta(claim_dir) if claim_exists else None
                
                logger.debug(f"[{self._process_id}] choose_shard: revalidating {os.path.basename(shard)}, claim_exists={claim_exists}, in_memory_lease={lease_id[:12] if lease_id else None}..., disk_lease={meta.get('lease_id', 'N/A')[:12] if meta else None}...")
                
                if (lease_id is not None and meta is not None and meta.get("lease_id") == lease_id):
                    # we still own the lease
                    if (not os.path.exists(shard) or os.path.getsize(shard) < self.max_bytes):
                        # shard is not full, keep using it
                        self._current_shard = os.path.realpath(shard)
                        return self._current_shard
                    if os.path.getsize(shard) >= self.max_bytes:
                        # shard is full, release claim
                        logger.debug(f"[{self._process_id}] choose_shard: shard {os.path.basename(shard)} is full, releasing")
                        self.release_shard(shard)
                        self._current_shard = None
                    # if the shard is full, release our claim
                else:
                    # Lost the lease
                    logger.debug(f"[{self._process_id}] choose_shard: LOST LEASE on {os.path.basename(shard)}! in_memory={lease_id[:12] if lease_id else None}..., disk_meta={meta}")
                self._current_shard = None

        # No current shard: find a new one
        if not self.use_claims:
            # pick first non-existent or non-full shard
            i = 0
            while True:
                shard = os.path.realpath(os.path.join(self.shard_dir, f"{self.prefix}-{i:06d}.pack"))
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

        # claim-mode behaviour: collect candidates first to avoid thundering herd
        while True:
            # Each iteration of this outer loop is a full scan attempt
            candidates: List[Tuple[str, str, str]] = []  # (shard_path, claim_dir, action: 'steal' or 'claim')
            first_nonexistent_shard: Optional[str] = None
            
            i = 0
            while True:
                shard = os.path.realpath(os.path.join(self.shard_dir, f"{self.prefix}-{i:06d}.pack"))
                claim_dir = self._claim_dir(shard)
                shard_exists = os.path.exists(shard)
                claim_exists = os.path.exists(claim_dir)
                
                # Non-existent shard: end of scan
                if not shard_exists:
                    first_nonexistent_shard = shard
                    break
                
                shard_full = os.path.getsize(shard) >= self.max_bytes
                
                if claim_exists:
                    if self._is_claim_stale(claim_dir) and not shard_full:
                        # Stale claim on non-full shard: candidate to steal
                        candidates.append((shard, claim_dir, 'steal'))
                        logger.debug(f"[{self._process_id}] choose_shard: found stale claim candidate {os.path.basename(shard)}")
                    # else: active claim or full shard, skip
                else:
                    # No claim exists
                    if not shard_full:
                        # Non-full shard without claim: candidate to claim
                        candidates.append((shard, claim_dir, 'claim'))
                        logger.debug(f"[{self._process_id}] choose_shard: found unclaimed candidate {os.path.basename(shard)}")
                
                i += 1
            
            logger.debug(f"[{self._process_id}] choose_shard: scan complete. {len(candidates)} candidates, first_nonexistent={os.path.basename(first_nonexistent_shard) if first_nonexistent_shard else None}")
            
            # Decision logic
            if not candidates:
                # No candidates found, use the first non-existent shard
                shard = first_nonexistent_shard
                claim_dir = self._claim_dir(shard)
                if self._try_mkdir_claim(claim_dir):
                    if self._verify_claim_ownership(claim_dir):
                        fd = os.open(shard, os.O_CREAT | os.O_RDWR, 0o644)
                        os.close(fd)
                        self._current_shard = shard
                        return shard
                # Failed to claim new shard, restart scan
                time.sleep(random.uniform(0.05, 0.5))
                continue
            
            # Candidates exist: wait random delay then pick one randomly
            time.sleep(random.uniform(0.0, 1.0))
            
            # Shuffle and try candidates
            random.shuffle(candidates)
            for shard, claim_dir, action in candidates:
                # Re-check conditions as they may have changed during delay
                if not os.path.exists(shard):
                    # Shard was deleted, skip
                    continue
                if os.path.getsize(shard) >= self.max_bytes:
                    # Shard is now full, skip
                    continue
                
                success = False
                if action == 'steal':
                    # Re-verify claim is still stale
                    if os.path.exists(claim_dir) and self._is_claim_stale(claim_dir):
                        success = self._steal_stale_claim(claim_dir, shard)
                elif action == 'claim':
                    # Re-verify no claim exists
                    if not os.path.exists(claim_dir):
                        success = self._try_mkdir_claim(claim_dir)
                
                if success and self._verify_claim_ownership(claim_dir):
                    fd = os.open(shard, os.O_CREAT | os.O_RDWR, 0o644)
                    os.close(fd)
                    self._current_shard = shard
                    return shard
                
                # Failed this candidate, try next
                logger.debug(f"[{self._process_id}] choose_shard: failed to {action} {os.path.basename(shard)}, trying next candidate")
            
            # All candidates failed, restart scan with backoff
            logger.debug(f"[{self._process_id}] choose_shard: all candidates failed, restarting scan")
            time.sleep(random.uniform(0.1, 0.5))

    # ---------- append ----------
    def append(self, shard_path: str, payload: bytes, rec_id: Optional[bytes] = None) -> PackRecord:
        """Append a PACK record. With claims enabled, verify our lease matches on disk."""
        shard_path = os.path.realpath(shard_path)
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

        # Authoritative lease validation
        claim_dir = self._claim_dir(shard_path) if self.use_claims else None
        if self.use_claims:
            lease_id = self._owned_claims.get(claim_dir or "")
            claim_exists = claim_dir and os.path.exists(claim_dir)
            meta = self._read_claim_meta(claim_dir) if claim_exists else None
            
            logger.debug(f"[{self._process_id}] append: shard={os.path.basename(shard_path)} claim_exists={claim_exists} in_memory_lease={lease_id[:12] if lease_id else None}... disk_meta={meta}")
            
            if not (lease_id and meta and meta.get("lease_id") == lease_id):
                # Log warning with detailed diagnostics instead of raising RuntimeError
                logger.warning(
                    f"[{self._process_id}] LEASE MISMATCH - will retry with new shard. "
                    f"shard_path={shard_path}, claim_dir={claim_dir}, claim_exists={claim_exists}, "
                    f"in_memory_lease={lease_id}, disk_lease={meta.get('lease_id') if meta else None}"
                )
                logger.debug(f"[{self._process_id}] LEASE MISMATCH DETAILS:")
                logger.debug(f"  shard_path: {shard_path}")
                logger.debug(f"  claim_dir: {claim_dir}")
                logger.debug(f"  claim_dir_exists: {claim_exists}")
                logger.debug(f"  in_memory_lease_id: {lease_id}")
                logger.debug(f"  disk_meta: {meta}")
                logger.debug(f"  _owned_claims keys: {list(self._owned_claims.keys())}")
                logger.debug(f"  _current_shard: {self._current_shard}")
                # Try to get fresh stat info
                if claim_dir:
                    try:
                        stat_info = os.stat(claim_dir)
                        logger.debug(f"  claim_dir stat: mtime={stat_info.st_mtime} ({time.time() - stat_info.st_mtime:.1f}s ago)")
                    except Exception as e:
                        logger.debug(f"  claim_dir stat failed: {e}")
                
                # Clear state so caller can retry with a new shard
                self._current_shard = None
                if claim_dir:
                    self._owned_claims.pop(claim_dir, None)
                
                raise LeaseMismatchRetry(
                    f"Shard {shard_path} lease lost (stolen by another process). Retry with new shard."
                )

        fd = os.open(shard_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            off = os.lseek(fd, 0, os.SEEK_END)
            os.pwrite(fd, record, off)
            try:
                os.fsync(fd)   # ensure the record is pushed to server/disk
            except Exception:
                pass
            data_off = off + len(hdr) + id_len
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

        if self.use_claims:
            # heartbeat after successful write (keeps TTL fresh on dir mtime)
            try:
                os.utime(claim_dir, None)  # heartbeat
            except Exception:
                pass

        return {"id_hex": id_hex, "id_bytes": rec_id, "off": off, "data_off": data_off, "length": data_len, "shard_path": os.path.basename(shard_path)}

# --------------- Delta appenders ---------------
class DeltaAppender:
    """Buffer rows per schema and flush to a Delta table via deltalake."""
    def __init__(self, table_path: str, schema: pa.Schema, batch_size: int = 2500, retry_config: Optional[RetryConfig] = None):
        self.retry_config = retry_config if retry_config is not None else RetryConfig()
        self.table_uri = f"file://{os.path.abspath(table_path)}"
        self.schema = schema
        self.batch_size = batch_size
        self.buf: Dict[str, List[Any]] = {f.name: [] for f in schema}

        self.retry_config = retry_config

    def row_count(self) -> int:
        return len(self.buf[self.schema[0].name])

    def add_row(self, row: Dict[str, Any]) -> None:
        # Strictly adhere to schema fields; missing fields become None
        for f in self.schema:
            self.buf[f.name].append(row.get(f.name))
        if len(self.buf[self.schema[0].name]) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.buf[self.schema[0].name]:
            return
        
        # if not DeltaTable.is_deltatable(self.table_uri):
        #     raise FileNotFoundError(f"Delta table at {self.table_uri} does not exist. Always create before appending to make sure configuration settings are set correctly. (Concurrent writers may cause cause problems when changing configuration settings.)")
        
        tbl = pa.Table.from_pydict(self.buf, schema=self.schema)
        delay = self.retry_config.base_sleep
        for attempt in range(1, self.retry_config.max_retries + 1):
            try:
                # mode="append" will create table if missing
                write_deltalake(self.table_uri, tbl, mode="append", configuration={'delta.checkpointInterval': '500'})
                break  # success
            except DeltaError as e:
                if attempt < self.retry_config.max_retries and is_retryable_delta_error(e):
                    sleep_for = delay * (1 + self.retry_config.jitter * random.uniform(-1, 1))
                    logger.warning(f"write_deltalake failed with retryable error (attempt {attempt}/{self.retry_config.max_retries}). Retrying in {sleep_for:.2f}s... error: {e}")
                    time.sleep(sleep_for)
                    delay = min(self.retry_config.max_sleep, delay * 1.5)
                    continue
                # not retryable or out of retries
                logger.error(f"write_deltalake failed permanently after {attempt} attempts.")
                raise

        for k in self.buf:
            self.buf[k].clear()
