"""Cheap availability probes for AbstractMusic providers.

Discovery must stay import-light. Listing providers and models must never import
a model runtime (``torch`` / ``transformers`` / ``diffusers``) and must never
load weights, so the answer is derived from two dependency-free probes:

* **Local** — a model is available when its weights are already in the Hugging
  Face cache. That is a handful of ``stat`` calls, no imports.
* **Remote** — a provider is available when its API answers. Every remote
  provider is probed concurrently under a single short deadline.

This module is stdlib-only and holds no music-specific knowledge: callers pass
model ids and endpoints, and get back plain answers.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple

__all__ = [
    "REMOTE_PROBE_TIMEOUT_S",
    "RemoteEndpoint",
    "RemoteProbe",
    "clear_probe_cache",
    "hf_cache_root",
    "is_model_cached",
    "cached_model_ids",
    "probe_endpoints",
]

# Remote providers get one short deadline. Discovery is interactive; a provider
# that cannot answer within this budget is reported as unreachable, not waited on.
REMOTE_PROBE_TIMEOUT_S = 5.0

# Probe results are reused for this long so that a discovery call that asks for
# providers, models and the full catalog costs one round of requests, not three.
PROBE_CACHE_TTL_S = 30.0

_WEIGHT_SUFFIXES = frozenset(
    {".safetensors", ".bin", ".ckpt", ".pt", ".pth", ".gguf", ".onnx", ".msgpack"}
)

# Directories inside a snapshot that never hold weights.
_SKIPPED_SNAPSHOT_DIRS = frozenset({".cache", ".git"})

# --------------------------------------------------------------------------- #
# Local model presence
# --------------------------------------------------------------------------- #


def _env_path(key: str) -> Optional[Path]:
    """Read a path from the environment the way ``huggingface_hub`` does.

    One deliberate divergence: a blank value is treated as unset rather than as
    the relative path ``huggingface_hub`` would derive from it.
    """

    value = os.environ.get(key)
    if value is None or not value.strip():
        return None
    return Path(os.path.expandvars(os.path.expanduser(value.strip())))


def hf_cache_root() -> Path:
    """Return the Hugging Face cache directory this process would download into.

    This mirrors ``huggingface_hub``'s own precedence so that discovery looks
    exactly where the loader will later look. Replicating it (rather than
    importing ``huggingface_hub``) keeps the base package dependency-free.
    """

    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        root = _env_path(key)
        if root is not None:
            return root
    hf_home = _env_path("HF_HOME")
    if hf_home is None:
        xdg_cache = _env_path("XDG_CACHE_HOME") or (Path.home() / ".cache")
        hf_home = xdg_cache / "huggingface"
    return hf_home / "hub"


def _repo_dir_name(model_id: str) -> str:
    return "models--" + str(model_id).strip().replace("/", "--")


def _scandir(path: Path) -> List[os.DirEntry]:
    """List a directory, treating any failure as empty. Never raises."""

    try:
        with os.scandir(path) as entries:
            return list(entries)
    except (OSError, ValueError):  # missing, unreadable, or an unrepresentable name
        return []


def _walk_snapshot(snapshot_dir: Path):
    """Yield every file entry in a snapshot, skipping bookkeeping directories."""

    pending = [snapshot_dir]
    while pending:
        for entry in _scandir(pending.pop()):
            # Never follow directory symlinks: real snapshot subdirectories are
            # plain directories, and refusing to follow rules out link cycles.
            if entry.is_dir(follow_symlinks=False):
                if entry.name not in _SKIPPED_SNAPSHOT_DIRS:
                    pending.append(Path(entry.path))
            else:
                yield entry


def _shards_are_complete(index_entry: os.DirEntry) -> bool:
    """Return True when every shard an index file references is materialized.

    A sharded checkpoint whose first shard landed but whose second did not would
    otherwise read as present, and loading it would silently start a multi-GB
    download — the exact stall this module exists to prevent.
    """

    try:
        with open(index_entry.path, "r", encoding="utf-8") as handle:
            index = json.load(handle)
    except (OSError, ValueError):
        return True  # unreadable index: fall back to the plain presence check
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict):
        return True
    shard_dir = Path(index_entry.path).parent
    return all((shard_dir / str(name)).is_file() for name in set(weight_map.values()))


def _snapshot_has_weights(snapshot_dir: Path) -> bool:
    """Return True when this snapshot holds a complete set of weights.

    Snapshot entries are symlinks into ``blobs/``; ``is_file()`` follows them, so
    a link left dangling by an interrupted download does not count as present.
    """

    found_weights = False
    for entry in _walk_snapshot(snapshot_dir):
        name = entry.name
        if name.endswith(".index.json"):
            if not _shards_are_complete(entry):
                return False
            continue
        if not found_weights and os.path.splitext(name)[1].lower() in _WEIGHT_SUFFIXES:
            try:
                found_weights = entry.is_file()  # follows the symlink into blobs/
            except OSError:
                pass
    return found_weights


def _snapshots_to_check(repo_dir: Path) -> List[Path]:
    """Return the snapshots that decide presence, preferring the checked-out ref.

    Loaders resolve a revision through ``refs/``; when a ref exists, an unrelated
    snapshot left over from an older revision must not vouch for it.
    """

    referenced: List[Path] = []
    for ref in _scandir(repo_dir / "refs"):
        try:
            commit = Path(ref.path).read_text(encoding="utf-8").strip()
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        snapshot = repo_dir / "snapshots" / commit
        if commit and snapshot.is_dir():
            referenced.append(snapshot)
    if referenced:
        return referenced
    return [Path(entry.path) for entry in _scandir(repo_dir / "snapshots") if entry.is_dir(follow_symlinks=False)]


def is_model_cached(model_id: str, *, cache_root: Optional[Path] = None) -> bool:
    """Return True when a Hugging Face repo has usable weights on this machine.

    In-flight downloads need no special handling: ``huggingface_hub`` only links
    a file into a snapshot once its blob is complete, and it keeps ``.incomplete``
    blobs around deliberately so interrupted downloads can resume.
    """

    repo_id = str(model_id or "").strip()
    if "/" not in repo_id:
        return False
    repo_dir = (cache_root if cache_root is not None else hf_cache_root()) / _repo_dir_name(repo_id)
    return any(_snapshot_has_weights(snapshot) for snapshot in _snapshots_to_check(repo_dir))


def cached_model_ids(model_ids: Iterable[str]) -> FrozenSet[str]:
    """Return the subset of ``model_ids`` whose weights are already on this machine."""

    root = hf_cache_root()
    if not root.is_dir():
        return frozenset()
    return frozenset(model_id for model_id in model_ids if is_model_cached(model_id, cache_root=root))


# --------------------------------------------------------------------------- #
# Remote endpoint probes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RemoteEndpoint:
    """A cheap read-only URL that proves a remote provider is answering."""

    url: str
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RemoteProbe:
    """Outcome of a single remote endpoint probe."""

    status: str  # available | unauthorized | unavailable | unreachable
    detail: str = ""
    latency_ms: Optional[int] = None

    @property
    def usable(self) -> bool:
        return self.status == "available"


def _classify(code: int) -> Tuple[str, str]:
    if code < 400:
        return "available", ""
    if code in (401, 403):
        return "unauthorized", f"HTTP {code}: credentials rejected"
    if code >= 500:
        return "unavailable", f"HTTP {code}: provider error"
    # Any other 4xx means the server answered and accepted our credentials; the
    # endpoint shape is not our business here.
    return "available", f"HTTP {code}"


def probe_endpoint(endpoint: RemoteEndpoint, *, timeout_s: float = REMOTE_PROBE_TIMEOUT_S) -> RemoteProbe:
    """Probe one endpoint. Never raises; failures come back as a status."""

    headers = {
        "Accept": "application/json",
        "User-Agent": "abstractmusic/availability",
        **{str(k): str(v) for k, v in dict(endpoint.headers).items()},
    }
    request = urllib.request.Request(str(endpoint.url), headers=headers, method="GET")
    started = time.monotonic()

    def elapsed_ms() -> int:
        return int((time.monotonic() - started) * 1000)

    try:
        # nosec B310 - the URL is the user's own configured provider endpoint.
        with urllib.request.urlopen(request, timeout=float(timeout_s)) as response:
            status, detail = _classify(int(getattr(response, "status", 200) or 200))
    except urllib.error.HTTPError as exc:
        # An error status is still an answer: the provider is up and talking.
        status, detail = _classify(int(exc.code))
    except (urllib.error.URLError, OSError, http.client.HTTPException, ValueError) as exc:
        # Timeouts, DNS and TLS failures, proxy errors, malformed URLs. Anything
        # outside this set is a bug in our own code and must not be reported as
        # a provider being down.
        reason = getattr(exc, "reason", None) or exc
        return RemoteProbe(status="unreachable", detail=str(reason), latency_ms=elapsed_ms())
    return RemoteProbe(status=status, detail=detail, latency_ms=elapsed_ms())


def probe_endpoints(
    endpoints: Mapping[str, RemoteEndpoint],
    *,
    timeout_s: float = REMOTE_PROBE_TIMEOUT_S,
    use_cache: bool = True,
) -> Dict[str, RemoteProbe]:
    """Probe every endpoint concurrently and return results keyed as given.

    Total wall time is bounded by ``timeout_s`` even when a probe thread is stuck
    in a name lookup, which no socket timeout can interrupt.
    """

    targets = dict(endpoints)
    if not targets:
        return {}

    timeout_s = float(timeout_s)
    results: Dict[str, RemoteProbe] = {}
    pending: Dict[str, RemoteEndpoint] = {}
    for key, endpoint in targets.items():
        if not use_cache:
            pending[key] = endpoint
            continue
        cached = _cached_probe(endpoint)
        if cached is not None:
            results[key] = cached
        elif _claim_inflight(endpoint):
            pending[key] = endpoint
        else:
            # An earlier round already has a probe out for this endpoint that
            # overran its deadline. Do not start a second one, and do not wait.
            results[key] = RemoteProbe(status="unreachable", detail="probe still in progress")

    if pending:
        lock = threading.Lock()
        fresh: Dict[str, RemoteProbe] = {}

        def run(key: str, endpoint: RemoteEndpoint) -> None:
            probe = None
            try:
                probe = probe_endpoint(endpoint, timeout_s=timeout_s)
            finally:
                if use_cache:
                    # The worker owns the outcome even when it lands after the
                    # deadline, so a slow provider self-heals on the next call.
                    if probe is not None:
                        _store_probe(endpoint, probe)
                    _release_inflight(endpoint)
                if probe is not None:
                    with lock:
                        fresh[key] = probe

        threads = [
            threading.Thread(target=run, args=(key, endpoint), name=f"abstractmusic-probe-{key}", daemon=True)
            for key, endpoint in pending.items()
        ]
        deadline = time.monotonic() + timeout_s
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))

        with lock:
            for key in pending:
                results[key] = fresh.get(key) or RemoteProbe(
                    status="unreachable",
                    detail=f"no response within {timeout_s:g}s",
                    latency_ms=int(timeout_s * 1000),
                )

    return results


# --------------------------------------------------------------------------- #
# Probe cache
# --------------------------------------------------------------------------- #

_cache_lock = threading.Lock()
_probe_cache: Dict[str, Tuple[float, RemoteProbe]] = {}
_inflight: set = set()


def _cache_key(endpoint: RemoteEndpoint) -> str:
    """Identify an endpoint without retaining its credentials in module state."""

    material = repr((str(endpoint.url), sorted((str(k), str(v)) for k, v in dict(endpoint.headers).items())))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _claim_inflight(endpoint: RemoteEndpoint) -> bool:
    """Reserve the right to probe this endpoint. False when one is already out."""

    key = _cache_key(endpoint)
    with _cache_lock:
        if key in _inflight:
            return False
        _inflight.add(key)
        return True


def _release_inflight(endpoint: RemoteEndpoint) -> None:
    with _cache_lock:
        _inflight.discard(_cache_key(endpoint))


def _cached_probe(endpoint: RemoteEndpoint) -> Optional[RemoteProbe]:
    key = _cache_key(endpoint)
    now = time.monotonic()
    with _cache_lock:
        entry = _probe_cache.get(key)
        if entry is None:
            return None
        expires_at, probe = entry
        if expires_at <= now:
            _probe_cache.pop(key, None)
            return None
        return probe


def _store_probe(endpoint: RemoteEndpoint, probe: RemoteProbe) -> None:
    key = _cache_key(endpoint)
    with _cache_lock:
        _probe_cache[key] = (time.monotonic() + PROBE_CACHE_TTL_S, probe)


def clear_probe_cache() -> None:
    """Drop cached remote probe results (tests, or after changing credentials)."""

    with _cache_lock:
        _probe_cache.clear()
        _inflight.clear()
