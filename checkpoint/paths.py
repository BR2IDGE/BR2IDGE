from pathlib import Path
import hashlib
import json
import os

DEFAULT_RUNS_DIR = Path(os.getenv("RUNS_DIR", "artifacts")).resolve()

def config_hash(cfg: dict) -> str:
    s = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:10]

def ensure_run_dir(dataset: str, model: str, cfg_hash_str: str, base_dir: Path | str = DEFAULT_RUNS_DIR) -> Path:
    base = Path(base_dir)
    run_dir = base / dataset / model / cfg_hash_str
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def f(run_dir: Path | str, *parts: str) -> Path:
    return Path(run_dir).joinpath(*parts)
