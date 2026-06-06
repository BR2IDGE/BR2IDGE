import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Optional

import requests

ZENODO_RECORD_ID = os.environ.get("BR2IDGE_ZENODO_RECORD_ID", "20492270")
ZENODO_FILES_ARCHIVE_URL = (
    os.environ.get("BR2IDGE_DATASETS_ZENODO_ARCHIVE_URL")
    or f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files-archive"
)

DATASETS: Dict[str, Dict] = {
    "ml-25m": {
        "folder": "ml-25m",
        "archive": "ml-25m.zip",
        "required": [
            "ratings.csv",
            "movies.csv",
            "genome-scores.csv",
            "genome-tags.csv",
            "tags.csv",
            "links.csv",
        ],
        "aliases": ["movielens", "movielens25m", "ml25m", "ml-25m"],
    },
    "amazonElectronics": {
        "folder": "amazonElectronics",
        "archive": "amazonElectronics.zip",
        "required": ["ratings.csv", "meta_Electronics.json.gz"],
        "aliases": [
            "amazon",
            "amazonelectronics",
            "amazonElectronics",
            "amazonEletronics",
            "amazoneletronics",
        ],
    },
    "lastfm-dataset-360K": {
        "folder": "lastfm-dataset-360K",
        "archive": "lastfm-dataset-360K.zip",
        "required": [
            "usersha1-artmbid-artname-plays.tsv",
            "usersha1-profile.tsv",
        ],
        "aliases": ["lastfm360k", "lastfm-360k", "lastfm_360k", "lastfm-dataset-360K"],
    },
    "lastfm-hybrid": {
        "folder": "lastfm-hybrid",
        "archive": "lastfm-hybrid.zip",
        "required": ["artists.dat", "tags.dat", "user_taggedartists.dat", "user_artists.dat"],
        "aliases": ["lastfm-hybrid", "lastfm_hetrec", "lastfmhetrec", "hetrec-lastfm"],
    },
}


def _repo_root() -> Path:
    cur = Path(__file__).resolve()
    for parent in [cur, *cur.parents]:
        if (parent / "framework.py").exists():
            return parent
    return Path.cwd()


def _data_root() -> Path:
    return _repo_root() / "data"


def _norm_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _dataset_key(name: str) -> str:
    wanted = _norm_key(name)
    for key, spec in DATASETS.items():
        if wanted == _norm_key(key) or wanted == _norm_key(spec["folder"]):
            return key
        for alias in spec.get("aliases", []):
            if wanted == _norm_key(alias):
                return key
    raise KeyError(f"Unknown dataset '{name}'. Known datasets: {list(DATASETS)}")


def _required_exist(path: Path, required: Iterable[str]) -> bool:
    return path.is_dir() and all((path / rel).exists() for rel in required)


def get_dataset_path(name: str) -> Path:
    key = _dataset_key(name)
    return _data_root() / DATASETS[key]["folder"]


def _archive_candidates(archive_name: str) -> Iterable[Path]:
    root = _repo_root()
    data = _data_root()
    env_dir = os.environ.get("BR2IDGE_DATA_ARCHIVE_DIR")
    if env_dir:
        yield Path(env_dir) / archive_name
    yield data / "_archives" / archive_name
    yield data / archive_name
    yield root / archive_name


def _safe_extract_zip(zip_path: Path, target_root: Path) -> Path:
    tmp_dir = target_root / "_extracting" / zip_path.stem
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            dest = (tmp_dir / member.filename).resolve()
            if not str(dest).startswith(str(tmp_dir.resolve())):
                raise RuntimeError(f"Unsafe path in zip archive {zip_path}: {member.filename}")
        zf.extractall(tmp_dir)

    return tmp_dir


def _move_extracted_dataset(tmp_dir: Path, target_dir: Path, required: Iterable[str]) -> None:
    required = list(required)
    candidates = [tmp_dir]
    candidates.extend([p for p in tmp_dir.rglob("*") if p.is_dir()])

    source = None
    for candidate in candidates:
        if _required_exist(candidate, required):
            source = candidate
            break

    if source is None:
        raise FileNotFoundError(
            f"Extracted archive did not contain required files {required}. "
            f"Looked under {tmp_dir}."
        )

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if target_dir.exists():
        # The caller only reaches here when required files are missing. Keep any
        # existing partial files and copy the extracted files on top.
        for item in source.iterdir():
            dest = target_dir / item.name
            if dest.exists():
                continue
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
    else:
        shutil.copytree(source, target_dir)


def _looks_like_zip(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0 and zipfile.is_zipfile(path)
    except Exception:
        return False


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(
        url,
        stream=True,
        timeout=(30, 120),  # (connect, read) — read timeout is per-chunk, not total
        headers={"User-Agent": "BR2IDGE-datasets/1.0"},
    )
    response.raise_for_status()

    tmp = dest.with_suffix(dest.suffix + ".part")
    with tmp.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                fh.write(chunk)
    tmp.replace(dest)

    if not _looks_like_zip(dest):
        preview = dest.read_text(errors="ignore")[:300] if dest.exists() else ""
        try:
            dest.unlink()
        except Exception:
            pass
        raise RuntimeError(
            f"Download did not produce a valid zip: {url}. First bytes/text: {preview!r}"
        )


def _download_zenodo_bundle_and_extract_archives(archives_dir: Path) -> None:
    archives_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = archives_dir / "_zenodo_bundle.zip"

    print(f"[datasets] Downloading Zenodo bundle (zip-of-zips) from {ZENODO_FILES_ARCHIVE_URL} ...")
    print("[datasets] Note: this is a single large download (~2 GB) containing all datasets.")
    _download_file(ZENODO_FILES_ARCHIVE_URL, bundle_path)

    extracted = []
    try:
        with zipfile.ZipFile(bundle_path, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                inner_name = Path(member.filename).name
                if not inner_name.lower().endswith(".zip"):
                    continue
                dest = archives_dir / inner_name
                with zf.open(member) as src, dest.open("wb") as out:
                    shutil.copyfileobj(src, out, length=1024 * 1024)
                extracted.append(inner_name)
                print(f"[datasets] Extracted inner archive: {inner_name}")
    finally:
        try:
            bundle_path.unlink()
        except Exception:
            pass

    if not extracted:
        raise RuntimeError(
            f"Zenodo bundle did not contain any .zip files. URL: {ZENODO_FILES_ARCHIVE_URL}"
        )


def _ensure_archive_from_zenodo(archive_path: Path) -> None:
    """
    Make sure `archive_path` (e.g. data/_archives/ml-25m.zip) exists, fetching it
    from Zenodo if needed. Downloading the bundle populates every per-dataset
    archive at once, so this is a no-op after the first successful call.
    """
    if _looks_like_zip(archive_path):
        return

    _download_zenodo_bundle_and_extract_archives(archive_path.parent)

    if not _looks_like_zip(archive_path):
        raise RuntimeError(
            f"After downloading the Zenodo bundle, the expected archive "
            f"'{archive_path.name}' was not found in {archive_path.parent}. "
            f"Check that the Zenodo record contains a file named '{archive_path.name}'."
        )


def ensure_dataset(name: str) -> Path:
    key = _dataset_key(name)
    spec = DATASETS[key]
    target_dir = _data_root() / spec["folder"]
    required = spec["required"]

    if _required_exist(target_dir, required):
        return target_dir

    archive_name = spec["archive"]
    archive = None
    for candidate in _archive_candidates(archive_name):
        if not candidate.exists():
            continue
        if _looks_like_zip(candidate):
            archive = candidate
            break
        print(f"[datasets] Ignoring invalid zip archive: {candidate}")
        try:
            candidate.unlink()
        except Exception:
            pass
    if archive is None:
        archive = _data_root() / "_archives" / archive_name
        _ensure_archive_from_zenodo(archive)

    print(f"[datasets] Extracting {archive.name} -> {target_dir}")
    tmp_dir = _safe_extract_zip(archive, _data_root())
    try:
        _move_extracted_dataset(tmp_dir, target_dir, required)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    if not _required_exist(target_dir, required):
        missing = [rel for rel in required if not (target_dir / rel).exists()]
        raise FileNotFoundError(f"Dataset '{key}' is still missing files after extraction: {missing}")

    return target_dir
