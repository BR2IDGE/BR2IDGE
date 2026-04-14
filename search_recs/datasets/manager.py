import json
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Optional

import requests


DEFAULT_DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1c8gLAue8Vp-b9KM93RQwh7hIAAonEgF5"

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


def _manifest_path() -> Path:
    return _repo_root() / "config_files" / "datasets" / "google_drive_datasets.json"


def _load_manifest() -> Dict:
    path = _manifest_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not read dataset Drive manifest at {path}: {exc}") from exc


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


def _extract_file_id(value: str) -> Optional[str]:
    if not value:
        return None
    patterns = [
        r"/file/d/([A-Za-z0-9_-]+)",
        r"[?&]id=([A-Za-z0-9_-]+)",
        r"uc\?export=download&id=([A-Za-z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", value.strip()):
        return value.strip()
    return None


def _confirm_token(response: requests.Response) -> Optional[str]:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    match = re.search(r"confirm=([0-9A-Za-z_]+)", response.text[:100_000])
    return match.group(1) if match else None


def _drive_warning_form(response: requests.Response) -> Optional[tuple[str, Dict[str, str]]]:
    text = response.text[:200_000]
    if "download-form" not in text:
        return None

    form = re.search(r'<form[^>]+id="download-form"[^>]+action="([^"]+)"[^>]*>(.*?)</form>', text, re.S)
    if not form:
        return None

    action = form.group(1).replace("&amp;", "&")
    body = form.group(2)
    params: Dict[str, str] = {}
    for name, value in re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', body):
        params[name] = value.replace("&amp;", "&")
    return action, params


def _looks_like_zip(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0 and zipfile.is_zipfile(path)
    except Exception:
        return False


def _download_drive_file(file_id: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    url = "https://drive.google.com/uc"
    params = {"export": "download", "id": file_id}
    response = session.get(url, params=params, stream=True, timeout=60)
    token = _confirm_token(response)
    if token:
        params["confirm"] = token
        response = session.get(url, params=params, stream=True, timeout=60)
    else:
        warning_form = _drive_warning_form(response)
        if warning_form:
            action, form_params = warning_form
            response.close()
            response = session.get(action, params=form_params, stream=True, timeout=60)
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
            f"Google Drive download for file id '{file_id}' did not produce a valid zip. "
            f"First bytes/text: {preview!r}"
        )


def _download_url(url: str, dest: Path) -> None:
    file_id = _extract_file_id(url)
    if file_id:
        _download_drive_file(file_id, dest)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".part")
    with tmp.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                fh.write(chunk)
    tmp.replace(dest)

    if dest.suffix.lower() == ".zip" and not _looks_like_zip(dest):
        preview = dest.read_text(errors="ignore")[:300] if dest.exists() else ""
        try:
            dest.unlink()
        except Exception:
            pass
        raise RuntimeError(f"Download URL did not produce a valid zip: {url}. First bytes/text: {preview!r}")


def _find_file_id_in_public_folder(folder_url: str, filename: str) -> Optional[str]:
    if not folder_url:
        return None

    response = requests.get(folder_url, timeout=60)
    response.raise_for_status()
    html = response.text
    ids_to_skip = set(re.findall(r"/folders/([A-Za-z0-9_-]+)", folder_url))

    for match in re.finditer(re.escape(filename), html):
        window = html[max(0, match.start() - 1200): match.end() + 1200]
        direct = re.search(r"/file/d/([A-Za-z0-9_-]+)", window)
        if direct and direct.group(1) not in ids_to_skip:
            return direct.group(1)

        ids = re.findall(r'["\\]([A-Za-z0-9_-]{20,})["\\]', window)
        for file_id in ids:
            if file_id not in ids_to_skip:
                return file_id

    return None


def _download_archive_from_manifest(key: str, archive_path: Path) -> None:
    manifest = _load_manifest()
    datasets = manifest.get("datasets", {})
    spec = datasets.get(key, {})
    archive_name = DATASETS[key]["archive"]

    env_prefix = re.sub(r"[^A-Z0-9]+", "_", key.upper()).strip("_")
    env_url = os.environ.get(f"BR2IDGE_{env_prefix}_ZIP_URL")
    env_id = os.environ.get(f"BR2IDGE_{env_prefix}_ZIP_ID")

    if env_id:
        print(f"[datasets] Downloading {archive_name} from Google Drive id in BR2IDGE_{env_prefix}_ZIP_ID...")
        _download_drive_file(env_id, archive_path)
        return

    url = env_url or spec.get("url") or spec.get("drive_url")
    file_id = spec.get("file_id") or _extract_file_id(str(url or ""))
    if file_id:
        print(f"[datasets] Downloading {archive_name} from configured Google Drive file id...")
        _download_drive_file(file_id, archive_path)
        return

    if url:
        print(f"[datasets] Downloading {archive_name} from configured URL...")
        _download_url(url, archive_path)
        return

    folder_url = (
        os.environ.get("BR2IDGE_DATASETS_DRIVE_FOLDER_URL")
        or manifest.get("folder_url")
        or DEFAULT_DRIVE_FOLDER_URL
    )
    print(f"[datasets] Looking for {archive_name} in Google Drive folder...")
    found_id = _find_file_id_in_public_folder(folder_url, archive_name)
    if not found_id:
        raise RuntimeError(
            f"Could not locate '{archive_name}' in the configured Google Drive folder. "
            f"Make the folder public, set file_id/url in {_manifest_path()}, or place the zip in data/_archives/."
        )
    _download_drive_file(found_id, archive_path)


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
        _download_archive_from_manifest(key, archive)

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
