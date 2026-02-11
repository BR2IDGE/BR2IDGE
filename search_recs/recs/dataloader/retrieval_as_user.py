# Retrieval-as-User DataLoader for hybrid (search↔recs) experiments.

import json
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from search_recs.recs.dataloader import RecsDataLoader
except Exception:
    RecsDataLoader = object

try:
    from search_recs.search.model import BM25Model as SearchBM25Model
except Exception:
    SearchBM25Model = None


def _find_repo_root(start: Path) -> Path:
    cur = start
    for _ in range(12):
        if (cur / "framework.py").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.parent


def _resolve_dataset_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        out = p
    else:
        repo = _find_repo_root(Path(__file__).resolve())
        parts = [x for x in p.parts if x not in (".", "")]
        if parts and parts[0] == "data":
            out = repo / Path(*parts)
        elif str(path_str).startswith("./data") or str(path_str).startswith("data/"):
            out = repo / p
        else:
            out = repo / "data" / p
    out.mkdir(parents=True, exist_ok=True)
    return out


def _import_class(class_path: str):
    mod_name, cls_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def _ensure_nltk():
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt_tab/english")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass


def _read_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in [".tsv", ".dat"]:
        return pd.read_csv(path, sep="\t")
    if suf == ".jsonl":
        return pd.read_json(path, lines=True)
    if suf == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return pd.json_normalize(obj)

    return pd.read_csv(path)


def _infer_interactions_file(dataset_path: Path) -> Path:
    candidates = [
        dataset_path / "interactions.csv",
        dataset_path / "interactions.parquet",
        dataset_path / "ratings.csv",
        dataset_path / "ratings.parquet",
        dataset_path / "user_item.csv",
        dataset_path / "user_item.parquet",
    ]
    for fp in candidates:
        if fp.exists():
            return fp
    raise FileNotFoundError(
        f"Could not infer interactions file in {dataset_path}. "
        f"Expected one of: {[c.name for c in candidates]}"
    )


def _infer_items_file(dataset_path: Path) -> Path:
    candidates = [
        dataset_path / "items.csv",
        dataset_path / "items.parquet",
        dataset_path / "products.csv",
        dataset_path / "products.parquet",
        dataset_path / "metadata.csv",
        dataset_path / "metadata.parquet",
    ]
    for fp in candidates:
        if fp.exists():
            return fp
    raise FileNotFoundError(
        f"Could not infer items file in {dataset_path}. "
        f"Expected one of: {[c.name for c in candidates]}"
    )


def _build_item_text(
    df_items: pd.DataFrame,
    item_col: str,
    text_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    if item_col not in df_items.columns:
        raise ValueError(f"items: item_col '{item_col}' not found. Columns={list(df_items.columns)}")

    if text_cols is None:
        text_cols = [c for c in df_items.columns if c != item_col]

    for c in text_cols:
        if c not in df_items.columns:
            raise ValueError(f"items: text_col '{c}' not found. Columns={list(df_items.columns)}")

    doc_text = []
    for _, row in df_items.iterrows():
        parts = []
        for c in text_cols:
            v = row.get(c, "")
            if pd.isna(v):
                continue
            parts.append(str(v))
        doc_text.append(" ".join(parts).strip())

    out = df_items[[item_col]].copy()
    out["doc_text"] = doc_text
    return out


def _materialize_from_base_loader(
    dataset_path: Path,
    base_loader: Dict[str, Any],
    user_col: str,
    item_col: str,
    time_col: Optional[str],
    rating_col: Optional[str],
    item_text_cols: Optional[List[str]],
) -> Tuple[Path, Path]:
    class_path = base_loader.get("class_path")
    base_cfg = base_loader.get("config", {}) or {}
    if not class_path:
        raise ValueError("base_loader.class_path was not provided.")

    LoaderCls = _import_class(class_path)
    loader = LoaderCls(base_cfg)
    df = loader.load_data()

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f"base_loader returned an empty DataFrame. class_path={class_path}")

    ucol = "user" if "user" in df.columns else user_col
    icol = "item" if "item" in df.columns else item_col

    tcol = None
    if time_col and time_col in df.columns:
        tcol = time_col
    elif "time" in df.columns:
        tcol = "time"

    rcol = None
    if rating_col and rating_col in df.columns:
        rcol = rating_col
    elif "label" in df.columns:
        rcol = "label"

    for c in [ucol, icol]:
        if c not in df.columns:
            raise ValueError(f"base_loader DF is missing column '{c}'. Columns={list(df.columns)}")

    inter = pd.DataFrame({user_col: df[ucol], item_col: df[icol]})
    inter[time_col or "time"] = df[tcol] if tcol else 0
    inter[rating_col or "label"] = df[rcol] if rcol else 1

    inter_fp = dataset_path / "interactions.parquet"
    inter.to_parquet(inter_fp, index=False)

    items = pd.DataFrame({item_col: df[icol]})
    items = items.drop_duplicates(subset=[item_col]).copy()

    chosen_text_cols: List[str] = []
    if item_text_cols:
        chosen_text_cols = [c for c in item_text_cols if c in df.columns]
    elif "artist_name" in df.columns:
        chosen_text_cols = ["artist_name"]

    if not chosen_text_cols:
        items["item_text"] = items[item_col].astype(str)
        chosen_text_cols = ["item_text"]
    else:
        for c in chosen_text_cols:
            tmp = df[[icol, c]].drop_duplicates(subset=[icol]).copy()
            tmp = tmp.rename(columns={icol: item_col})
            items = items.merge(tmp, on=item_col, how="left")

    items_fp = dataset_path / "items.parquet"
    items.to_parquet(items_fp, index=False)

    return inter_fp, items_fp


def _normalize_retriv_output(out: Any, top_k: int) -> List[Tuple[Any, float]]:
    if out is None:
        return []

    if isinstance(out, dict) and out:
        try:
            if all(isinstance(v, (int, float, np.floating)) for v in out.values()):
                pairs = sorted(out.items(), key=lambda kv: float(kv[1]), reverse=True)
                return [(k, float(v)) for k, v in pairs[:top_k]]
        except Exception:
            pass

    if isinstance(out, list) and out:
        first = out[0]
        if isinstance(first, tuple) and len(first) >= 2:
            return [(p[0], float(p[1])) for p in out[:top_k]]
        if isinstance(first, dict):
            pairs = []
            for d in out[:top_k]:
                doc_id = d.get("doc_id", d.get("id", d.get("document_id", d.get("docid"))))
                score = d.get("score", d.get("bm25", d.get("similarity", 0.0)))
                if doc_id is not None:
                    pairs.append((doc_id, float(score)))
            return pairs

    if isinstance(out, tuple) and len(out) == 2:
        ids, scores = out
        ids = list(ids)[:top_k]
        scores = list(scores)[:top_k]
        return [(ids[i], float(scores[i])) for i in range(min(len(ids), len(scores)))]

    if isinstance(out, dict):
        ids = out.get("ids", out.get("doc_ids", out.get("documents", out.get("docid"))))
        scores = out.get("scores", out.get("score", out.get("bm25")))
        if ids is not None and scores is not None:
            ids = list(ids)[:top_k]
            scores = list(scores)[:top_k]
            return [(ids[i], float(scores[i])) for i in range(min(len(ids), len(scores)))]

    return []


def _bm25_search_topk(bm25: Any, query: str, top_k: int) -> List[Tuple[Any, float]]:
    q = str(query)

    for meth_name in ("search", "retrieve", "query", "rank"):
        if hasattr(bm25, meth_name):
            meth = getattr(bm25, meth_name)
            if callable(meth):
                try:
                    return _normalize_retriv_output(meth(q, top_k=top_k), top_k)
                except TypeError:
                    return _normalize_retriv_output(meth(q, top_k), top_k)
                except Exception:
                    pass

    for attr in ("_sr", "sr", "retriever", "_retriever", "model", "_model"):
        if hasattr(bm25, attr):
            sr = getattr(bm25, attr)
            if sr is None:
                continue
            if hasattr(sr, "search"):
                try:
                    return _normalize_retriv_output(sr.search(q, cutoff=int(top_k)), top_k)
                except TypeError:
                    return _normalize_retriv_output(sr.search(q, int(top_k)), top_k)
                except Exception:
                    pass

    raise RuntimeError("[Retrieval-as-User] BM25 search is not available.")


def _bm25_cfg_for_indexing(bm25_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(bm25_cfg or {})
    cfg["task"] = "normal"
    return cfg


def _make_bm25_model(bm25_config: Dict[str, Any], dataset_path: Optional[Path] = None):
    if SearchBM25Model is None:
        raise ImportError(
            "Could not import SearchBM25Model. "
            "Check whether 'search_recs.search.model' exports BM25Model correctly."
        )

    cfg = dict(bm25_config or {})
    cfg.setdefault("task", "normal")
    if dataset_path is not None:
        cfg.setdefault("dataset_path", str(dataset_path))

    model_name = cfg.get("model", "bm25")
    return SearchBM25Model({"model": model_name, "parameters": cfg})


def _load_movielens_search_dataset(
    dataset_path: Path,
    min_relevance: float,
    query_col: str,
    doc_col: str,
    doc_id_col: str,
    max_tags_per_doc: int = 30,
) -> pd.DataFrame:
    base = Path(dataset_path)
    tags = pd.read_csv(base / "genome-tags.csv")
    scores = pd.read_csv(base / "genome-scores.csv")
    movies = pd.read_csv(base / "movies.csv")

    df = scores.merge(tags[["tagId", "tag"]], on="tagId", how="left")
    df["relevance"] = pd.to_numeric(df["relevance"], errors="coerce").fillna(0.0)

    df_rel = df[df["relevance"] >= float(min_relevance)].copy() if float(min_relevance) > 0 else df.copy()

    tag_agg = (
        df_rel.sort_values(["movieId", "relevance"], ascending=[True, False])
        .groupby("movieId")["tag"]
        .agg(list)
        .reset_index()
    )

    movies = movies.copy()
    movies[doc_id_col] = movies["movieId"].astype(int)

    base_doc = (
        movies["title"].astype(str).fillna("").str.strip()
        + " \n "
        + movies["genres"].astype(str).fillna("").str.replace("|", " ", regex=False).str.strip()
    )

    movies = movies.merge(tag_agg, on="movieId", how="left")

    def tags_to_text(xs) -> str:
        if not isinstance(xs, list) or not xs:
            return ""
        xs = [str(t).strip() for t in xs if t and str(t).lower() != "nan"][: int(max_tags_per_doc)]
        return " \n " + " ".join(xs) if xs else ""

    movies[doc_col] = base_doc + movies["tag"].apply(tags_to_text)

    out = df_rel.merge(movies[["movieId", doc_id_col, doc_col]], on="movieId", how="left")
    out = out.rename(columns={"tag": query_col})
    out = out[[query_col, doc_col, doc_id_col]].dropna()
    out = out[out[doc_col].astype(str).str.strip() != ""].copy()
    return out


def _download_with_progress(url: str, dest_path: Path, min_bytes_ok: int = 100 * 1024 * 1024) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size >= int(min_bytes_ok):
        print(f"[RetrievalAsUser][Amazon] File found: {dest_path.name} ({dest_path.stat().st_size} bytes)")
        return

    print(f"[RetrievalAsUser][Amazon] Downloading: {url}")
    tmp_part = dest_path.with_suffix(dest_path.suffix + ".part")

    try:
        import urllib.request
        import shutil

        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=300) as resp, open(tmp_part, "wb") as f:
            shutil.copyfileobj(resp, f)

        tmp_part.replace(dest_path)
        print(f"[RetrievalAsUser][Amazon] Download completed: {dest_path}")
    except Exception as e:
        try:
            if tmp_part.exists():
                tmp_part.unlink()
        except Exception:
            pass
        raise RuntimeError(f"[RetrievalAsUser][Amazon] Failed to download meta: {e}")


def _amazon_clean_description(desc_raw: Any) -> str:
    if isinstance(desc_raw, list):
        return " ".join([str(d).strip() for d in desc_raw if d]).strip()
    if isinstance(desc_raw, str):
        return desc_raw.strip()
    return ""


def _amazon_extract_categories(cat_raw: Any, min_len: int = 3) -> List[str]:
    if not isinstance(cat_raw, list):
        return []

    unique_cats = set()
    for chain in cat_raw:
        if isinstance(chain, list):
            for c in chain:
                if not c:
                    continue
                s = str(c).strip()
                if len(s) < int(min_len):
                    continue
                if s.lower() in {"electronics"}:
                    continue
                unique_cats.add(s)
        else:
            s = str(chain).strip()
            if len(s) >= int(min_len) and s.lower() not in {"electronics"}:
                unique_cats.add(s)

    return list(unique_cats)


def _load_amazon_category_search_dataset(
    dataset_path: Path,
    meta_url: str,
    meta_file: str,
    cache_pairs: str,
    max_products: int,
    min_cat_len: int,
    min_docs_per_query: int,
    query_col: str,
    doc_col: str,
    doc_id_col: str,
) -> pd.DataFrame:
    cache_fp = dataset_path / str(cache_pairs)
    if cache_fp.exists():
        try:
            df = pd.read_parquet(cache_fp)
            need_cols = {query_col, doc_col, doc_id_col}
            if need_cols.issubset(df.columns) and len(df) > 0:
                print(f"[RetrievalAsUser][Amazon] Cache loaded: {cache_fp} | rows={len(df)}")
                return df[[query_col, doc_col, doc_id_col]].copy()
        except Exception as e:
            print(f"[RetrievalAsUser][Amazon] Warning: failed to read cache {cache_fp}: {e} (rebuilding...)")

    meta_fp = dataset_path / str(meta_file)
    _download_with_progress(meta_url, meta_fp)

    import gzip

    print("[RetrievalAsUser][Amazon] Streaming meta and generating category->product pairs...")
    rows: List[Dict[str, Any]] = []
    seen_pairs = set()

    n = 0
    kept = 0

    with gzip.open(meta_fp, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            n += 1
            if max_products and n > int(max_products):
                break

            try:
                obj = json.loads(line)
            except Exception:
                continue

            asin = obj.get("asin")
            title = str(obj.get("title", "") or "").strip()
            if not asin or not title:
                continue

            desc = _amazon_clean_description(obj.get("description", obj.get("desc", "")))

            cat_raw = obj.get("category", None)
            if cat_raw is None:
                cat_raw = obj.get("categories", None)

            cats = _amazon_extract_categories(cat_raw, min_len=int(min_cat_len))
            if not cats:
                continue

            doc_text = f"{title}. {desc}" if desc else title

            for c in cats:
                key = (str(c), str(asin))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                rows.append({query_col: c, doc_col: doc_text, doc_id_col: str(asin)})
                kept += 1

            if (n % 50_000) == 0:
                print(f"[RetrievalAsUser][Amazon] read={n} | pairs={kept}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.drop_duplicates(subset=[query_col, doc_id_col]).copy()

    if int(min_docs_per_query) > 1:
        vc = df[query_col].value_counts()
        valid = vc[vc >= int(min_docs_per_query)].index
        before_q = len(vc)
        df = df[df[query_col].isin(valid)].copy()
        after_q = len(valid)
        print(
            f"[RetrievalAsUser][Amazon] Queries before={before_q} | after(min_docs={min_docs_per_query})={after_q} | pairs={len(df)}"
        )

    try:
        df.to_parquet(cache_fp, index=False)
        print(f"[RetrievalAsUser][Amazon] Cache saved: {cache_fp}")
    except Exception as e:
        print(f"[RetrievalAsUser][Amazon] Warning: could not save cache {cache_fp}: {e}")

    return df[[query_col, doc_col, doc_id_col]].copy()


def _load_profile_query_inputs(
    dataset_path: Path,
    base_loader: Optional[Dict[str, Any]],
    interactions_file: Optional[str],
    items_file: Optional[str],
    user_col: str,
    item_col: str,
    time_col: Optional[str],
    rating_col: Optional[str],
    min_rating: float,
    item_text_cols: Optional[List[str]],
    max_docs: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    inter_fp: Optional[Path] = None
    items_fp: Optional[Path] = None

    if interactions_file:
        fp = dataset_path / interactions_file
        if fp.exists():
            inter_fp = fp
    if items_file:
        fp = dataset_path / items_file
        if fp.exists():
            items_fp = fp

    if inter_fp is None:
        try:
            inter_fp = _infer_interactions_file(dataset_path)
        except FileNotFoundError:
            inter_fp = None

    if items_fp is None:
        try:
            items_fp = _infer_items_file(dataset_path)
        except FileNotFoundError:
            items_fp = None

    if (inter_fp is None or items_fp is None) and base_loader:
        _materialize_from_base_loader(
            dataset_path=dataset_path,
            base_loader=base_loader,
            user_col=user_col,
            item_col=item_col,
            time_col=time_col,
            rating_col=rating_col,
            item_text_cols=item_text_cols,
        )
        if inter_fp is None:
            inter_fp = _infer_interactions_file(dataset_path)
        if items_fp is None:
            items_fp = _infer_items_file(dataset_path)

    if inter_fp is None:
        raise FileNotFoundError(
            f"Could not locate interactions in {dataset_path}. "
            f"Provide interactions_file or a base_loader."
        )
    if items_fp is None:
        raise FileNotFoundError(
            f"Could not locate items in {dataset_path}. "
            f"Provide items_file or a base_loader."
        )

    df_inter = _read_any(inter_fp)
    df_items = _read_any(items_fp)

    for c in [user_col, item_col]:
        if c not in df_inter.columns:
            raise ValueError(f"interactions: missing '{c}'. Columns={list(df_inter.columns)}")

    if rating_col and rating_col in df_inter.columns:
        df_inter = df_inter[df_inter[rating_col].astype(float) >= float(min_rating)].copy()

    canon = pd.DataFrame({"user": df_inter[user_col].astype(str), "item": df_inter[item_col].astype(str)})

    if time_col and time_col in df_inter.columns:
        try:
            if np.issubdtype(df_inter[time_col].dtype, np.number):
                canon["time"] = df_inter[time_col].fillna(0).astype("int64")
            else:
                canon["time"] = pd.to_datetime(df_inter[time_col], errors="coerce").astype("int64") // 10**9
                canon["time"] = canon["time"].fillna(0).astype("int64")
        except Exception:
            canon["time"] = 0
    else:
        canon["time"] = 0

    canon["label"] = 1.0

    df_items = df_items.copy()
    if max_docs is not None:
        df_items = df_items.iloc[: int(max_docs)].copy()

    docs = _build_item_text(df_items, item_col=item_col, text_cols=item_text_cols)
    docs = docs.rename(columns={item_col: "item"})
    docs["item"] = docs["item"].astype(str)

    return canon, docs


def _build_user_profile_queries(
    interactions: pd.DataFrame,
    item_docs: pd.DataFrame,
    max_profile_items: int,
) -> pd.DataFrame:
    item2text = dict(zip(item_docs["item"].astype(str).tolist(), item_docs["doc_text"].astype(str).tolist()))
    profiles = []
    grouped = interactions.groupby("user", sort=False)

    for u, g in grouped:
        items = g["item"].astype(str).tolist()
        if max_profile_items and len(items) > max_profile_items:
            items = items[-max_profile_items:]

        parts = []
        for it in items:
            t = item2text.get(it, "")
            if t:
                parts.append(t)

        q = " ".join(parts).strip()
        profiles.append((str(u), q))

    return pd.DataFrame(profiles, columns=["user", "query_text"])


def _safe_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8", errors="ignore")
        except Exception:
            return str(x)
    return str(x)


def _clean_name(x: Any) -> str:
    s = _safe_text(x).strip().lower()
    s = " ".join(s.split())
    return s


def _find_first(root: Path, filename: str) -> Optional[Path]:
    try:
        for p in root.rglob(filename):
            return p
    except Exception:
        return None
    return None


def _download_hetrec_2k_if_needed(dataset_path: Path, url: str) -> None:
    tags_fp = _find_first(dataset_path, "tags.dat")
    uta_fp = _find_first(dataset_path, "user_taggedartists.dat")
    art_fp = _find_first(dataset_path, "artists.dat")

    if tags_fp and uta_fp and art_fp:
        return

    print("[RetrievalAsUser][LastFM][tag_query] Downloading HetRec2011 LastFM-2K (for tags)...")
    dataset_path.mkdir(parents=True, exist_ok=True)

    content = None
    try:
        import requests

        resp = requests.get(url, timeout=60, allow_redirects=True)
        resp.raise_for_status()
        content = resp.content
    except Exception:
        try:
            import urllib.request

            with urllib.request.urlopen(url, timeout=60) as r:
                content = r.read()
        except Exception as e:
            raise RuntimeError(
                f"[LastFM tag_query] Failed to download HetRec 2K. "
                f"Place tags.dat / user_taggedartists.dat / artists.dat manually into {dataset_path}. "
                f"Error: {e}"
            )

    import zipfile
    import io

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            z.extractall(dataset_path)
    except Exception as e:
        raise RuntimeError(f"[LastFM tag_query] Failed to extract HetRec 2K zip: {e}")

    tags_fp = _find_first(dataset_path, "tags.dat")
    uta_fp = _find_first(dataset_path, "user_taggedartists.dat")
    art_fp = _find_first(dataset_path, "artists.dat")
    if not (tags_fp and uta_fp and art_fp):
        raise RuntimeError(
            "[LastFM tag_query] Download/extract completed, but tags.dat / user_taggedartists.dat / artists.dat "
            "were not found. Check the extracted folder structure."
        )


def _load_hetrec_2k_tables(dataset_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tags_fp = _find_first(dataset_path, "tags.dat")
    uta_fp = _find_first(dataset_path, "user_taggedartists.dat")
    art_fp = _find_first(dataset_path, "artists.dat")
    if not (tags_fp and uta_fp and art_fp):
        raise FileNotFoundError(
            f"HetRec 2K files not found in {dataset_path} (or subfolders)."
        )

    tags_df = pd.read_csv(tags_fp, sep="\t", encoding="latin-1", on_bad_lines="skip")
    tagging_df = pd.read_csv(uta_fp, sep="\t", encoding="latin-1", on_bad_lines="skip")
    artists_2k_df = pd.read_csv(art_fp, sep="\t", encoding="latin-1", on_bad_lines="skip")

    if "tagValue" not in tags_df.columns:
        cand = [c for c in tags_df.columns if "value" in c.lower() or "tag" in c.lower()]
        if cand:
            tags_df = tags_df.rename(columns={cand[-1]: "tagValue"})
    if "tagID" not in tags_df.columns:
        cand = [c for c in tags_df.columns if "id" in c.lower()]
        if cand:
            tags_df = tags_df.rename(columns={cand[0]: "tagID"})

    if "artistID" not in tagging_df.columns:
        cand = [c for c in tagging_df.columns if "artist" in c.lower()]
        if cand:
            tagging_df = tagging_df.rename(columns={cand[0]: "artistID"})
    if "tagID" not in tagging_df.columns:
        cand = [c for c in tagging_df.columns if "tag" in c.lower() and "id" in c.lower()]
        if cand:
            tagging_df = tagging_df.rename(columns={cand[0]: "tagID"})

    if "name" not in artists_2k_df.columns:
        cand = [c for c in artists_2k_df.columns if "name" in c.lower()]
        if cand:
            artists_2k_df = artists_2k_df.rename(columns={cand[0]: "name"})
    if "id" not in artists_2k_df.columns:
        cand = [c for c in artists_2k_df.columns if c.lower() in ("artistid", "id", "artist_id")]
        if cand:
            artists_2k_df = artists_2k_df.rename(columns={cand[0]: "id"})

    for c in ("tagID", "tagValue"):
        if c not in tags_df.columns:
            raise ValueError(f"HetRec tags.dat is missing column '{c}'. Columns={list(tags_df.columns)}")
    for c in ("artistID", "tagID"):
        if c not in tagging_df.columns:
            raise ValueError(
                f"HetRec user_taggedartists.dat is missing column '{c}'. Columns={list(tagging_df.columns)}"
            )
    for c in ("id", "name"):
        if c not in artists_2k_df.columns:
            raise ValueError(f"HetRec artists.dat is missing column '{c}'. Columns={list(artists_2k_df.columns)}")

    tags_df["tagValue"] = tags_df["tagValue"].astype(str).map(_clean_name)
    artists_2k_df["name_clean"] = artists_2k_df["name"].astype(str).map(_clean_name)

    return tags_df, tagging_df, artists_2k_df


def _build_lastfm_tag_assets_from_hetrec(
    dataset_path: Path,
    items_docs: pd.DataFrame,
    min_tag_interactions: int,
    min_relevance: float,
    max_tags_per_doc: int,
    hetr_ec_url: str,
) -> Tuple[pd.DataFrame, List[str]]:
    _download_hetrec_2k_if_needed(dataset_path, hetr_ec_url)
    tags_df, tagging_df, artists_2k_df = _load_hetrec_2k_tables(dataset_path)

    df_tags_full = tagging_df.merge(tags_df[["tagID", "tagValue"]], on="tagID", how="inner")
    df_tags_full = df_tags_full.merge(
        artists_2k_df[["id", "name_clean"]],
        left_on="artistID",
        right_on="id",
        how="inner",
    )

    df_group = (
        df_tags_full.groupby(["name_clean", "tagValue"])
        .size()
        .reset_index(name="tag_weight")
    )
    if df_group.empty:
        raise RuntimeError("[LastFM tag_query] HetRec 2K loaded, but did not produce (artist,tag) weights.")

    max_w = df_group.groupby("name_clean")["tag_weight"].transform("max")
    df_group["relevance"] = df_group["tag_weight"] / (max_w.astype(float) + 1e-9)

    if float(min_relevance) > 0:
        df_group = df_group[df_group["relevance"] >= float(min_relevance)].copy()

    if df_group.empty:
        raise RuntimeError(
            f"[LastFM tag_query] After min_relevance={min_relevance}, no (artist,tag) pairs remained."
        )

    if int(min_tag_interactions) > 0:
        tag_global = df_group.groupby("tagValue")["tag_weight"].sum()
        valid_tags = tag_global[tag_global >= int(min_tag_interactions)].index
        df_group = df_group[df_group["tagValue"].isin(valid_tags)].copy()

    if df_group.empty:
        raise RuntimeError(
            f"[LastFM tag_query] After min_tag_interactions={min_tag_interactions}, no tags remained."
        )

    df_group = df_group.sort_values(["name_clean", "relevance", "tag_weight"], ascending=[True, False, False])
    df_best = df_group.groupby("name_clean", sort=False).head(int(max_tags_per_doc)).copy()

    queries = df_best["tagValue"].dropna().astype(str).unique().tolist()

    docs = items_docs.copy()
    docs["base_name"] = docs["doc_text"].map(_safe_text).map(lambda s: s.strip())
    docs["name_clean"] = docs["base_name"].map(_clean_name)

    tag_join = (
        df_best.groupby("name_clean")["tagValue"]
        .agg(lambda xs: " \n " + " ".join([str(x) for x in xs if str(x).strip()]))
        .to_dict()
    )

    docs["tag_text"] = docs["name_clean"].map(tag_join).fillna("")
    docs["doc_text"] = (docs["base_name"].astype(str) + docs["tag_text"].astype(str)).astype(str)

    docs = docs[["item", "doc_text"]].copy()
    docs["item"] = docs["item"].astype(str)
    docs["doc_text"] = docs["doc_text"].astype(str)

    return docs, queries


def _load_items_docs_only(
    dataset_path: Path,
    base_loader: Optional[Dict[str, Any]],
    items_file: Optional[str],
    user_col: str,
    item_col: str,
    time_col: Optional[str],
    rating_col: Optional[str],
    item_text_cols: Optional[List[str]],
    max_docs: Optional[int],
) -> pd.DataFrame:
    items_fp: Optional[Path] = None
    if items_file:
        fp = dataset_path / items_file
        if fp.exists():
            items_fp = fp

    if items_fp is None:
        try:
            items_fp = _infer_items_file(dataset_path)
        except FileNotFoundError:
            items_fp = None

    if items_fp is None and base_loader:
        _materialize_from_base_loader(
            dataset_path=dataset_path,
            base_loader=base_loader,
            user_col=user_col,
            item_col=item_col,
            time_col=time_col,
            rating_col=rating_col,
            item_text_cols=item_text_cols,
        )
        items_fp = _infer_items_file(dataset_path)

    if items_fp is None:
        raise FileNotFoundError(
            f"[LastFM tag_query] Items file not found in {dataset_path}. "
            f"Provide items_file or a base_loader."
        )

    df_items = _read_any(items_fp)

    if item_col not in df_items.columns:
        raise ValueError(f"items: item_col '{item_col}' does not exist. Columns={list(df_items.columns)}")

    if max_docs is not None:
        df_items = df_items.iloc[: int(max_docs)].copy()

    docs = _build_item_text(df_items, item_col=item_col, text_cols=item_text_cols)
    docs = docs.rename(columns={item_col: "item"})
    docs["item"] = docs["item"].astype(str)

    docs["doc_text"] = docs["doc_text"].map(_safe_text)

    return docs


@dataclass
class RetrievalAsUserConfig:
    path: str = "./data/ml-25m"
    mode: str = "movielens_genome"

    seed: int = 42
    top_k: int = 100
    query_limit: Optional[int] = None

    min_relevance: float = 0.9
    max_tags_per_doc: int = 30

    flush_every: int = 250_000
    progress_every: int = 200

    base_loader: Optional[Dict[str, Any]] = None
    interactions_file: Optional[str] = None
    items_file: Optional[str] = None
    user_col: str = "user"
    item_col: str = "item"
    time_col: Optional[str] = "time"
    rating_col: Optional[str] = None
    min_rating: float = 1.0
    item_text_cols: Optional[List[str]] = None
    max_docs: Optional[int] = None
    max_profile_items: int = 50

    min_tag_interactions: int = 0
    hetr_ec_tags_url: str = "http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"

    amazon_meta_url: str = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Electronics.json.gz"
    amazon_meta_file: str = "meta_Electronics.json.gz"
    amazon_max_products: int = 200000
    amazon_cache_pairs: str = "amazon_search_pairs.parquet"
    amazon_min_cat_len: int = 3
    amazon_min_docs_per_query: int = 10

    bm25_config: Optional[Dict[str, Any]] = None
    label_mode: str = "binary"


class RetrievalAsUserDataLoader(RecsDataLoader):
    def __init__(self, full_config: dict):
        dl = full_config.get("dataloader", full_config) or {}
        super().__init__(dl)

        self.cfg = self._parse_cfg(dl)
        self.dataset_path = _resolve_dataset_path(self.cfg.path)

        print(f"[RetrievalAsUser] dataset_path: {self.dataset_path} | mode={self.cfg.mode}")

    def _parse_cfg(self, dl: dict) -> RetrievalAsUserConfig:
        bm25_cfg = dl.get("bm25_config", {}) or {}

        itc = dl.get("item_text_cols")
        if itc is None and dl.get("item_text_col"):
            itc = [str(dl.get("item_text_col"))]

        mpi = dl.get("max_profile_items")
        if mpi is None and dl.get("profile_topn") is not None:
            mpi = int(dl.get("profile_topn"))

        return RetrievalAsUserConfig(
            path=str(dl.get("path", "./data/ml-25m")),
            mode=str(dl.get("mode", dl.get("dataset_mode", "movielens_genome"))),

            seed=int(dl.get("seed", 42)),
            top_k=int(dl.get("top_k", 100)),
            query_limit=dl.get("query_limit"),

            min_relevance=float(dl.get("min_relevance", 0.9)),
            max_tags_per_doc=int(dl.get("max_tags_per_doc", 30)),

            flush_every=int(dl.get("flush_every", 250_000)),
            progress_every=int(dl.get("progress_every", 200)),

            base_loader=dl.get("base_loader"),
            interactions_file=dl.get("interactions_file"),
            items_file=dl.get("items_file"),
            user_col=str(dl.get("user_col", "user")),
            item_col=str(dl.get("item_col", "item")),
            time_col=dl.get("time_col", "time"),
            rating_col=dl.get("rating_col"),
            min_rating=float(dl.get("min_rating", 1.0)),
            item_text_cols=itc,
            max_docs=dl.get("max_docs"),
            max_profile_items=int(mpi) if mpi is not None else 50,

            min_tag_interactions=int(dl.get("min_tag_interactions", 0) or 0),
            hetr_ec_tags_url=str(
                dl.get(
                    "hetrec_tags_url",
                    "http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip",
                )
            ),

            amazon_meta_url=str(
                dl.get(
                    "amazon_meta_url",
                    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Electronics.json.gz",
                )
            ),
            amazon_meta_file=str(dl.get("amazon_meta_file", "meta_Electronics.json.gz")),
            amazon_max_products=int(dl.get("amazon_max_products", 200000) or 200000),
            amazon_cache_pairs=str(dl.get("amazon_cache_pairs", "amazon_search_pairs.parquet")),
            amazon_min_cat_len=int(dl.get("amazon_min_cat_len", 3) or 3),
            amazon_min_docs_per_query=int(dl.get("amazon_min_docs_per_query", 10) or 10),

            bm25_config=bm25_cfg,
            label_mode=str(dl.get("label_mode", "binary")),
        )

    def load_data(self) -> pd.DataFrame:
        mode = str(self.cfg.mode).lower().strip()
        if mode == "movielens_genome":
            return self._build_movielens_query_as_user()
        if mode == "profile_query":
            return self._build_profile_query_as_user()
        if mode == "tag_query":
            return self._build_lastfm_tag_query_as_user()
        if mode in {"amazon_category", "amazon_electronics", "amazon"}:
            return self._build_amazon_category_query_as_user()

        raise ValueError(
            f"Unknown mode='{self.cfg.mode}'. Use 'movielens_genome', 'profile_query', 'tag_query', or 'amazon_category'."
        )

    def _build_movielens_query_as_user(self) -> pd.DataFrame:
        bm25_cfg = dict(self.cfg.bm25_config or {})
        query_col = bm25_cfg.get("query_col", "search_query")
        doc_col = bm25_cfg.get("doc_col", "document")
        doc_id_col = bm25_cfg.get("doc_id_col", "document_id")

        print("[RetrievalAsUser][MovieLens] Building search_df (tag -> movie doc)...")
        search_df = _load_movielens_search_dataset(
            dataset_path=self.dataset_path,
            min_relevance=self.cfg.min_relevance,
            query_col=query_col,
            doc_col=doc_col,
            doc_id_col=doc_id_col,
            max_tags_per_doc=self.cfg.max_tags_per_doc,
        )

        unique_docs = search_df[[doc_id_col, doc_col]].drop_duplicates(subset=[doc_id_col]).copy()
        print(f"[RetrievalAsUser][MovieLens] Unique docs for indexing: {len(unique_docs)}")

        _ensure_nltk()
        bm25_index_cfg = _bm25_cfg_for_indexing(bm25_cfg)
        bm25 = _make_bm25_model(bm25_index_cfg, dataset_path=self.dataset_path)
        bm25.preprocess(train_data=unique_docs)
        bm25.fit()

        queries = search_df[query_col].dropna().astype(str).unique()

        if self.cfg.query_limit is not None:
            ql = int(self.cfg.query_limit)
            if ql > 0 and len(queries) > ql:
                rng = np.random.default_rng(self.cfg.seed)
                rng.shuffle(queries)
                queries = queries[:ql]

        print(f"[RetrievalAsUser][MovieLens] Queries (tags) to search: {len(queries)}")

        rows: List[Dict[str, Any]] = []
        chunks: List[pd.DataFrame] = []
        flush_every = max(10_000, int(self.cfg.flush_every))

        for q_idx, q in enumerate(queries):
            top_pairs = _bm25_search_topk(bm25, q, top_k=int(self.cfg.top_k))
            if not top_pairs:
                continue

            u = f"retrieval::{q}"
            for rank, (doc_id, score) in enumerate(top_pairs):
                try:
                    s = float(score)
                except Exception:
                    continue
                if s <= 0:
                    continue

                it = str(doc_id)
                rows.append({"user": u, "item": it, "time": int(rank), "score": float(s)})

            if len(rows) >= flush_every:
                chunks.append(pd.DataFrame(rows))
                rows = []

            if int(self.cfg.progress_every) > 0 and (q_idx + 1) % int(self.cfg.progress_every) == 0:
                built = sum(len(c) for c in chunks) + len(rows)
                print(f"[RetrievalAsUser][MovieLens] {q_idx+1}/{len(queries)} | rows={built}")

        if rows:
            chunks.append(pd.DataFrame(rows))

        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        if df.empty:
            return df

        stats = df.groupby("user")["score"].agg(["min", "max"]).reset_index().rename(columns={"min": "_min", "max": "_max"})
        df = df.merge(stats, on="user", how="left")
        df["score"] = (df["score"] - df["_min"]) / (df["_max"] - df["_min"] + 1e-9)
        df = df.drop(columns=["_min", "_max"])

        if str(self.cfg.label_mode).lower() == "dense":
            df["label"] = df["score"].astype(float)
        else:
            df["label"] = 1.0

        return df[["user", "item", "label", "time", "score"]].copy()

    def _build_profile_query_as_user(self) -> pd.DataFrame:
        positives, item_docs = _load_profile_query_inputs(
            dataset_path=self.dataset_path,
            base_loader=self.cfg.base_loader,
            interactions_file=self.cfg.interactions_file,
            items_file=self.cfg.items_file,
            user_col=self.cfg.user_col,
            item_col=self.cfg.item_col,
            time_col=self.cfg.time_col,
            rating_col=self.cfg.rating_col,
            min_rating=self.cfg.min_rating,
            item_text_cols=self.cfg.item_text_cols,
            max_docs=self.cfg.max_docs,
        )

        user_queries = _build_user_profile_queries(
            interactions=positives,
            item_docs=item_docs,
            max_profile_items=self.cfg.max_profile_items,
        )

        if self.cfg.query_limit is not None:
            ql = int(self.cfg.query_limit)
            if ql > 0 and len(user_queries) > ql:
                rng = np.random.default_rng(self.cfg.seed)
                idx = np.arange(len(user_queries))
                rng.shuffle(idx)
                user_queries = user_queries.iloc[idx[:ql]].copy()

        bm25_cfg = dict(self.cfg.bm25_config or {})
        bm25_index_cfg = _bm25_cfg_for_indexing(bm25_cfg)
        dcol = bm25_cfg.get("doc_col", "document")
        did = bm25_cfg.get("doc_id_col", "document_id")

        docs_df = item_docs.copy().rename(columns={"item": did, "doc_text": dcol})
        docs_df[did] = docs_df[did].astype(str)
        docs_df[dcol] = docs_df[dcol].astype(str).fillna("")

        _ensure_nltk()
        bm25 = _make_bm25_model(bm25_index_cfg, dataset_path=self.dataset_path)
        bm25.preprocess(train_data=docs_df)
        bm25.fit()

        rows_u: List[str] = []
        rows_i: List[str] = []
        rows_t: List[int] = []
        rows_s: List[float] = []
        chunks: List[pd.DataFrame] = []
        flush_every = max(10_000, int(self.cfg.flush_every))

        for idx_row, r in enumerate(user_queries.itertuples(index=False), start=1):
            u = str(getattr(r, "user"))
            qt = str(getattr(r, "query_text", "") or "").strip()
            if not qt:
                continue

            top_pairs = _bm25_search_topk(bm25, qt, top_k=int(self.cfg.top_k))
            if not top_pairs:
                continue

            for rank, (doc_id, score) in enumerate(top_pairs):
                try:
                    s = float(score)
                except Exception:
                    continue
                if s <= 0:
                    continue

                it = str(doc_id)
                rows_u.append(u)
                rows_i.append(it)
                rows_t.append(int(rank))
                rows_s.append(float(s))

            if len(rows_u) >= flush_every:
                chunks.append(pd.DataFrame({"user": rows_u, "item": rows_i, "time": rows_t, "score": rows_s}))
                rows_u, rows_i, rows_t, rows_s = [], [], [], []

            if int(self.cfg.progress_every) > 0 and (idx_row % int(self.cfg.progress_every) == 0):
                built = sum(len(c) for c in chunks) + len(rows_u)
                print(f"[RetrievalAsUser][Profile] {idx_row}/{len(user_queries)} | rows={built}")

        if rows_u:
            chunks.append(pd.DataFrame({"user": rows_u, "item": rows_i, "time": rows_t, "score": rows_s}))

        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        if df.empty:
            return df

        stats = df.groupby("user")["score"].agg(["min", "max"]).reset_index().rename(columns={"min": "_min", "max": "_max"})
        df = df.merge(stats, on="user", how="left")
        df["score"] = (df["score"] - df["_min"]) / (df["_max"] - df["_min"] + 1e-9)
        df = df.drop(columns=["_min", "_max"])

        if str(self.cfg.label_mode).lower() == "dense":
            df["label"] = df["score"].astype(float)
        else:
            df["label"] = 1.0

        return df[["user", "item", "label", "time", "score"]].copy()

    def _build_lastfm_tag_query_as_user(self) -> pd.DataFrame:
        print("[RetrievalAsUser][LastFM][tag_query] Building tags via HetRec 2K and mapping to 360K (by name)...")

        item_docs = _load_items_docs_only(
            dataset_path=self.dataset_path,
            base_loader=self.cfg.base_loader,
            items_file=self.cfg.items_file,
            user_col=self.cfg.user_col,
            item_col=self.cfg.item_col,
            time_col=self.cfg.time_col,
            rating_col=self.cfg.rating_col,
            item_text_cols=self.cfg.item_text_cols,
            max_docs=self.cfg.max_docs,
        )

        docs_with_tags, queries = _build_lastfm_tag_assets_from_hetrec(
            dataset_path=self.dataset_path,
            items_docs=item_docs,
            min_tag_interactions=int(self.cfg.min_tag_interactions or 0),
            min_relevance=float(self.cfg.min_relevance or 0.0),
            max_tags_per_doc=int(self.cfg.max_tags_per_doc or 20),
            hetr_ec_url=str(self.cfg.hetr_ec_tags_url),
        )

        print(f"[RetrievalAsUser][LastFM][tag_query] Docs (artists) indexed: {len(docs_with_tags)}")
        print(f"[RetrievalAsUser][LastFM][tag_query] Candidate queries (tags): {len(queries)}")

        if self.cfg.query_limit is not None:
            ql = int(self.cfg.query_limit)
            if ql > 0 and len(queries) > ql:
                rng = np.random.default_rng(self.cfg.seed)
                rng.shuffle(queries)
                queries = queries[:ql]

        print(f"[RetrievalAsUser][LastFM][tag_query] Queries (tags) to search: {len(queries)}")

        bm25_cfg = dict(self.cfg.bm25_config or {})
        dcol = bm25_cfg.get("doc_col", "document")
        did = bm25_cfg.get("doc_id_col", "document_id")

        docs_df = docs_with_tags.rename(columns={"item": did, "doc_text": dcol}).copy()
        docs_df[did] = docs_df[did].astype(str)
        docs_df[dcol] = docs_df[dcol].astype(str).fillna("")

        _ensure_nltk()
        bm25_index_cfg = _bm25_cfg_for_indexing(bm25_cfg)
        bm25 = _make_bm25_model(bm25_index_cfg, dataset_path=self.dataset_path)
        bm25.preprocess(train_data=docs_df)
        bm25.fit()

        rows: List[Dict[str, Any]] = []
        chunks: List[pd.DataFrame] = []
        flush_every = max(10_000, int(self.cfg.flush_every))

        for q_idx, q in enumerate(queries, start=1):
            top_pairs = _bm25_search_topk(bm25, q, top_k=int(self.cfg.top_k))
            if not top_pairs:
                continue

            u = f"retrieval::{q}"
            for rank, (doc_id, score) in enumerate(top_pairs):
                try:
                    s = float(score)
                except Exception:
                    continue
                if s <= 0:
                    continue

                it = str(doc_id)
                rows.append({"user": u, "item": it, "time": int(rank), "score": float(s)})

            if len(rows) >= flush_every:
                chunks.append(pd.DataFrame(rows))
                rows = []

            if int(self.cfg.progress_every) > 0 and (q_idx % int(self.cfg.progress_every) == 0):
                built = sum(len(c) for c in chunks) + len(rows)
                print(f"[RetrievalAsUser][LastFM][tag_query] {q_idx}/{len(queries)} | rows={built}")

        if rows:
            chunks.append(pd.DataFrame(rows))

        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        if df.empty:
            return df

        stats = df.groupby("user")["score"].agg(["min", "max"]).reset_index().rename(columns={"min": "_min", "max": "_max"})
        df = df.merge(stats, on="user", how="left")
        df["score"] = (df["score"] - df["_min"]) / (df["_max"] - df["_min"] + 1e-9)
        df = df.drop(columns=["_min", "_max"])

        if str(self.cfg.label_mode).lower() == "dense":
            df["label"] = df["score"].astype(float)
        else:
            df["label"] = 1.0

        return df[["user", "item", "label", "time", "score"]].copy()

    def _build_amazon_category_query_as_user(self) -> pd.DataFrame:
        bm25_cfg = dict(self.cfg.bm25_config or {})
        query_col = bm25_cfg.get("query_col", "search_query")
        doc_col = bm25_cfg.get("doc_col", "document")
        doc_id_col = bm25_cfg.get("doc_id_col", "document_id")

        print("[RetrievalAsUser][Amazon] Building search_df (category -> product doc)...")
        search_df = _load_amazon_category_search_dataset(
            dataset_path=self.dataset_path,
            meta_url=str(self.cfg.amazon_meta_url),
            meta_file=str(self.cfg.amazon_meta_file),
            cache_pairs=str(self.cfg.amazon_cache_pairs),
            max_products=int(self.cfg.amazon_max_products or 0),
            min_cat_len=int(self.cfg.amazon_min_cat_len or 3),
            min_docs_per_query=int(self.cfg.amazon_min_docs_per_query or 10),
            query_col=query_col,
            doc_col=doc_col,
            doc_id_col=doc_id_col,
        )

        if search_df is None or search_df.empty:
            print("[RetrievalAsUser][Amazon] search_df is empty. Nothing to do.")
            return pd.DataFrame()

        unique_docs = search_df[[doc_id_col, doc_col]].drop_duplicates(subset=[doc_id_col]).copy()
        print(f"[RetrievalAsUser][Amazon] Unique docs for indexing: {len(unique_docs)}")

        _ensure_nltk()
        bm25 = _make_bm25_model(bm25_cfg, dataset_path=self.dataset_path)
        bm25.preprocess(train_data=unique_docs)
        bm25.fit()

        queries = search_df[query_col].dropna().astype(str).unique()

        if self.cfg.query_limit is not None:
            ql = int(self.cfg.query_limit)
            if ql > 0 and len(queries) > ql:
                rng = np.random.default_rng(self.cfg.seed)
                rng.shuffle(queries)
                queries = queries[:ql]

        print(f"[RetrievalAsUser][Amazon] Queries (categories) to search: {len(queries)}")

        rows: List[Dict[str, Any]] = []
        chunks: List[pd.DataFrame] = []
        flush_every = max(10_000, int(self.cfg.flush_every))

        for q_idx, q in enumerate(queries, start=1):
            top_pairs = _bm25_search_topk(bm25, q, top_k=int(self.cfg.top_k))
            if not top_pairs:
                continue

            u = f"retrieval::{q}"
            for rank, (doc_id, score) in enumerate(top_pairs):
                try:
                    s = float(score)
                except Exception:
                    continue
                if s <= 0:
                    continue

                it = str(doc_id)
                rows.append({"user": u, "item": it, "time": int(rank), "score": float(s)})

            if len(rows) >= flush_every:
                chunks.append(pd.DataFrame(rows))
                rows = []

            if int(self.cfg.progress_every) > 0 and (q_idx % int(self.cfg.progress_every) == 0):
                built = sum(len(c) for c in chunks) + len(rows)
                print(f"[RetrievalAsUser][Amazon] {q_idx}/{len(queries)} | rows={built}")

        if rows:
            chunks.append(pd.DataFrame(rows))

        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        if df.empty:
            return df

        stats = df.groupby("user")["score"].agg(["min", "max"]).reset_index().rename(columns={"min": "_min", "max": "_max"})
        df = df.merge(stats, on="user", how="left")
        df["score"] = (df["score"] - df["_min"]) / (df["_max"] - df["_min"] + 1e-9)
        df = df.drop(columns=["_min", "_max"])

        if str(self.cfg.label_mode).lower() == "dense":
            df["label"] = df["score"].astype(float)
        else:
            df["label"] = 1.0

        return df[["user", "item", "label", "time", "score"]].copy()


__all__ = ["RetrievalAsUserDataLoader", "RetrievalAsUserConfig"]
