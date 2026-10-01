"""Native search reference under the SAME protocol as the Search -> Rec adaptations, and Transfer Ratio.

For every evaluated pseudo-user (= a query) of an adaptation run, the candidates and labels saved
by framework.py in runs/<run_id>/eval_scores.parquet (positives + sampled negatives) are re-ranked
by two native search systems:

  bm25      BM25 on the query text alone
  bm25_fb   BM25 + Rocchio feedback from exactly the documents the recommender was trained on for
            that query (Query-as-User: its training qrels; Retrieval-as-User: its retrieved history),
            i.e. a native system with the same information as the adaptation

and the adaptation's own scores (``_score``) are evaluated with the same metric code, so the three
systems differ only in the ranking function.

Outputs (analysis/out/):
  native_per_user.csv   one row per experiment x run x query x system, metrics @5/10/20/50
  transfer_ratio.csv    per experiment x reference: mean metrics, TR = adapted / native (ratio of
                        means, bootstrap 95% CI over queries), Wilcoxon signed-rank over queries

Needs analysis/matrix_stats.py to have been run (reads analysis/out/experiments.json) and the
texts of the datasets:
  BEIR      data/beir-<subset>/  (corpus.jsonl, queries.jsonl) -- or --beir-dir
  MS MARCO  data/msmarco_trec_dl/2019/  (queries + top1000 files) -- or --msmarco-dir
  hybrid    data/hybrid_dataset/hybrid_dataset/  (items.parquet, queries.parquet) -- or --hybrid-dir

Usage:
  python analysis/native_reference.py [--groups base step1 fullrank] [--beir-dir DIR] [--msmarco-dir DIR]
"""

import argparse
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from search_recs.metric import NdcgAtK, PrecisionAtK, RecallAtK  # noqa: E402

TOP_KS = (5, 10, 20, 50)
METRICS = {"NDCG": NdcgAtK(), "PRECISION": PrecisionAtK(), "RECALL": RecallAtK()}
STOPWORDS = set("""a an and are as at be but by for from has have in is it its of on or that the this to was
were will with what which who how when where why do does did not no can i you we they he she de la le""".split())
TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str):
    return [t for t in TOKEN.findall(str(text).lower()) if len(t) > 1 and t not in STOPWORDS]


# ----------------------------------------------------------------------------- BM25
class BM25:
    def __init__(self, doc_ids, texts, k1=1.5, b=0.75):
        self.doc_index = {str(d): i for i, d in enumerate(doc_ids)}
        vocab, rows, cols, vals = {}, [], [], []
        for i, text in enumerate(texts):
            for term, tf in Counter(tokenize(text)).items():
                j = vocab.setdefault(term, len(vocab))
                rows.append(i)
                cols.append(j)
                vals.append(tf)
        self.vocab = vocab
        tf = sp.csr_matrix((np.asarray(vals, dtype=np.float64), (rows, cols)), shape=(len(texts), len(vocab)))
        dl = np.asarray(tf.sum(axis=1)).ravel()
        df = np.bincount(tf.indices, minlength=len(vocab))
        idf = np.log(1 + (len(texts) - df + 0.5) / (df + 0.5))
        norm = k1 * (1 - b + b * dl / max(dl.mean(), 1e-9))
        W = tf.tocoo()
        data = idf[W.col] * W.data * (k1 + 1) / (W.data + norm[W.row])
        self.W = sp.csr_matrix((data, (W.row, W.col)), shape=tf.shape)

    def query_vector(self, text: str) -> np.ndarray:
        q = np.zeros(len(self.vocab))
        for term in tokenize(text):
            j = self.vocab.get(term)
            if j is not None:
                q[j] += 1.0
        return q

    def feedback_vector(self, q: np.ndarray, feedback_docs, alpha=0.5, n_terms=20) -> np.ndarray:
        idx = [self.doc_index[d] for d in feedback_docs if d in self.doc_index]
        if not idx:
            return q
        rows = self.W[idx]
        lens = np.sqrt(np.asarray(rows.multiply(rows).sum(axis=1)).ravel()) + 1e-12
        centroid = np.asarray((sp.diags(1 / lens) @ rows).mean(axis=0)).ravel()
        if n_terms and np.count_nonzero(centroid) > n_terms:
            cut = np.partition(centroid, -n_terms)[-n_terms]
            centroid = np.where(centroid >= cut, centroid, 0.0)
        qn, cn = np.linalg.norm(q), np.linalg.norm(centroid)
        if cn == 0:
            return q
        return (alpha * q / qn if qn else 0) + (1 - alpha) * centroid / cn

    def score(self, q: np.ndarray, doc_ids) -> np.ndarray:
        idx = np.array([self.doc_index.get(str(d), -1) for d in doc_ids])
        out = np.full(len(idx), -np.inf)
        known = idx >= 0
        if known.any():
            out[known] = self.W[idx[known]] @ q
        return out


# ----------------------------------------------------------------------------- texts
def load_beir(beir_dir: Path):
    def jsonl(p):
        with p.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    corpus = jsonl(beir_dir / "corpus.jsonl")
    queries = {str(r["_id"]): r.get("text", "") for r in jsonl(beir_dir / "queries.jsonl")}
    ids = [str(r["_id"]) for r in corpus]
    texts = [f"{r.get('title', '')} {r.get('text', '')}" for r in corpus]
    return queries, ids, texts


def load_msmarco(ms_dir: Path):
    """Same corpus as the framework's MS MARCO loaders: the top-1000 pools of the judged queries."""
    judged = set()
    with (ms_dir / "2019qrels-pass.txt").open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if parts:
                judged.add(parts[0])
    queries = {}
    with gzip.open(ms_dir / "msmarco-test2019-queries.tsv.gz", "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                queries[parts[0]] = parts[1]
    passages = {}
    with gzip.open(ms_dir / "msmarco-passagetest2019-top1000.tsv.gz", "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and parts[0] in judged:
                passages.setdefault(parts[1], parts[3])
    return queries, list(passages), list(passages.values())


def load_hybrid(hybrid_dir: Path):
    """Product catalogue and queries of the hybrid dataset. Same text fields as the hybrid loaders
    (title, description, category, brand); not imported from them to avoid the framework's heavy deps."""
    items = pd.read_parquet(hybrid_dir / "items.parquet")
    fields = items[["title", "description", "category", "brand"]].fillna("").astype(str)
    texts = fields.apply(lambda row: "\n".join(v.strip() for v in row if v.strip()), axis=1)
    q = pd.read_parquet(hybrid_dir / "queries.parquet")
    queries = dict(zip(q["query_id"].astype(str), q["query_text"].astype(str)))
    return queries, items["asin"].astype(str).tolist(), texts.tolist()


LOADERS = {"beir": load_beir, "msmarco": load_msmarco, "hybrid": load_hybrid}


# ----------------------------------------------------------------------------- evaluation
def evaluate(y_true, y_pred):
    out = {}
    for k in TOP_KS:
        for name, m in METRICS.items():
            out[f"{name}@{k}"] = float(m.evaluate_metric(y_pred=list(y_pred), y_true=list(y_true), topk=k))
    return out


def run_experiment(exp, index: BM25, queries: dict):
    rows = []
    matrices = Path(exp["matrices"])
    run_to_stem = {}
    for meta_path in matrices.glob("*_meta.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        run_to_stem[meta["run_id"]] = meta_path.name[: -len("_meta.json")]

    for hash_dir in exp["hash_dirs"]:
        for scores_path in sorted(Path(hash_dir).glob("runs/*/eval_scores.parquet")):
            run_id = scores_path.parent.name
            if run_id not in run_to_stem or scores_path.stat().st_size == 0:
                continue
            scores = pd.read_parquet(scores_path)
            train = pd.read_parquet(matrices / f"{run_to_stem[run_id]}_train.parquet", columns=["user", "item"])
            history = train.assign(user=train["user"].astype(str), item=train["item"].astype(str)) \
                .groupby("user")["item"].apply(list).to_dict()
            rng = np.random.default_rng(0)
            for user, g in scores.groupby("user", sort=False):
                # eval_scores lists the positives first; a model that ties candidates (e.g. -inf for an
                # unknown user) would then be credited by the sort's tie order, which also differs across
                # numpy builds. A fixed random order, shared by all systems, breaks ties neutrally.
                g = g.iloc[rng.permutation(len(g))]
                y_true = (pd.to_numeric(g["label"], errors="coerce").fillna(0) > 0).astype(float).to_numpy()
                if y_true.sum() == 0:
                    continue
                qid = str(user).split("::", 1)[-1]
                qtext = queries.get(qid)
                if qtext is None:
                    continue
                q = index.query_vector(qtext)
                q_fb = index.feedback_vector(q, history.get(str(user), []))
                items = g["item"].astype(str).to_numpy()
                adapted = pd.to_numeric(g["_score"], errors="coerce").fillna(-1e30).to_numpy()
                base = {"experiment": exp["experiment"], "dataset": exp["dataset"], "strategy": exp["strategy"],
                        "group": exp["group"], "run_id": run_id, "user": user, "n_pos": int(y_true.sum()),
                        "n_feedback_docs": len(history.get(str(user), []))}
                for system, pred in (("adapted", adapted), ("bm25", index.score(q, items)),
                                     ("bm25_fb", index.score(q_fb, items))):
                    pred = np.where(np.isfinite(pred), pred, -1e30)
                    rows.append({**base, "system": system, **evaluate(y_true, pred)})
    return rows


def transfer_table(df: pd.DataFrame, n_boot=2000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metric_cols = [c for c in df.columns if "@" in c]
    # one value per query: mean over the runs in which it was evaluated
    per_q = df.groupby(["experiment", "dataset", "strategy", "group", "system", "user"])[metric_cols].mean().reset_index()
    out = []
    for (exp, ds, strat, grp), g in per_q.groupby(["experiment", "dataset", "strategy", "group"]):
        wide = g.pivot_table(index="user", columns="system", values=metric_cols)
        for ref in ("bm25", "bm25_fb"):
            for m in metric_cols:
                if (m, "adapted") not in wide or (m, ref) not in wide:
                    continue
                a, n = wide[(m, "adapted")].to_numpy(), wide[(m, ref)].to_numpy()
                ok = ~(np.isnan(a) | np.isnan(n))
                a, n = a[ok], n[ok]
                if len(a) == 0:
                    continue
                idx = rng.integers(0, len(a), size=(n_boot, len(a)))
                boot = a[idx].mean(axis=1) / np.maximum(n[idx].mean(axis=1), 1e-12)
                p = stats.wilcoxon(a, n).pvalue if np.any(a != n) else 1.0
                out.append({"experiment": exp, "dataset": ds, "strategy": strat, "group": grp, "reference": ref,
                            "metric": m, "n_queries": len(a), "adapted": a.mean(), "native": n.mean(),
                            "transfer_ratio": a.mean() / n.mean() if n.mean() > 0 else np.nan,
                            "tr_ci_low": np.percentile(boot, 2.5), "tr_ci_high": np.percentile(boot, 97.5),
                            "gap": n.mean() - a.mean(), "wilcoxon_p": p})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="analysis/out")
    ap.add_argument("--groups", nargs="+", default=["base", "step1", "fullrank"])
    ap.add_argument("--beir-dir", default=None, help="default: data/beir-nfcorpus")
    ap.add_argument("--msmarco-dir", default="data/msmarco_trec_dl/2019")
    ap.add_argument("--hybrid-dir", default="data/hybrid_dataset/hybrid_dataset")
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    args = ap.parse_args()
    out = Path(args.out)

    experiments = json.loads((out / "experiments.json").read_text(encoding="utf-8"))
    experiments = [e for e in experiments if e["group"] in args.groups]

    sources = {
        "beir": Path(args.beir_dir) if args.beir_dir else ROOT / "data" / "beir-nfcorpus",
        "msmarco": Path(args.msmarco_dir),
        "hybrid": Path(args.hybrid_dir),
    }
    rows = []
    for dataset in sorted({e["dataset"] for e in experiments}):
        src = sources.get(dataset)
        try:
            queries, ids, texts = LOADERS[dataset](src)
        except (FileNotFoundError, KeyError) as e:
            print(f"[native] skipping {dataset}: texts not found ({e}). Pass --{dataset}-dir.")
            continue
        index = BM25(ids, texts, k1=args.k1, b=args.b)
        print(f"[native] {dataset}: {len(ids)} documents indexed, {len(queries)} queries")
        for exp in [e for e in experiments if e["dataset"] == dataset]:
            got = run_experiment(exp, index, queries)
            print(f"[native]   {exp['experiment']}: {len(got) // 3} query x run evaluations")
            rows.extend(got)

    if not rows:
        raise SystemExit("[native] nothing evaluated.")
    per_user = pd.DataFrame(rows)
    per_user.to_csv(out / "native_per_user.csv", index=False)
    tr = transfer_table(per_user)
    tr.to_csv(out / "transfer_ratio.csv", index=False)

    pd.set_option("display.width", 220)
    show = tr[tr["metric"].isin(["NDCG@10", "RECALL@50"])]
    print("\n=== Transfer Ratio (adapted / native, same candidates and labels) ===")
    print(show[["experiment", "reference", "metric", "n_queries", "adapted", "native", "transfer_ratio",
                "tr_ci_low", "tr_ci_high", "wilcoxon_p"]].round(4).to_string(index=False))
    print(f"\n[native] wrote {out / 'native_per_user.csv'} and {out / 'transfer_ratio.csv'}")


if __name__ == "__main__":
    main()
