
# BR2IDGE - Benchmarking Recommendation and Retrieval Integration for Diverse and Generalized Evaluation

## Introduction

**BR2IDGE** is a **config-driven benchmarking framework** that unifies **Information Retrieval (Search)** and **Recommender Systems (Recs)** under a single, consistent pipeline.

Designed for reproducibility and extensibility, BR2IDGE allows researchers to execute Search and Recommendation experiments using a single runner. It features a plug-in architecture to easily integrate **new datasets**, **models**, and **ranking metrics**.

Key features include:

* **Unified Pipeline:** A single entry point (`framework.py`) for both Search and Recs tasks.
* **Shared Evaluation:** Consistent interface for ranking metrics (e.g., NDCG, Precision, Recall).
* **Bridging Strategies:** Built-in support for "inverse tasks," allowing the study of **Search → Recs** (using retrieval to build user interactions) and **Recs → Search** (transforming signals into query-document tasks).
* **Reproducibility:** Experiments are defined via structured JSON configs, utilizing config hashing for stable environments.
* **Efficiency:** Supports checkpoints and stage-level caching to skip or reuse expensive computation steps.

## Supported Models and How to Run Them

Some implemented models shown in the paper include `LightFMModel`, `DeepFMModel`, `Bert4REC`, `LightGCNModel`, `ItemKNNModel`, `BM25Model`, `DenseRetrieverModel`, `BiEncoderSearchModel`, `SpladeKnnModel`, and `ColBERTModel`; implemented datasets include `MovieLens` and `LastFM`; and implemented metrics include `NDCG`, `PRECISION`, and `RECALL`.

## Requirements

* **Python:** 3.10 (Required)
* **OS:** Linux (recommended for large-scale runs), macOS, or Windows.
* **Compute:**
  * CPU-only runs are fully supported.
  * GPU is optional (recommended for deep models, transformers or dense retrievers).
* **Storage/RAM:**
  * Plan accordingly when using large datasets (e.g., MovieLens 25M, Amazon Reviews).
  * Ensure sufficient disk space if `n_runs > 1` is configured.

## Installation
### Manual Installation

**Requires pip ≤ 25.2**

1. Before installing, downgrade pip if needed:
    ```bash
    python -m pip install "pip<=25.2"
    ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## Datasets

BR2IDGE downloads the benchmark datasets **automatically** the first time you run an
experiment that needs them. You normally do not have to download anything by hand.

### How automatic download works

When a dataloader is asked for a dataset, the framework (`search_recs/datasets/manager.py`):

1. Checks whether the dataset already exists under `data/<folder>/`. If the required
   files are present, it is used as-is.
2. If not, it looks for the dataset's `.zip` archive in (in order):
   `$BR2IDGE_DATA_ARCHIVE_DIR/`, then `data/_archives/`, then `data/`, then the repo root.
3. If no archive is found, it downloads the **Zenodo bundle** — a single
   "zip-of-zips" (~2 GB) that contains every dataset archive — and unpacks each inner
   `.zip` into `data/_archives/`.
4. It extracts the matching archive into `data/<folder>/` and verifies the required files.

The four datasets, their archives, and where they are extracted:

| Dataset (config `name`)        | Archive (`.zip`)             | Extracted folder            | Required files |
| ------------------------------ | ---------------------------- | --------------------------- | -------------- |
| `movielens` / MovieLens-25M    | `ml-25m.zip`                 | `data/ml-25m/`              | `ratings.csv`, `movies.csv`, `genome-scores.csv`, `genome-tags.csv`, `tags.csv`, `links.csv` |
| `amazonElectronics`            | `amazonElectronics.zip`      | `data/amazonElectronics/`   | `ratings.csv`, `meta_Electronics.json.gz` |
| `lastfm` (LastFM-360K, recs)   | `lastfm-dataset-360K.zip`    | `data/lastfm-dataset-360K/` | `usersha1-artmbid-artname-plays.tsv`, `usersha1-profile.tsv` |
| `lastfm` (HetRec hybrid, search) | `lastfm-hybrid.zip`        | `data/lastfm-hybrid/`       | `artists.dat`, `tags.dat`, `user_taggedartists.dat`, `user_artists.dat` |

### Manual download (if the automatic download fails)

If you are offline, behind a proxy, or the Zenodo download fails, you can fetch the
archives yourself from either mirror:

* **Zenodo:** https://zenodo.org/records/20492270
  (download the individual per-dataset `.zip` files, or the full "Download all" bundle)
* **Google Drive:** https://drive.google.com/drive/u/2/folders/1_Wvj_lLrqrahwi4d8aXC1bSzgz59_7iA

You do **not** need to download every dataset — only the ones your experiments use.

#### Where to put the files

You have two options. **Option A is the easiest** and the recommended one.

**Option A — Drop the `.zip` files into `data/_archives/` (recommended)**

Place the downloaded archives, *without unzipping them*, into the `data/_archives/`
folder. The framework will detect them and extract them automatically on the next run:

```
BR2IDGE/
└── data/
    └── _archives/
        ├── ml-25m.zip
        ├── amazonElectronics.zip
        ├── lastfm-dataset-360K.zip
        └── lastfm-hybrid.zip
```

Keep the archive filenames exactly as listed in the table above. Then just run your
experiment normally — extraction happens for you.

> Tip: you can keep the archives anywhere and point the framework at that directory with
> the `BR2IDGE_DATA_ARCHIVE_DIR` environment variable, e.g.
> `export BR2IDGE_DATA_ARCHIVE_DIR=/path/to/my/archives`.

**Option B — Extract the archives yourself into `data/<folder>/`**

If you prefer to unzip manually, extract each archive so that its **required files sit
directly inside** the matching `data/<folder>/` (no extra nested folder). For example,
for MovieLens-25M the result must be:

```
BR2IDGE/
└── data/
    └── ml-25m/
        ├── ratings.csv
        ├── movies.csv
        ├── genome-scores.csv
        ├── genome-tags.csv
        ├── tags.csv
        └── links.csv
```

Make sure the files are **directly** under the folder (e.g. `data/ml-25m/ratings.csv`),
**not** double-nested (`data/ml-25m/ml-25m/ratings.csv`). If the required files are found,
the framework skips downloading entirely.


## Hugging Face Token (SPLADE)

The SPLADE model (`SpladeKnnModel`) loads pretrained weights (`naver/splade-v3`) from the
Hugging Face Hub. Because this model is gated, you must supply a Hugging Face access token
before running any SPLADE experiment.

1. Create a (free) account at https://huggingface.co and accept the model terms on the
   model page: https://huggingface.co/naver/splade-v3
2. Generate a **read** token at https://huggingface.co/settings/tokens
3. Open `config_files/models/spladeknn.json` and paste your token into the `auth_token`
   field (replacing the `YOUR HF TOKEN` placeholder):

   ```json
   {
     "model": {
       "model_type": "SpladeKnnModel",
       "parameters": {
         "pretrained_name": "naver/splade-v3",
         "auth_token": "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
         "...": "..."
       }
     }
   }
   ```

>The token is your personal credential — do not commit it to a public repository.
> Only `SpladeKnnModel` requires it; the other models run without a Hugging Face token.


## Configs

BR2IDGE is driven by a hierarchical configuration system. You run experiments by passing a **General Experiment Config** (JSON) to the CLI, which resolves specific **Dataset** and **Model** configurations.

### 1. General Experiment Config

This is the entry point for any execution. It defines the task type, the components to use, and execution controls.

`task_type` must be one of:

* `recs`: runs recommendation models (RS domain)
* `search`: runs information retrieval models (IR domain)

Choose `task_type` first, then select compatible `dataset` and `model`.

**Example `my_experiment.json`:**

**JSON**

```json
{
  "experiment_name": "BM25_MovieLens_Baseline",
  "task_type": "search",
  "output_dir": "artifacts",

  "dataset": { "name": "movielens" },

  "model": [
    { "name": "BM25Model" }
  ],

  "execution": {
    "skip": [],
    "force": ["preprocess", "index"],
    "no_save": [],
    "n_runs": 1
  },

  "evaluation": {
    "top_ks": [5, 10, 20, 50],
    "metrics": ["NDCG", "PRECISION", "RECALL"]
  }
}
```

**How to configure dataset and model**

In your experiment file (`.json`), set:

```json
"dataset": { "name": "movielens" },
"model": [{ "name": "LightFMModel" }]
```

You can switch to another dataset/model by changing only these fields, for example:

```json
"dataset": { "name": "lastfm" },
"model": [{ "name": "BM25Model" }]
```

You can also run more than one model in the same experiment:

```json
"model": [
  { "name": "LightFMModel" },
  { "name": "DeepFMModel" },
  { "name": "Bert4REC" }
]
```

Run command pattern for your custom file:

```bash
python framework.py --config my_experiment.json
```

### 2. Dataset & Model Resolution

The framework automatically looks for configuration files in the `config_files/` directory based on the names provided in the experiment config.

* **Datasets:** Looks in `config_files/datasets/` for a file where `dataloader.dataset_name` matches your requested name.
* **Models:** Looks in `config_files/models/` for a file where `model.model_name` matches your requested model.

**Overriding Configs:**
You can bypass auto-resolution by providing an explicit path:

**JSON**

```json
"model": [
  {
    "name": "MyCustomModel",
    "config_path": "my_configs/models/custom_model.json"
  }
]
```

### 3. Execution Controls

The pipeline consists of five stages: `preprocess` → `features` → `train` → `index` → `eval`.
Use `execution` to control run flow at a high level:

* **`skip`**: skips listed stages
* **`force`**: forces listed stages to run again
* **`no_save`**: runs listed stages without persisting their artifacts
* **`n_runs`**: repeats the full experiment multiple times

For detailed cache/checkpoint behavior and resume logic, see **Topic 7 (Checkpoints and Resume)**.

### 4. Hybrid (Bridging) Mode (Unified for the 4 Strategies)

The 4 hybrid strategies are configured in the **same way** in the main experiment `.json`.

Add these fields right below `task_type`:

* **`hybrid`**: `true` or `false` (default: `false`)
* **`hybridStrategie`** (or `hybridStrategy`): only used when `hybrid=true`

Supported strategies:

* `"query-as-user"` (**recs**)
* `"retrieval-as-user"` (**recs**)
* `"centroid-vector"` (**search**)
* `"tag-query"` (**search**)

#### 4.1 Query-as-User (Recs) - Model used: `LightFMModel`

```json
{
  "experiment_name": "Hybrid_QueryAsUser_Recs",
  "task_type": "recs",
  "hybrid": true,
  "hybridStrategie": "query-as-user",
  "dataset": { "name": "movielens" },
  "model": [{ "name": "LightFMModel" }],
  "execution": { "n_runs": 1 },
  "evaluation": {
    "n_pos_samples": 20,
    "n_neg_samples": 200,
    "top_ks": [5, 10, 20, 50],
    "metrics": ["NDCG", "PRECISION", "RECALL"]
  }
}
```

Run:

```bash
python framework.py --config hybrid_query_as_user_recs.json
```

#### 4.2 Retrieval-as-User (Recs) - Model used: `LightFMModel`

```json
{
  "experiment_name": "Hybrid_RetrievalAsUser_Recs",
  "task_type": "recs",
  "hybrid": true,
  "hybridStrategie": "retrieval-as-user",
  "dataset": { "name": "amazonElectronics" },
  "model": [{ "name": "LightFMModel" }],
  "execution": { "n_runs": 1 },
  "evaluation": {
    "n_pos_samples": 50,
    "n_neg_samples": 1000,
    "top_ks": [5, 10, 20, 50],
    "metrics": ["NDCG", "PRECISION", "RECALL"]
  }
}
```

Run:

```bash
python framework.py --config hybrid_retrieval_as_user_recs.json
```

#### 4.3 Centroid-Vector (Search) - Model used: `BiEncoderSearchModel`

```json
{
  "experiment_name": "Hybrid_CentroidVector_Search",
  "task_type": "search",
  "hybrid": true,
  "hybridStrategie": "centroid-vector",
  "dataset": { "name": "movielens" },
  "model": [{ "name": "BiEncoderSearchModel" }],
  "execution": { "n_runs": 1 },
  "evaluation": {
    "top_ks": [5, 10, 20, 50],
    "metrics": ["NDCG", "PRECISION", "RECALL"]
  }
}
```

Run:

```bash
python framework.py --config hybrid_centroid_vector_search.json
```

#### 4.4 Tag-Query (Search) - Model used: `BM25Model`

```json
{
  "experiment_name": "Hybrid_TagQuery_Search",
  "task_type": "search",
  "hybrid": true,
  "hybridStrategie": "tag-query",
  "dataset": { "name": "lastfm" },
  "model": [{ "name": "BM25Model" }],
  "execution": { "n_runs": 1 },
  "evaluation": {
    "top_ks": [5, 10, 20, 50],
    "metrics": ["NDCG", "PRECISION", "RECALL"]
  }
}
```

Run:

```bash
python framework.py --config hybrid_tag_query_search.json
```

### 5. Positive/Negative Sampling Rules

Use the following standard values:

* **Recs models (non-hybrid)**:
  * All use **20 positives / 200 negatives**
* **Search models on Amazon Electronics**:
  * **50 positives / 1000 negatives**
* **Query-as-User** and **Retrieval-as-User**:
  * Amazon Electronics only: **50 positives / 1000 negatives**
* **Centroid Vector** and **Tag Query**:
  * MovieLens: **20 positives / 200 negatives**
  * LastFM: **20 positives / 200 negatives**
  * Amazon Electronics: **50 positives / 1000 negatives**

Set these in `evaluation`:

```json
"evaluation": {
  "n_pos_samples": 50,
  "n_neg_samples": 1000,
  "top_ks": [5, 10, 20, 50],
  "metrics": ["NDCG", "PRECISION", "RECALL"]
}
```

### 6. Statistical Tests (Wilcoxon and Paired t-test)

BR2IDGE can run statistical significance tests after evaluation, using the `evaluation.stats` block.

Available tests:

* `wilcoxon`
* `ttest_rel` (paired t-test)

Important:

* You must configure at least **2 models** in `model`.
* Use `execution.n_runs >= 2` for paired comparisons.
* If only one model is configured, tests are skipped.

**Example with both tests enabled:**

```json
{
  "execution": {
    "n_runs": 3
  },
  "evaluation": {
    "top_ks": [5, 10, 20, 50],
    "metrics": ["NDCG", "PRECISION", "RECALL"],
    "stats": {
      "tests": ["wilcoxon", "ttest_rel"],
      "min_runs": 2,
      "alpha": 0.05,
      "symbol": "▲"
    }
  }
}
```

Expected output files in `experimental_results/<dataset>/<task>/`:

* `<experiment_name>_wilcoxon_marked.csv`
* `<experiment_name>_wilcoxon_diagnostics.csv`
* `<experiment_name>_ttest_marked.csv`
* `<experiment_name>_ttest_diagnostics.csv`

### 7. Checkpoints and Resume

BR2IDGE saves run artifacts and automatically reuses them whenever possible.

Where files are stored:

* `artifacts/<dataset>/<model>/<config_hash>/runs/<run_id>/`
* Example files: `run_meta.json`, `metrics.json`, `model.pkl`, `last.pt`, `lightfm.npz`, `als_factors.npz`, `nmf_factors.npz`, `keras_weights.weights.h5`

How resume works:

* In `features` and `train`, the framework tries to load checkpoints from the current `run_dir`.
* If no checkpoint is found, it tries to load from a previous run under the same `<dataset>/<model>/<config_hash>/runs/`.
* In `eval`, when `skip` is active for `eval`, it tries to reuse `metrics.json` (including from previous runs) and avoids re-evaluating.

Operational behavior with `execution` flags:

* `skip` + existing artifacts: stage is reused from cache/checkpoint when available
* `force` on a stage: that stage runs again even if artifacts already exist
* `no_save` on a stage: stage runs, but no new artifact is persisted for that stage

Quick example:

```json
"execution": {
  "skip": ["preprocess", "features", "train"],
  "force": ["eval"],
  "no_save": [],
  "n_runs": 1
}
```

In this example, preprocess/features/train try to reuse artifacts, while `eval` is forced to run again.

## Minimal Working Example

### Running a Recommendation Experiment

To run a baseline recommender using the standard Recs pipeline:

**Bash**

```
python framework.py --config recs_example.json
```

### Running a Search Experiment

To run a baseline IR model using the standard IR pipeline:

**Bash**

```
python framework.py --config search_example.json
```

### Outputs

BR2IDGE writes outputs to two main locations:

* **Run artifacts and checkpoints:** `artifacts/<dataset>/<model>/<config_hash>/runs/<run_id>/`
* **Evaluation tables and statistical reports:** `experimental_results/<dataset>/<task>/`

Typical output files include:

* `metrics.json` per run
* `<experiment_name>_<run_id>_<model>_table.csv` (and `.tex`)
* `<experiment_name>_metrics_all.csv` (when multi-model stats are enabled)
* `<experiment_name>_wilcoxon_marked.csv`, `<experiment_name>_wilcoxon_diagnostics.csv`
* `<experiment_name>_ttest_marked.csv`, `<experiment_name>_ttest_diagnostics.csv`
