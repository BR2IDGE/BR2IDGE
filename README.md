
# BRIDGE

## Introduction

**BRIDGE** is a **config-driven benchmarking framework** that unifies **Information Retrieval (Search)** and **Recommender Systems (Recs)** under a single, consistent pipeline.

Designed for reproducibility and extensibility, BRIDGE allows researchers to execute Search and Recommendation experiments using a single runner. It features a plug-in architecture to easily integrate  **new datasets** ,  **models** , and  **ranking metrics** .

Key features include:

* **Unified Pipeline:** A single entry point (`framework.py`) for both Search and Recs tasks.
* **Shared Evaluation:** Consistent interface for ranking metrics (NDCG, Recall, Precision, MAP, MRR).
* **Bridging Strategies:** Built-in support for "inverse tasks," allowing the study of **Search → Recs** (using retrieval to build user interactions) and **Recs → Search** (transforming signals into query-document tasks).
* **Reproducibility:** Experiments are defined via structured JSON configs, utilizing config hashing for stable environments.
* **Efficiency:** Supports checkpoints and stage-level caching to skip or reuse expensive computation steps.

## Requirements

* **Python:** 3.10+ (Recommended)
* **OS:** Linux (recommended for large-scale runs), macOS, or Windows.
* **Compute:**
  * CPU-only runs are fully supported.
  * GPU is optional (recommended for deep models, transformers or dense retrievers).
* **Storage/RAM:**
  * Plan accordingly when using large datasets (e.g., MovieLens 25M, Amazon Reviews).
  * Ensure sufficient disk space if `n_runs > 1` is configured.

## Installation

### Core Dependencies

To install the fundamental requirements for the framework:

**Bash**

```
pip install -U pip
pip install -r requirements.txt
```

## Configs

BRIDGE is driven by a hierarchical configuration system. You run experiments by passing a **General Experiment Config** (JSON) to the CLI, which resolves specific **Dataset** and **Model** configurations.

### 1. General Experiment Config

This is the entry point for any execution. It defines the task type, the components to use, and execution controls.

**Example `experiment.json`:**

**JSON**

```json
{
  "experiment_name": "BM25_Baseline",
  "task_type": "search",
  "output_dir": "artifacts",

  "dataset": { "name": "amazoneletronics" },

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
    "metrics": ["NDCG", "RECALL", "MAP", "PRECISION"]
  }
}
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

The pipeline consists of five stages: `preprocess` → `features` → `train` → `index` → `eval`. You can control these via the `execution` block:

* **`skip`**: Stages to reuse from cache/checkpoints.
* **`force`**: Stages that must re-run, overwriting previous artifacts.
* **`n_runs`**: Number of times to repeat the pipeline (useful for statistical significance).

### 4. Hybrid (Bridging) Mode in Recs

BRIDGE also supports **hybrid recommender executions** using bridging strategies.
To enable this, add the following fields **right below** `task_type` in your experiment config:

* **`hybrid`**: `true` or `false` (default is `false` if omitted).
* **`hybridStrategie`**: only used if `hybrid=true`. Options:

  * `"query-as-user"`
  * `"retrieval-as-user"`

When `hybrid=true`, the run behaves like a normal Recs execution (same stages, same evaluation block, same outputs), but the pipeline switches to the **hybrid data construction / loading** flow according to the selected strategy.

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

### Running a Hybrid Recommendation Experiment (Query-as-User / Retrieval-as-User)

Below is a minimal example of a **hybrid Recs** config.
It adds `hybrid` and `hybridStrategie` (everything else remains a normal execution).

**Example `recs_hybrid_example.json`:**

```json
{
  "experiment_name": "Hybrid_Recs_Baseline",
  "task_type": "recs",

  "hybrid": true,
  "hybridStrategie": "query-as-user",

  "output_dir": "artifacts",

  "dataset": { "name": "movielens" },

  "model": [
    { "name": "LightFMModel" }
  ],

  "execution": {
    "skip": [],
    "force": [],
    "no_save": [],
    "n_runs": 1
  },

  "evaluation": {
    "top_ks": [5, 10, 20, 50],
    "metrics": ["NDCG", "RECALL", "MAP", "PRECISION", "MRR"]
  }
}
```

To switch strategies, change only:

```json
"hybridStrategie": "retrieval-as-user"
```

Run it normally:

```bash
python framework.py --config recs_hybrid_example.json
```

### Running a Hybrid Search Experiment (Centroid Vector / Tag Query)

#### Dataset Configuration for Hybrid Mode

To enable hybrid functionality at the data level, ensure your dataset configuration uses the "hybrid" mode. 

```json
{
  "dataloader": {
    "dataset_name": "movielens_hybrid",
    "path": "./data/ml-25m",
    "mode": "hybrid",
    "test_size": 0.2,
    "val_size": 0.1,
    "seed": 45
  },
  "features": {
    "query_col": "search_query",
    "doc_col": "document",
    "cat_col": "category",
    "id_col": "document_id"
  }
}
```

#### Centroid Vector

To use the Centroid Vector strategy, you must explicitly set the task to hybrid within your model's specific configuration file `bi_encoder.json`.

```json
  {
    "model": {
      "model_name": "BiEncoderSearchModel",
      "name": "BiEncoderSearchModel",
      "parameters": {
        "task": "hybrid",
        "pretrained_name": "all-MiniLM-L6-v2",
        "token_max_length": 128,
        "batch_size": 96,
        "learning_rate": 1e-7,
        "epochs": 3,
        "annoy_n_trees": 10,
        "model_dir": "./output/bi_encoder"
      }
    }
  }
```
#### Tag Query

Similarly, to utilize the Tag Query strategy, you must update the specific model configuration file `bm25_search.json` by setting the task to hybrid. This ensures the BM25 retriever correctly processes the expanded query signals derived from tags.

```json
{
  "model": {
    "model_name": "BM25Model",
    "name": "BM25Model",
    "parameters": {
      "task": "hybrid",
      "index_name": "bm25_index",
      "model": "bm25",
      "query_col": "search_query",
      "doc_col": "document",
      "doc_id_col": "document_id",
      "hyperparams": { "b": 0.75, "k1": 1.5 },
      "dataset_path": "./data/ml-25m",
      "tagfusion_min_relevance": 0.9
    }
  }
}

```

### Outputs

All results are generated in the `experimental_results/`. The directory structure is organized by:

