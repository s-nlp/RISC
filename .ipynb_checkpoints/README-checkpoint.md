# Reranker pipeline

This repository contains a three-step pipeline:

1. export cached train/test feature tables,
2. train and evaluate a LightGBM reranker from cached features,
3. run feature ablations from cached features.

The current workflow uses five core signals:

- `share_ratio_to_best`
- `ans_len_min`
- `ans_dist2_to_id_centroid`
- `step_to_chain_centroid_min`
- `shared_checkpoint_count`

## Repository files

- `SC_all_datasets_checkpoints_optimised_refine_fast.py`  
  Core feature engineering, embedding cache, and ranking utilities.

- `run_feature_sets_fast.py`  
  Dataset loading and dataset config source of truth.

- `reranker_configs.py`  
  Shared reranker configs:
  - dataset cache paths
  - feature sets
  - default LightGBM params
  - LightGBM parameter grid
  - `make_feature_cfg(...)`

- `export_fixed5_feature_tables_one_split.py`  
  Export one split at a time:
  - `--split train` → multibudget train feature table
  - `--split test` → test feature table budget-by-budget

- `run_feature_sets_fast_custom_feature_paths.py`  
  Train / evaluate a reranker from explicit train/test feature pickles.

- `ablation_rerankers_fast_no_search_updated_paths_two_ablation.py`  
  Full model + single-feature-drop + optional two-feature-drop ablations from cached feature tables.

## 1) Feature export and caching

This step creates `.pkl` files with features and metadata.

### Caches
Embedding caches are loaded from:
- `--cache-path` if provided
- otherwise the dataset default cache path in `run_feature_sets_fast.py`

If the cache file does not exist, the script starts from an empty cache and can save it at the end.

### Train export
Train export requires:
- `--dataset`
- `--split train`
- `--train-budgets`

Example:

```bash
python export_fixed5_feature_tables_one_split.py \
  --dataset hotpotqa \
  --split train \
  --train-csv-path "/path/to/hotpotqa_train.csv" \
  --train-budgets 2 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 \
  --output-path reranker_feature_tables/hotpotqa/train_features.pkl \
  --cache-path caches_olmo/hotpotqa_embedding_caches.joblib \
  --prefix-sim-threshold 0.75 \
  --prefix-rel-support-thr 0.5 \
  --metadata-output-path reranker_feature_tables/hotpotqa/train_metadata.json
```

### Test export
Test export supports either:
- explicit `--test-budgets`
- or range form `--eval-min` / `--eval-max`

Example with explicit budgets:

```bash
python export_fixed5_feature_tables_one_split.py \
  --dataset hotpotqa \
  --split test \
  --test-csv-path "/path/to/hotpotqa_test.csv" \
  --test-budgets 1 2 3 4 5 10 20 40 60 80 100 \
  --output-path reranker_feature_tables/hotpotqa/test_features.pkl \
  --cache-path caches_olmo/hotpotqa_embedding_caches.joblib \
  --prefix-sim-threshold 0.75 \
  --prefix-rel-support-thr 0.5 \
  --metadata-output-path reranker_feature_tables/hotpotqa/test_metadata.json
```

## 2) Train and evaluate reranker

This stage uses already prepared feature pickles.

Inputs:
- `--train-features-path`
- `--test-features-path`
- `--prepared-metadata-path` → should point to the **train metadata JSON**

What happens inside:
- loads prepared train/test features
- optionally runs LightGBM hyperparameter search on a train/validation split of the prepared train table
- refits on the full prepared train table
- scores the test table budget-by-budget
- saves reranker, scores, best params, metadata

### Hyperparameter search
By default, hyperparameter search is enabled and uses:
- `DEFAULT_LGB_PARAMS`
- `PARAM_GRID`
from `reranker_configs.py`

To skip search and use defaults directly, add:
- `--no-hparam-search`

Example:

```bash
python run_feature_sets_fast_custom_feature_paths.py \
  --dataset popqa \
  --feature-set set7_hyp_search_adaptive \
  --train-features-path reranker_feature_tables/popqa/train_features.pkl \
  --test-features-path reranker_feature_tables/popqa/test_features.pkl \
  --prepared-metadata-path reranker_feature_tables/popqa/train_metadata.json \
  --device cuda \
  --batch-size 1024 \
  --budgets 2 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 \
  --test-budgets 1 2 3 4 5 10 20 40 60 80 100 \
  --reranker-dir rerankers \
  --scores-dir reranker_scores \
  --metadata-dir reranker_metadata \
  --prefix-sim-threshold 0.75 \
  --prefix-rel-support-thr 0.5
```

Example without search:

```bash
python run_feature_sets_fast_custom_feature_paths.py \
  --dataset popqa \
  --feature-set set7_hyp_search_adaptive \
  --train-features-path reranker_feature_tables/popqa/train_features.pkl \
  --test-features-path reranker_feature_tables/popqa/test_features.pkl \
  --prepared-metadata-path reranker_feature_tables/popqa/train_metadata.json \
  --device cuda \
  --batch-size 1024 \
  --budgets 2 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 \
  --test-budgets 1 2 3 4 5 10 20 40 60 80 100 \
  --reranker-dir rerankers \
  --scores-dir reranker_scores \
  --metadata-dir reranker_metadata \
  --prefix-sim-threshold 0.75 \
  --prefix-rel-support-thr 0.5 \
  --no-hparam-search
```

## 3) Feature ablations

The ablation script supports:
- full model
- leave-one-feature-out variants
- optional leave-two-features-out variants via `--include-two-feature-ablation`

It can:
- load explicit feature-table paths
- reuse signature-matched cached feature tables
- or recompute them if allowed

Example:

```bash
python ablation_rerankers_fast_no_search_updated_paths_two_ablation.py \
  --dataset math500 \
  --feature-set set7_hyp_search_adaptive \
  --train-features-path reranker_feature_tables/math500/train_features.pkl \
  --test-features-path reranker_feature_tables/math500/test_features.pkl \
  --prepared-metadata-path reranker_feature_tables/math500/train_metadata.json \
  --device cuda \
  --batch-size 1024 \
  --budgets 2 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 \
  --test-budgets 1 2 3 4 5 10 20 40 60 80 100 \
  --lgbm-params-path reranker_metadata/math500_set7_hyp_search_adaptive_best_params.json \
  --reranker-dir reranker_ablations/rerankers \
  --scores-dir reranker_ablations/reranker_scores \
  --metadata-dir reranker_ablations/reranker_metadata \
  --include-two-feature-ablation \
  --skip-feature-recalc
```

## Notes

- Train filtering in feature export is enabled through `filter_questions_for_reranker(...)`.
- `--test-budgets` is now supported in export, training, and ablation scripts. If provided, it overrides range-based test-budget generation.
- For downstream training scripts, the metadata path should point to the **train metadata JSON**, not the test metadata JSON.
