# Ranking Improved Self-Consistency

This repository contains the code for **Ranking Improved Self-Consistency (RISC)**.

The repository is based on the paper **“Boosting Self-Consistency with Ranking”**, accepted to **ACL SRW 2026**.

## Main result

<p align="center">
  <img src="Images/popqa_vertical_two_panel_fin_risc.png" alt="Accuracy versus the number of sampled responses on PopQA for self-consistency and RISC" width="100%">
</p>

**Figure.** Accuracy versus the number of sampled responses on PopQA for self-consistency and RISC. RISC consistently achieves higher accuracy while substantially reducing computational cost: with only 18 samples, it already surpasses the performance of self-consistency with 99 samples. It also delivers systematic accuracy gains over self-consistency across the full range of LLM-call budgets.

## Repository structure

- `SC_all_datasets_checkpoints_optimised_refine_fast.py`  
  Core feature engineering, embedding cache, and ranking utilities.

- `run_feature_sets_fast.py`  
  Dataset loading and dataset config source of truth.

- `reranker_configs.py`  
  Shared reranker configs:
  - dataset cache paths
  - feature sets
  - default LightGBM parameters
  - LightGBM hyperparameter grid
  - `make_feature_cfg(...)`

- `export_fixed5_feature_tables_one_split.py`  
  Export one split at a time:
  - `--split train` builds a multibudget train feature table
  - `--split test` builds a test feature table budget-by-budget

- `run_feature_sets_fast_custom_feature_paths.py`  
  Train and evaluate a reranker from already prepared feature pickles.

- `ablation_rerankers_fast_no_search_updated_paths_two_ablation.py`  
  Full model, leave-one-feature-out, and optional leave-two-features-out ablations.

## Features

The main feature set uses the following five signals.

### 1. Vote concentration: `share_ratio_to_best`

`share_ratio_to_best(a) = c(a) / c(a*)`

where:
- `c(a)` is the number of sampled chains that end with answer `a`
- `a*` is the most frequent answer for the current question

This measures how close the candidate answer's support is to the strongest vote winner.

### 2. Minimum answer length: `ans_len_min`

`ans_len_min(a) = min_r len(r)`

where the minimum is taken over all sampled traces `r` that produce answer `a`.

This captures the shortest formulation observed for the candidate answer.

### 3. Distance to answer centroid: `ans_dist2_to_id_centroid`

`ans_dist2_to_id_centroid(a) = ||e_a - mu_id||^2`

where:
- `e_a` is the embedding of candidate answer `a`
- `mu_id` is the centroid of all sampled answer embeddings for the same question

Lower values mean the candidate answer lies closer to the overall semantic center of the sampled responses.

### 4. Step-to-chain centroid coherence: `step_to_chain_centroid_min`

`step_to_chain_centroid_min = min_t cos(s_t, mu_chain)`

where:
- `s_t` is the embedding of reasoning step `t`
- `mu_chain` is the centroid of all step embeddings in the chain

This feature measures the weakest local coherence of any reasoning step with respect to the overall chain semantics.

### 5. Shared checkpoints across traces: `shared_checkpoint_count`

`shared_checkpoint_count = sum_i 1[p_i is shared]`

where:
- `p_i` is a prefix checkpoint
- `1[...]` is an indicator that equals 1 when the checkpoint is supported by semantically aligned prefixes from other traces

A checkpoint is counted as shared when it matches prefixes from other traces with:
- cosine similarity above a threshold
- depth difference within a tolerance
- enough relative support across distinct traces

This counts how many semantically aligned intermediate reasoning checkpoints are shared across independent sampled traces.

## Feature export and caching

Feature export creates `.pkl` files with features and a metadata JSON.

Embedding caches are loaded from:
- `--cache-path` if provided
- otherwise the dataset default cache path in `run_feature_sets_fast.py`

If the cache file does not exist, the script starts from an empty cache and can save it at the end.

### Train export

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

With explicit test budgets:

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

## Reranker training and evaluation

This step uses already prepared feature pickles.

Inputs:
- `--train-features-path`
- `--test-features-path`
- `--prepared-metadata-path` → should point to the **train metadata JSON**

What happens inside:
- loads prepared train/test features
- optionally runs LightGBM hyperparameter search on a train/validation split of the prepared train table
- refits on the full prepared train table
- scores the test table budget-by-budget
- saves reranker, scores, best params, and metadata

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

## Feature ablations

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
- `--test-budgets` is supported in export, training, and ablation scripts. If provided, it overrides range-based budget generation.
- For downstream training scripts, the metadata path should point to the **train metadata JSON**, not the test metadata JSON.
