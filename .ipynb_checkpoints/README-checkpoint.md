# Ranking Improved Self-Consistency (RISC)

This repository contains code for **Ranking Improved Self-Consistency (RISC)**.

The implementation is based on the paper **“Boosting Self-Consistency with Ranking”**, accepted to **ACL SRW 2026**.

## Main result

```latex
\begin{figure*}[t!]
    \centering
    \includegraphics[trim=.25cm .35cm .5cm .25cm, clip, width=\linewidth]{Images/3datasets_compare_stars.pdf}
    \caption{Comparison of RISC against the Self-Consistency, Stable Rank, ReASC, and CISC on three datasets. RISC consistently outperforms the baselines on the QA datasets across all LLM call budgets, while remaining competitive on MATH500.
    }
    \label{fig:baselines_comparison}
    \vspace{-0.5cm}
\end{figure*}
```



## Repository structure

- `SC_all_datasets_checkpoints_optimised_refine_fast.py`  
  Core feature engineering, embedding cache, and ranking utilities.

- `run_feature_sets_fast.py`  
  Dataset loading and dataset config source of truth.

- `reranker_configs.py`  
  Shared reranker configs: dataset cache paths, feature sets, default LightGBM parameters, parameter grid, and `make_feature_cfg(...)`.

- `export_fixed5_feature_tables_one_split.py`  
  Export one split at a time:
  - `--split train` builds a multibudget train feature table
  - `--split test` builds a test feature table budget-by-budget

- `run_feature_sets_fast_custom_feature_paths.py`  
  Train and evaluate a reranker from already prepared feature pickles.

- `ablation_rerankers_fast_no_search_updated_paths_two_ablation.py`  
  Full model, leave-one-feature-out, and optional leave-two-features-out ablations.

## Features

The main feature set uses five signals.

### Vote concentration: `share_ratio_to_best`

Let \( c(a) \) be the number of sampled chains that end with answer \( a \), and let \( a^\star \) be the most frequent answer for the current question. Then

\[
\texttt{share\_ratio\_to\_best}(a) = \frac{c(a)}{c(a^\star)}.
\]

This measures how close the candidate answer’s support is to the strongest vote winner.

### Minimum answer length: `ans_len_min`

For all sampled traces that produce answer \( a \), let \( \ell(r) \) denote the answer length of trace \( r \). Then

\[
\texttt{ans\_len\_min}(a) = \min_{r : \mathrm{ans}(r)=a} \ell(r).
\]

This captures the shortest formulation observed for the candidate answer.

### Distance to answer centroid: `ans_dist2_to_id_centroid`

Let \( e_r \in \mathbb{R}^d \) be the embedding of a sampled answer trace \( r \) for the same question, and let

\[
\mu_{\text{id}} = \frac{1}{N}\sum_{r=1}^{N} e_r
\]

be the centroid over all sampled traces for that question. For candidate answer \( a \) with embedding \( e_a \),

\[
\texttt{ans\_dist2\_to\_id\_centroid}(a) = \lVert e_a - \mu_{\text{id}} \rVert_2^2.
\]

Lower values indicate that the candidate answer lies closer to the overall semantic center of the sampled responses.

### Step-to-chain centroid coherence: `step_to_chain_centroid_min`

Let a chain contain step embeddings \( s_1, \dots, s_T \), and let

\[
\mu_{\text{chain}} = \frac{1}{T}\sum_{t=1}^{T} s_t
\]

be the centroid of the chain. Then

\[
\texttt{step\_to\_chain\_centroid\_min} = \min_{t \in \{1,\dots,T\}} \cos(s_t, \mu_{\text{chain}}).
\]

This feature measures the weakest local coherence of a reasoning step with respect to the overall chain semantics.

### Shared checkpoints across traces: `shared_checkpoint_count`

For each prefix point \( p_i \) in a trace, define a checkpoint if it matches prefixes from other traces with:
- cosine similarity above a threshold,
- depth difference within a tolerance,
- and enough relative support across distinct traces.

If \( \mathbb{1}[p_i \text{ is shared}] \) indicates that prefix \( p_i \) is a supported checkpoint, then

\[
\texttt{shared\_checkpoint\_count} = \sum_i \mathbb{1}[p_i \text{ is shared}].
\]

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
- `--prepared-metadata-path` — this should point to the **train metadata JSON**

What happens inside:
- loads prepared train/test features,
- optionally runs LightGBM hyperparameter search on a train/validation split of the prepared train table,
- refits on the full prepared train table,
- scores the test table budget-by-budget,
- saves reranker, scores, best params, and metadata.

### Hyperparameter search

By default, hyperparameter search is enabled and uses `DEFAULT_LGB_PARAMS` and `PARAM_GRID` from `reranker_configs.py`.

To skip search and use defaults directly, add `--no-hparam-search`.

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
- the full model,
- leave-one-feature-out variants,
- optional leave-two-features-out variants via `--include-two-feature-ablation`.

It can load explicit feature-table paths, reuse signature-matched cached feature tables, or recompute them if allowed.

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
