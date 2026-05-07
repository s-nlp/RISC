#!/usr/bin/env python3
import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from SC_all_datasets_checkpoints_optimised_refine_fast import (
    EmbeddingCaches,
    RankerPreparedData,
    fit_lgbm_ranker_on_prepared_data,
    save_scores_list,
)
from reranker_configs import (
    DATASET_CONFIGS,
    FEATURE_SET_CONFIGS,
    DEFAULT_LGB_PARAMS,
    PARAM_GRID,
    make_feature_cfg,
)


def ensure_dirs(*paths: str | Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _jsonable_scalar(v: Any) -> bool:
    return isinstance(v, (int, float, str, bool)) or v is None


def sanitize_lgbm_params(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if _jsonable_scalar(v)}


def load_prepared_train_from_explicit_paths(train_features_path: str, prepared_metadata_path: str) -> RankerPreparedData:
    df_train_feat = pd.read_pickle(train_features_path)
    meta = json.loads(Path(prepared_metadata_path).read_text(encoding="utf-8"))
    return RankerPreparedData(
        df_feat=df_train_feat,
        feature_cols=list(meta["prepared_feature_cols"]),
        max_budget=int(meta["max_budget"]),
        qid_col=str(meta["prepared_qid_col"]),
        orig_group_col=str(meta["prepared_orig_group_col"]),
        target_col=str(meta["prepared_target_col"]),
    )


def load_test_features_from_explicit_path(test_features_path: str) -> pd.DataFrame:
    return pd.read_pickle(test_features_path)


def iter_param_grid(param_grid: Dict[str, Sequence[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(param_grid.keys())
    values = [list(param_grid[k]) for k in keys]
    for combo in product(*values):
        yield dict(zip(keys, combo))


def split_prepared_data(prepared: RankerPreparedData, val_size: float, random_state: int) -> Tuple[RankerPreparedData, RankerPreparedData]:
    df_feat = prepared.df_feat.copy()
    orig_ids = pd.Index(df_feat[prepared.orig_group_col].drop_duplicates())
    if len(orig_ids) < 2:
        raise ValueError("Need at least 2 original groups to run train/validation split.")
    n_val = max(1, int(round(len(orig_ids) * float(val_size))))
    n_val = min(n_val, len(orig_ids) - 1)
    rng = np.random.default_rng(int(random_state))
    shuffled = orig_ids.to_numpy(copy=True)
    rng.shuffle(shuffled)
    val_orig_ids = set(shuffled[:n_val].tolist())
    train_mask = ~df_feat[prepared.orig_group_col].isin(val_orig_ids)
    val_mask = df_feat[prepared.orig_group_col].isin(val_orig_ids)
    df_train = df_feat.loc[train_mask].copy()
    df_val = df_feat.loc[val_mask].copy()
    if df_train.empty or df_val.empty:
        raise ValueError("Train/validation split produced an empty partition.")
    train_prepared = RankerPreparedData(df_feat=df_train, feature_cols=list(prepared.feature_cols), max_budget=int(prepared.max_budget), qid_col=str(prepared.qid_col), orig_group_col=str(prepared.orig_group_col), target_col=str(prepared.target_col))
    val_prepared = RankerPreparedData(df_feat=df_val, feature_cols=list(prepared.feature_cols), max_budget=int(prepared.max_budget), qid_col=str(prepared.qid_col), orig_group_col=str(prepared.orig_group_col), target_col=str(prepared.target_col))
    return train_prepared, val_prepared


def _sorted_group_frame(df: pd.DataFrame, qid_col: str) -> pd.DataFrame:
    return df.sort_values([qid_col], kind="stable").reset_index(drop=True)


def prepared_to_arrays(prepared: RankerPreparedData, *, feature_cols: Sequence[str] | None = None):
    cols = list(prepared.feature_cols if feature_cols is None else feature_cols)
    df = _sorted_group_frame(prepared.df_feat, prepared.qid_col)
    X = df[cols].to_numpy(dtype=float)
    y = df[prepared.target_col].to_numpy(dtype=float)
    group = df.groupby(prepared.qid_col, sort=False).size().to_numpy(dtype=int)
    return X, y, group, df


def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    rel = np.asarray(relevances, dtype=float)[:k]
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2, dtype=float))
    gains = np.power(2.0, rel) - 1.0
    return float(np.sum(gains * discounts))


def mean_group_ndcg(df_scored: pd.DataFrame, *, qid_col: str, target_col: str, score_col: str, k: int) -> float:
    ndcgs: List[float] = []
    for _, grp in df_scored.groupby(qid_col, sort=False):
        rel_true = grp[target_col].to_numpy(dtype=float)
        rel_pred_order = grp.sort_values(score_col, ascending=False, kind="stable")[target_col].to_numpy(dtype=float)
        ideal_order = np.sort(rel_true)[::-1]
        ideal = dcg_at_k(ideal_order, k)
        if ideal <= 0.0:
            ndcgs.append(0.0)
            continue
        ndcgs.append(dcg_at_k(rel_pred_order, k) / ideal)
    return float(np.mean(ndcgs)) if ndcgs else float("nan")


def fit_ranker_direct(prepared: RankerPreparedData, lgbm_params: Dict[str, Any]) -> lgb.LGBMRanker:
    X_train, y_train, group_train, _ = prepared_to_arrays(prepared)
    model = lgb.LGBMRanker(**sanitize_lgbm_params(lgbm_params))
    model.fit(X_train, y_train, group=group_train)
    return model


def run_hparam_search_on_prepared_data(prepared: RankerPreparedData, base_lgbm_params: Dict[str, Any], param_grid: Dict[str, Sequence[Any]], *, val_size: float, random_state: int, ndcg_at: int):
    train_prepared, val_prepared = split_prepared_data(prepared=prepared, val_size=val_size, random_state=random_state)
    X_val, _, _, df_val_sorted = prepared_to_arrays(val_prepared)
    rows: List[Dict[str, Any]] = []
    best_score = float("-inf")
    best_params: Dict[str, Any] = dict(base_lgbm_params)
    for trial_idx, trial_overrides in enumerate(iter_param_grid(param_grid), start=1):
        trial_params = dict(base_lgbm_params)
        trial_params.update(trial_overrides)
        trial_params["random_state"] = int(base_lgbm_params.get("random_state", random_state))
        model = fit_ranker_direct(train_prepared, trial_params)
        val_scores = model.predict(X_val)
        df_eval = df_val_sorted[[val_prepared.qid_col, val_prepared.target_col]].copy()
        df_eval["score"] = val_scores
        val_mean_ndcg = mean_group_ndcg(df_eval, qid_col=val_prepared.qid_col, target_col=val_prepared.target_col, score_col="score", k=int(ndcg_at))
        row = {"trial": int(trial_idx), "val_mean_ndcg": float(val_mean_ndcg)}
        row.update({k: trial_params[k] for k in sorted(trial_params.keys()) if _jsonable_scalar(trial_params[k])})
        rows.append(row)
        print(f"trial={trial_idx:03d} val_mean_ndcg@{int(ndcg_at)}={val_mean_ndcg:.6f} params={json.dumps(sanitize_lgbm_params(trial_overrides), ensure_ascii=False)}")
        if val_mean_ndcg > best_score:
            best_score = float(val_mean_ndcg)
            best_params = dict(trial_params)
    search_results = pd.DataFrame(rows).sort_values(["val_mean_ndcg", "trial"], ascending=[False, True]).reset_index(drop=True)
    return search_results, sanitize_lgbm_params(best_params)


def score_precomputed_test_features(bundle, df_test_feat: pd.DataFrame, budgets: Sequence[int]) -> List[float]:
    out: List[float] = []
    for b in budgets:
        sub = df_test_feat[df_test_feat["budget_n"] == int(b)].copy()
        if len(sub) == 0:
            out.append(float("nan"))
            continue
        sub["budget_n_norm"] = float(b) / float(bundle.max_budget)
        X = sub[bundle.feature_cols].to_numpy(dtype=float)
        sub["score"] = bundle.model.predict(X)
        pred_idx = sub.groupby("id", sort=False)["score"].idxmax()
        pred_rows = sub.loc[pred_idx]
        out.append(float(pred_rows["hit_mean"].mean()))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIGS.keys()))
    parser.add_argument("--feature-set", required=True, choices=sorted(FEATURE_SET_CONFIGS.keys()))
    parser.add_argument("--train-features-path", required=True, type=str)
    parser.add_argument("--test-features-path", required=True, type=str)
    parser.add_argument("--prepared-metadata-path", required=True, type=str)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 5, 10, 20, 40, 60, 80, 100])
    parser.add_argument("--test-budgets", type=int, nargs="+", default=None, help="Optional explicit test budgets. Overrides eval-min/eval-max.")
    parser.add_argument("--eval-min", type=int, default=1)
    parser.add_argument("--eval-max", type=int, default=100, help="Inclusive upper bound for test budgets if --test-budgets is omitted.")
    parser.add_argument("--reranker-dir", default="rerankers")
    parser.add_argument("--scores-dir", default="reranker_scores")
    parser.add_argument("--metadata-dir", default="reranker_metadata")
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--no-save-cache", action="store_true")
    parser.add_argument("--no-hparam-search", action="store_true")
    parser.add_argument("--adaptive-prefix-thresholds", action="store_true")
    parser.add_argument("--prefix-sim-threshold", type=float, default=None)
    parser.add_argument("--prefix-rel-support-thr", type=float, default=None)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--ndcg-at", type=int, default=10)
    args = parser.parse_args()

    dataset = args.dataset
    feature_set = args.feature_set
    ds_cfg = DATASET_CONFIGS[dataset]
    budgets_train = sorted(set(map(int, args.budgets)))
    budgets_test = sorted(set(map(int, args.test_budgets))) if args.test_budgets else list(range(int(args.eval_min), int(args.eval_max) + 1))

    reranker_dir = Path(args.reranker_dir)
    scores_dir = Path(args.scores_dir)
    metadata_dir = Path(args.metadata_dir)
    ensure_dirs(reranker_dir, scores_dir, metadata_dir)

    reranker_path = reranker_dir / f"{dataset}_{feature_set}.joblib"
    scores_path = scores_dir / f"{dataset}_{feature_set}.json"
    search_results_path = metadata_dir / f"{dataset}_{feature_set}_search_results.csv"
    best_params_path = metadata_dir / f"{dataset}_{feature_set}_best_params.json"
    metadata_path = metadata_dir / f"{dataset}_{feature_set}.json"

    feature_cfg = make_feature_cfg(
        feature_set_name=feature_set,
        device=args.device,
        batch_size=args.batch_size,
        prefix_sim_threshold=args.prefix_sim_threshold,
        prefix_rel_support_thr=args.prefix_rel_support_thr,
        adaptive_prefix_thresholds=args.adaptive_prefix_thresholds,
    )

    cache_path = Path(args.cache_path) if args.cache_path else Path(ds_cfg["cache_path"])
    if cache_path.exists():
        print(f"Loading caches from {cache_path}")
        caches = EmbeddingCaches.load(str(cache_path))
    else:
        print(f"No cache found at {cache_path}; starting with empty caches")
        caches = EmbeddingCaches()

    print("=" * 80)
    print(f"Dataset               : {dataset}")
    print(f"Feature set           : {feature_set}")
    print(f"Train features path   : {args.train_features_path}")
    print(f"Test features path    : {args.test_features_path}")
    print(f"Prepared metadata path: {args.prepared_metadata_path}")
    print(f"Train budgets         : {budgets_train}")
    print(f"Test budgets          : {budgets_test[:3]} ... {budgets_test[-3:] if budgets_test else []}")
    print(f"Reranker path         : {reranker_path}")
    print(f"Scores path           : {scores_path}")
    print(f"Best params path      : {best_params_path}")
    print("=" * 80)

    prepared_full = load_prepared_train_from_explicit_paths(args.train_features_path, args.prepared_metadata_path)
    df_test_feat = load_test_features_from_explicit_path(args.test_features_path)

    print(f"Prepared train feature shape: {prepared_full.df_feat.shape}")
    print(f"Prepared test feature shape : {df_test_feat.shape}")

    if args.no_hparam_search:
        search_results = pd.DataFrame()
        best_params = sanitize_lgbm_params(DEFAULT_LGB_PARAMS)
        print("Hyperparameter search skipped; using DEFAULT_LGB_PARAMS")
    else:
        search_results, best_params = run_hparam_search_on_prepared_data(
            prepared=prepared_full,
            base_lgbm_params=DEFAULT_LGB_PARAMS,
            param_grid=PARAM_GRID,
            val_size=args.val_size,
            random_state=args.random_state,
            ndcg_at=args.ndcg_at,
        )
        if not search_results.empty:
            search_results.to_csv(search_results_path, index=False)
            print(f"Saved search results to {search_results_path}")

    print("Refitting final reranker on full prepared training data with best params...")
    bundle, importances = fit_lgbm_ranker_on_prepared_data(
        prepared=prepared_full,
        feature_cfg=feature_cfg,
        lgbm_ranker_ctor=lgb.LGBMRanker,
        lgbm_params=best_params,
    )

    print("Scoring reranker on precomputed test features...")
    scores = score_precomputed_test_features(bundle, df_test_feat, budgets_test)

    save_scores_list(scores, str(scores_path))
    bundle.save(str(reranker_path))
    save_json(best_params_path, best_params)

    if cache_path and not args.no_save_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        caches.save(str(cache_path))
        print(f"Updated caches saved to {cache_path}")

    metadata = {
        "dataset": dataset,
        "feature_set": feature_set,
        "cache_path": str(cache_path),
        "reranker_path": str(reranker_path),
        "scores_path": str(scores_path),
        "best_params_path": str(best_params_path),
        "search_results_path": str(search_results_path),
        "loaded_from_train_features_path": args.train_features_path,
        "loaded_from_test_features_path": args.test_features_path,
        "loaded_from_prepared_metadata_path": args.prepared_metadata_path,
        "feature_config": FEATURE_SET_CONFIGS[feature_set],
        "base_lgbm_params": DEFAULT_LGB_PARAMS,
        "best_lgbm_params": best_params,
        "param_grid": PARAM_GRID,
        "no_hparam_search": bool(args.no_hparam_search),
        "adaptive_prefix_thresholds": bool(args.adaptive_prefix_thresholds),
        "manual_prefix_sim_threshold": args.prefix_sim_threshold,
        "manual_prefix_rel_support_thr": args.prefix_rel_support_thr,
        "budgets_train": budgets_train,
        "test_budgets": budgets_test,
        "val_size": float(args.val_size),
        "random_state": int(args.random_state),
        "ndcg_at": int(args.ndcg_at),
        "prepared_train_shape": list(prepared_full.df_feat.shape),
        "prepared_test_shape": list(df_test_feat.shape),
        "prepared_feature_cols": list(prepared_full.feature_cols),
        "prepared_qid_col": str(prepared_full.qid_col),
        "prepared_orig_group_col": str(prepared_full.orig_group_col),
        "prepared_target_col": str(prepared_full.target_col),
        "refit_on_full_prepared_train_after_search": True,
        "mean_score": float(pd.Series(scores, dtype=float).mean()),
        "notes": [
            "This script requires explicit train/test pickle paths and does not use features-dir fallback.",
            "Hyperparameter search runs on a group-based train/validation split of the prepared train feature table.",
            "After search, the final reranker is refit once on the full prepared training data using best_lgbm_params.",
            "Scoring uses the explicitly provided test feature pickle.",
            "If --test-budgets is passed, it overrides eval-min/eval-max.",
        ],
    }

    save_json(metadata_path, metadata)

    print(f"Saved reranker to    {reranker_path}")
    print(f"Saved scores to      {scores_path}")
    print(f"Saved best params to {best_params_path}")
    print(f"Saved metadata to    {metadata_path}")


if __name__ == "__main__":
    main()
