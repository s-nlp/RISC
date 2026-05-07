#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence
from itertools import combinations

import lightgbm as lgb
import pandas as pd

import run_feature_sets_fast as runner
from SC_all_datasets_checkpoints_optimised_refine_fast import (
    EmbeddingCaches,
    RankerPreparedData,
    build_features_for_topn,
    fit_lgbm_ranker_on_prepared_data,
    prepare_lgbm_reranker_multibudget_features,
    save_scores_list,
)
from reranker_configs import DATASET_CONFIGS, FEATURE_SET_CONFIGS, DEFAULT_LGB_PARAMS


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


def load_lgbm_params(path: str | None) -> Dict[str, Any]:
    if path is None:
        return dict(DEFAULT_LGB_PARAMS)
    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(loaded)}")
    for key in ["best_params", "best_lgbm_params", "lgbm_params"]:
        if key in loaded and isinstance(loaded[key], dict):
            return sanitize_lgbm_params(loaded[key])
    return sanitize_lgbm_params(loaded)


def _make_signature(dataset: str, feature_set: str, budgets_train: Sequence[int], budgets_test: Sequence[int], feature_cfg) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "feature_set": feature_set,
        "train_budgets": list(map(int, budgets_train)),
        "test_budgets": list(map(int, budgets_test)),
        "adaptive_prefix_thresholds": bool(getattr(feature_cfg, "adaptive_prefix_thresholds", False)),
        "prefix_sim_threshold": None if getattr(feature_cfg, "prefix_sim_threshold", None) is None else float(feature_cfg.prefix_sim_threshold),
        "prefix_rel_support_thr": None if getattr(feature_cfg, "prefix_rel_support_thr", None) is None else float(feature_cfg.prefix_rel_support_thr),
        "device": str(getattr(feature_cfg, "device", "")),
        "batch_size": int(getattr(feature_cfg, "batch_size", 0)),
        "handcrafted_cols": list(feature_cfg.handcrafted_cols),
        "embedding_cols": list(feature_cfg.resolved_embedding_cols()),
        "prefix_embedding_cols": list(feature_cfg.resolved_prefix_embedding_cols()),
        "test_feature_builder": "build_features_for_topn_loop",
    }


def maybe_load_feature_tables(base_path: Path, signature: Dict[str, Any]):
    train_path = base_path / "train_features.pkl"
    test_path = base_path / "test_features.pkl"
    meta_path = base_path / "feature_tables_metadata.json"
    if not (train_path.exists() and test_path.exists() and meta_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if meta.get("signature") != signature:
        return None
    df_train_feat = pd.read_pickle(train_path)
    df_test_feat = pd.read_pickle(test_path)
    return df_train_feat, df_test_feat, meta


def save_feature_tables(base_path: Path, prepared_train: RankerPreparedData, df_test_feat: pd.DataFrame, signature: Dict[str, Any]) -> None:
    base_path.mkdir(parents=True, exist_ok=True)
    train_path = base_path / "train_features.pkl"
    test_path = base_path / "test_features.pkl"
    meta_path = base_path / "feature_tables_metadata.json"
    prepared_train.df_feat.to_pickle(train_path)
    df_test_feat.to_pickle(test_path)
    save_json(meta_path, {
        "signature": signature,
        "train_features_path": str(train_path),
        "test_features_path": str(test_path),
        "prepared_feature_cols": list(prepared_train.feature_cols),
        "prepared_qid_col": prepared_train.qid_col,
        "prepared_orig_group_col": prepared_train.orig_group_col,
        "prepared_target_col": prepared_train.target_col,
        "max_budget": int(prepared_train.max_budget),
    })


def build_test_features_via_original_eval_builder(df_test: pd.DataFrame, budgets_test: Sequence[int], feature_cfg, caches: EmbeddingCaches) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for b in budgets_test:
        print(f"Preparing test features for budget={int(b)} via build_features_for_topn ...")
        df_b = build_features_for_topn(
            df_trace=df_test,
            top_n=int(b),
            feature_cfg=feature_cfg,
            embed_cache=caches.base_cache,
            prefix_embed_cache=caches.prefix_cache,
        )
        df_b = df_b.copy()
        df_b["budget_n"] = int(b)
        parts.append(df_b)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIGS.keys()))
    parser.add_argument("--feature-set", required=True, choices=sorted(FEATURE_SET_CONFIGS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--budgets", nargs="+", type=int, required=True)
    parser.add_argument("--test-budgets", nargs="+", type=int, default=None, help="Optional explicit test budgets. Overrides eval-min/eval-max.")
    parser.add_argument("--eval-min", type=int, default=1)
    parser.add_argument("--eval-max", type=int, default=100)
    parser.add_argument("--lgbm-params-path", type=str, default=None)
    parser.add_argument("--drop-feature", action="append", default=None)
    parser.add_argument("--reranker-dir", default="rerankers")
    parser.add_argument("--scores-dir", default="reranker_scores")
    parser.add_argument("--metadata-dir", default="reranker_metadata")
    parser.add_argument("--features-dir", default="reranker_feature_tables")
    parser.add_argument("--train-features-path", type=str, default=None)
    parser.add_argument("--test-features-path", type=str, default=None)
    parser.add_argument("--prepared-metadata-path", type=str, default=None)
    parser.add_argument("--skip-feature-recalc", action="store_true")
    parser.add_argument("--include-two-feature-ablation", action="store_true")
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--adaptive-prefix-thresholds", action="store_true")
    parser.add_argument("--prefix-sim-threshold", type=float, default=None)
    parser.add_argument("--prefix-rel-support-thr", type=float, default=None)
    parser.add_argument("--no-save-cache", action="store_true")
    args = parser.parse_args()

    ensure_dirs(args.reranker_dir, args.scores_dir, args.metadata_dir, args.features_dir)
    budgets_train = sorted(set(map(int, args.budgets)))
    budgets_test = sorted(set(map(int, args.test_budgets))) if args.test_budgets else list(range(int(args.eval_min), int(args.eval_max) + 1))

    feature_cfg = runner.make_feature_cfg(feature_set_name=args.feature_set, device=args.device, batch_size=args.batch_size)
    if args.adaptive_prefix_thresholds:
        feature_cfg.adaptive_prefix_thresholds = True
    if args.prefix_sim_threshold is not None:
        feature_cfg.prefix_sim_threshold = float(args.prefix_sim_threshold)
    if args.prefix_rel_support_thr is not None:
        feature_cfg.prefix_rel_support_thr = float(args.prefix_rel_support_thr)

    lgbm_params = load_lgbm_params(args.lgbm_params_path)
    ds_cfg = DATASET_CONFIGS[args.dataset]
    df_train_pruned, df_test, _ = runner.load_dataset(args.dataset)

    cache_path = args.cache_path or ds_cfg.get("cache_path")
    if cache_path and Path(cache_path).exists():
        print(f"Loading caches from {cache_path}")
        caches = EmbeddingCaches.load(str(cache_path))
    else:
        print("No cache found; starting with empty caches")
        caches = EmbeddingCaches()

    features_root = Path(args.features_dir) / f"{args.dataset}_{args.feature_set}"
    signature = _make_signature(args.dataset, args.feature_set, budgets_train, budgets_test, feature_cfg)

    prepared_full = None
    df_test_feat = None
    if args.train_features_path and args.test_features_path and args.prepared_metadata_path:
        print("Loading precomputed feature tables from explicit paths...")
        df_train_feat = pd.read_pickle(args.train_features_path)
        df_test_feat = pd.read_pickle(args.test_features_path)
        meta = json.loads(Path(args.prepared_metadata_path).read_text(encoding="utf-8"))
        prepared_full = RankerPreparedData(
            df_feat=df_train_feat,
            feature_cols=list(meta["prepared_feature_cols"]),
            max_budget=int(meta["max_budget"]),
            qid_col=str(meta["prepared_qid_col"]),
            orig_group_col=str(meta["prepared_orig_group_col"]),
            target_col=str(meta["prepared_target_col"]),
        )

    if prepared_full is None:
        loaded = maybe_load_feature_tables(features_root, signature)
        if loaded is not None:
            print(f"Reusing saved feature tables from {features_root}")
            df_train_feat, df_test_feat, meta = loaded
            prepared_full = RankerPreparedData(
                df_feat=df_train_feat,
                feature_cols=list(meta["prepared_feature_cols"]),
                max_budget=int(meta["max_budget"]),
                qid_col=str(meta["prepared_qid_col"]),
                orig_group_col=str(meta["prepared_orig_group_col"]),
                target_col=str(meta["prepared_target_col"]),
            )

    if prepared_full is None:
        if args.skip_feature_recalc:
            raise ValueError("No matching precomputed features found. Provide explicit paths or remove --skip-feature-recalc.")
        print("Preparing train features once using the exact fast training pipeline...")
        prepared_full = prepare_lgbm_reranker_multibudget_features(df_train_trace=df_train_pruned, budgets=budgets_train, feature_cfg=feature_cfg, caches=caches)
        print("Preparing test features once by looping budgets through build_features_for_topn...")
        df_test_feat = build_test_features_via_original_eval_builder(df_test=df_test, budgets_test=budgets_test, feature_cfg=feature_cfg, caches=caches)
        save_feature_tables(features_root, prepared_full, df_test_feat, signature)
        print(f"Saved feature tables to {features_root}")

    if cache_path and not args.no_save_cache:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        caches.save(str(cache_path))
        print(f"Updated caches saved to {cache_path}")

    base_feature_cols = [c for c in prepared_full.feature_cols if c != "budget_n_norm"]
    ablate_features = list(dict.fromkeys(args.drop_feature)) if args.drop_feature else list(base_feature_cols)
    unknown = [c for c in ablate_features if c not in base_feature_cols]
    if unknown:
        raise ValueError(f"Unknown ablation features: {unknown}. Available: {base_feature_cols}")

    summary_rows: List[Dict[str, Any]] = []
    variants = [("full", [])] + [(f"drop_{feat}", [feat]) for feat in ablate_features]
    if args.include_two_feature_ablation:
        variants += [(f"drop_{f1}__{f2}", [f1, f2]) for f1, f2 in combinations(ablate_features, 2)]

    for variant_name, dropped in variants:
        keep_cols = [c for c in base_feature_cols if c not in set(dropped)]
        prepared_variant = RankerPreparedData(
            df_feat=prepared_full.df_feat,
            feature_cols=keep_cols + ["budget_n_norm"],
            max_budget=prepared_full.max_budget,
            qid_col=prepared_full.qid_col,
            orig_group_col=prepared_full.orig_group_col,
            target_col=prepared_full.target_col,
        )
        print(f"Training variant={variant_name} ...")
        bundle, importances = fit_lgbm_ranker_on_prepared_data(prepared=prepared_variant, feature_cfg=feature_cfg, lgbm_ranker_ctor=lgb.LGBMRanker, lgbm_params=lgbm_params)
        print(f"Scoring variant={variant_name} ...")
        scores = score_precomputed_test_features(bundle, df_test_feat, budgets_test)
        mean_score = float(pd.Series(scores, dtype=float).mean())
        stem = f"{args.dataset}_{args.feature_set}" if variant_name == "full" else f"{args.dataset}_{args.feature_set}_{variant_name}"
        reranker_path = Path(args.reranker_dir) / f"{stem}.joblib"
        scores_path = Path(args.scores_dir) / f"{stem}.json"
        metadata_path = Path(args.metadata_dir) / f"{stem}.json"
        importances_path = Path(args.metadata_dir) / f"{stem}_importances.csv"
        bundle.save(reranker_path)
        save_scores_list(scores, scores_path)
        importances.to_csv(importances_path, index=False)
        save_json(metadata_path, {
            "dataset": args.dataset,
            "feature_set": args.feature_set,
            "variant": variant_name,
            "dropped_feature": dropped[0] if len(dropped) == 1 else "",
            "dropped_features": dropped,
            "kept_features": keep_cols,
            "all_base_features": base_feature_cols,
            "prepared_feature_cols_full": prepared_full.feature_cols,
            "train_budgets": budgets_train,
            "test_budgets": budgets_test,
            "max_budget": int(prepared_full.max_budget),
            "lgbm_params": lgbm_params,
            "adaptive_prefix_thresholds": bool(getattr(feature_cfg, "adaptive_prefix_thresholds", False)),
            "prefix_sim_threshold": None if getattr(feature_cfg, "prefix_sim_threshold", None) is None else float(feature_cfg.prefix_sim_threshold),
            "prefix_rel_support_thr": None if getattr(feature_cfg, "prefix_rel_support_thr", None) is None else float(feature_cfg.prefix_rel_support_thr),
            "features_dir": str(features_root),
            "train_features_path": str(features_root / "train_features.pkl"),
            "test_features_path": str(features_root / "test_features.pkl"),
            "input_train_features_path": args.train_features_path,
            "input_test_features_path": args.test_features_path,
            "input_prepared_metadata_path": args.prepared_metadata_path,
            "mean_score": mean_score,
        })
        summary_rows.append({
            "variant": variant_name,
            "dropped_feature": dropped[0] if len(dropped) == 1 else "",
            "dropped_features": dropped,
            "mean_score": mean_score,
            "reranker_path": str(reranker_path),
            "scores_path": str(scores_path),
            "metadata_path": str(metadata_path),
            "importances_path": str(importances_path),
        })
        print(f"Saved variant={variant_name} to {reranker_path}")

    summary_df = pd.DataFrame(summary_rows).sort_values(["mean_score", "variant"], ascending=[False, True]).reset_index(drop=True)
    summary_path = Path(args.metadata_dir) / f"{args.dataset}_{args.feature_set}_ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved ablation summary to {summary_path}")


if __name__ == "__main__":
    main()
