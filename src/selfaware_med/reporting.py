from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from .experiment import RunRecord
from .model import CallStats


def records_to_frame(records: list[RunRecord]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in records])


def save_per_method_csvs(df: pd.DataFrame, out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method].reset_index(drop=True)
        p = out_dir / f"{method}_results.csv"
        sub.to_csv(p, index=False)
        paths[method] = p
    return paths


def _spearman_safe(x, y) -> tuple[float | None, float | None]:
    if len(x) < 3:
        return None, None
    if x.nunique() < 2 or y.nunique() < 2:
        return None, None
    rho, p = stats.spearmanr(x, y)
    if pd.isna(rho) or pd.isna(p):
        return None, None
    return float(rho), float(p)


def _stop_reason_counts(sub: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in sub["stop_reason"].value_counts().items()}


def _consistency_group_accuracy(sub: pd.DataFrame) -> dict:
    valid = sub.dropna(subset=["consistency", "accuracy"])
    if len(valid) < 4:
        return {
            "low_consistency_mean_accuracy": None,
            "high_consistency_mean_accuracy": None,
            "median_consistency": None,
            "n_low": 0,
            "n_high": 0,
        }
    median = float(valid["consistency"].median())
    low = valid[valid["consistency"] <= median]
    high = valid[valid["consistency"] > median]
    return {
        "low_consistency_mean_accuracy": float(low["accuracy"].mean())
        if len(low)
        else None,
        "high_consistency_mean_accuracy": float(high["accuracy"].mean())
        if len(high)
        else None,
        "median_consistency": median,
        "n_low": int(len(low)),
        "n_high": int(len(high)),
    }


def _accuracy_by_sample_count(sub: pd.DataFrame) -> dict[str, dict]:
    valid = sub.dropna(subset=["n_consistency_samples", "accuracy"])
    if len(valid) == 0:
        return {}
    grouped = valid.groupby("n_consistency_samples")["accuracy"]
    return {
        str(int(k)): {
            "mean_accuracy": float(v.mean()),
            "count": int(v.count()),
        }
        for k, v in grouped
    }


def spearman_summary(df: pd.DataFrame) -> dict:
    summary: dict = {}
    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method]
        valid = sub.dropna(subset=["consistency", "rouge_l"])
        valid_acc = sub.dropna(subset=["consistency", "accuracy"])
        n = int(len(valid))
        n_acc = int(len(valid_acc))

        rho_rl, p_rl = _spearman_safe(valid["consistency"], valid["rouge_l"])
        rho_acc, p_acc = _spearman_safe(
            valid_acc["consistency"], valid_acc["accuracy"]
        )

        letter_rate = (
            float(sub["evaluation_starts_with_letter"].mean())
            if "evaluation_starts_with_letter" in sub
            else None
        )

        summary[method] = {
            "n": n,
            "n_with_accuracy": n_acc,
            "spearman_rho_consistency_rouge_l": rho_rl,
            "p_value_rouge_l": p_rl,
            "spearman_rho_consistency_accuracy": rho_acc,
            "p_value_accuracy": p_acc,
            "mean_consistency": float(sub["consistency"].mean())
            if sub["consistency"].notna().any()
            else None,
            "mean_rouge_l": float(sub["rouge_l"].mean())
            if sub["rouge_l"].notna().any()
            else None,
            "mean_accuracy": float(sub["accuracy"].mean())
            if sub["accuracy"].notna().any()
            else None,
            "mean_n_samples": float(sub["n_consistency_samples"].mean()),
            "mean_generation_calls": float(sub["generation_calls"].mean()),
            "mean_embedding_calls": float(sub["embedding_calls"].mean()),
            "mean_latency_s": float(sub["latency_s"].mean()),
            "error_count": int(sub["error"].notna().sum()),
            "stop_reason_counts": _stop_reason_counts(sub),
            "consistency_group_analysis": _consistency_group_accuracy(sub),
            "accuracy_by_sample_count": _accuracy_by_sample_count(sub),
        }
    return summary


def plot_consistency_vs_metrics(df: pd.DataFrame, out_path: str | Path) -> None:
    methods = sorted(df["method"].unique())
    fig, axes = plt.subplots(
        2, len(methods), figsize=(4 * len(methods), 8), squeeze=False
    )

    for col, method in enumerate(methods):
        sub = df[df["method"] == method]

        ax1 = axes[0][col]
        v1 = sub.dropna(subset=["consistency", "rouge_l"])
        ax1.scatter(v1["consistency"], v1["rouge_l"], s=14, alpha=0.7)
        ax1.set_title(f"{method} - rouge-L")
        ax1.set_xlabel("consistency")
        ax1.set_ylabel("rouge-L")
        if len(v1) >= 3 and v1["consistency"].nunique() > 1:
            rho, _ = stats.spearmanr(v1["consistency"], v1["rouge_l"])
            if not pd.isna(rho):
                ax1.text(
                    0.05,
                    0.95,
                    f"rho={rho:.3f}\nn={len(v1)}",
                    transform=ax1.transAxes,
                    va="top",
                )
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1][col]
        v2 = sub.dropna(subset=["consistency", "accuracy"])
        ax2.scatter(v2["consistency"], v2["accuracy"], s=14, alpha=0.7)
        ax2.set_title(f"{method} - accuracy")
        ax2.set_xlabel("consistency")
        ax2.set_ylabel("accuracy")
        if len(v2) >= 3 and v2["consistency"].nunique() > 1:
            rho, _ = stats.spearmanr(v2["consistency"], v2["accuracy"])
            if not pd.isna(rho):
                ax2.text(
                    0.05,
                    0.95,
                    f"rho={rho:.3f}\nn={len(v2)}",
                    transform=ax2.transAxes,
                    va="top",
                )
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_summary_json(
    summary: dict, stats_: CallStats, config: dict, out_path: str | Path
) -> None:
    payload = {
        "methods": summary,
        "global_stats": stats_.as_dict(),
        "config": config,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_run_metadata(meta: dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
