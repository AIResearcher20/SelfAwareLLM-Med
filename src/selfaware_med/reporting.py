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


def spearman_summary(df: pd.DataFrame) -> dict:
    summary: dict = {}
    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method]
        valid = sub.dropna(subset=["consistency", "rouge_l"])
        n = int(len(valid))

        if n < 3:
            summary[method] = {
                "n": n,
                "spearman_rho": None,
                "p_value": None,
                "mean_consistency": None,
                "mean_rouge_l": None,
                "mean_n_samples": None,
                "mean_generation_calls": float(sub["generation_calls"].mean())
                if n > 0
                else None,
                "mean_embedding_calls": float(sub["embedding_calls"].mean())
                if n > 0
                else None,
                "mean_latency_s": float(sub["latency_s"].mean()) if n > 0 else None,
                "error_count": int(sub["error"].notna().sum()),
            }
            continue

        rho, p = stats.spearmanr(valid["consistency"], valid["rouge_l"])
        summary[method] = {
            "n": n,
            "spearman_rho": float(rho),
            "p_value": float(p),
            "mean_consistency": float(valid["consistency"].mean()),
            "mean_rouge_l": float(valid["rouge_l"].mean()),
            "mean_n_samples": float(sub["n_consistency_samples"].mean()),
            "mean_generation_calls": float(sub["generation_calls"].mean()),
            "mean_embedding_calls": float(sub["embedding_calls"].mean()),
            "mean_latency_s": float(sub["latency_s"].mean()),
            "error_count": int(sub["error"].notna().sum()),
        }
    return summary


def plot_consistency_vs_rouge(df: pd.DataFrame, out_path: str | Path) -> None:
    methods = sorted(df["method"].unique())
    fig, axes = plt.subplots(
        1, len(methods), figsize=(4 * len(methods), 4), squeeze=False
    )

    for ax, method in zip(axes[0], methods, strict=False):
        sub = df[(df["method"] == method)].dropna(subset=["consistency", "rouge_l"])
        ax.scatter(sub["consistency"], sub["rouge_l"], s=12, alpha=0.6)
        ax.set_title(method)
        ax.set_xlabel("consistency")
        ax.set_ylabel("rouge-L")
        if len(sub) >= 2:
            rho, _ = stats.spearmanr(sub["consistency"], sub["rouge_l"])
            ax.text(
                0.05,
                0.95,
                f"rho={rho:.3f}\nn={len(sub)}",
                transform=ax.transAxes,
                va="top",
            )
        ax.grid(True, alpha=0.3)

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
