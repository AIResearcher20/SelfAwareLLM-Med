from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
import sys
from pathlib import Path

import yaml

from .data import filter_by_ids, load_hf, load_jsonl
from .experiment import run_experiment, save_responses_csv
from .model import EmbeddingCache, OllamaModel
from .reporting import (
    plot_consistency_vs_metrics,
    records_to_frame,
    save_per_method_csvs,
    save_run_metadata,
    save_summary_json,
    spearman_summary,
)


def _load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_items(config: dict):
    data_cfg = config.get("data", {})
    source = data_cfg.get("source", "jsonl")

    if source == "hf":
        items = load_hf(
            dataset_name=data_cfg.get("dataset_name", "awinml/medqa"),
            config=data_cfg.get("config", "questions"),
            split=data_cfg.get("split", "test"),
            limit=data_cfg.get("limit"),
            seed=data_cfg.get("sample_seed", 13),
        )
    elif source == "jsonl":
        items = load_jsonl(data_cfg["jsonl_path"])
    else:
        raise ValueError(f"unknown data.source: {source}")

    if data_cfg.get("sample_ids_path"):
        items = filter_by_ids(items, data_cfg["sample_ids_path"])

    return items


def _ollama_version() -> str | None:
    try:
        out = subprocess.check_output(["ollama", "--version"], text=True, timeout=5)
        return out.strip()
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="selfaware-med")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = _load_items(config)
    print(f"loaded {len(items)} items")

    client = OllamaModel(
        gen_model=config["model"]["gen_model"],
        embed_model=config["model"]["embed_model"],
        base_url=config["model"].get("base_url", "http://localhost:11434"),
        timeout_s=config["model"].get("timeout_s", 120.0),
        cache=EmbeddingCache(),
    )

    started = dt.datetime.utcnow().isoformat() + "Z"
    records, raw_outcomes = run_experiment(items, client, config)
    finished = dt.datetime.utcnow().isoformat() + "Z"

    df = records_to_frame(records)
    csv_paths = save_per_method_csvs(df, out_dir)
    response_paths = save_responses_csv(raw_outcomes, out_dir)

    summary = spearman_summary(df)
    save_summary_json(summary, client.stats(), config, out_dir / "summary.json")

    plot_consistency_vs_metrics(df, out_dir / "comparison.png")

    metadata = {
        "started_utc": started,
        "finished_utc": finished,
        "python": sys.version,
        "platform": platform.platform(),
        "ollama_version": _ollama_version(),
        "gen_model": client.gen_model,
        "embed_model": client.embed_model,
        "n_items": len(items),
        "csv_files": {k: str(v) for k, v in csv_paths.items()},
        "response_files": {k: str(v) for k, v in response_paths.items()},
        "effective_config": config,
    }
    save_run_metadata(metadata, out_dir / "run_metadata.json")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
