#!/usr/bin/env python3
"""Batch normality summaries for rebuttal-ready reporting."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import anderson, shapiro


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Root directory containing seed subdirectories.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="Seed ids to aggregate over.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=None,
        help="Output path without extension. Defaults inside results root.",
    )
    return parser.parse_args()


def split_run_name(run_name: str) -> tuple[str, str, str]:
    """Split a run name into dataset, model, and epsilon when possible."""
    if "_eps" not in run_name:
        return run_name, "", ""

    prefix, epsilon = run_name.rsplit("_eps", 1)
    if "_" not in prefix:
        return prefix, "", epsilon

    dataset, model = prefix.rsplit("_", 1)
    return dataset, model, epsilon


def get_run_names(results_root: Path, seeds: list[int]) -> list[str]:
    run_names = set()
    for seed in seeds:
        seed_dir = results_root / f"seed{seed}"
        if not seed_dir.exists():
            continue
        for run_dir in seed_dir.iterdir():
            if run_dir.is_dir():
                run_names.add(run_dir.name)
    return sorted(run_names)


def load_losses(results_root: Path, run_name: str, seeds: list[int]) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    losses_in = []
    losses_out = []
    used_seeds = []
    missing_seeds = []

    for seed in seeds:
        run_dir = results_root / f"seed{seed}" / run_name
        in_path = run_dir / "losses_in.npy"
        out_path = run_dir / "losses_out.npy"
        if in_path.exists() and out_path.exists():
            losses_in.append(np.load(in_path).reshape(-1))
            losses_out.append(np.load(out_path).reshape(-1))
            used_seeds.append(seed)
        else:
            missing_seeds.append(seed)

    if not used_seeds:
        raise FileNotFoundError(f"No losses found for run {run_name}")

    return np.concatenate(losses_in), np.concatenate(losses_out), used_seeds, missing_seeds


def anderson_5pct(sample: np.ndarray) -> tuple[float, float, str]:
    result = anderson(sample, dist="norm")
    sig_levels = np.array(result.significance_level, dtype=float)
    crit_values = np.array(result.critical_values, dtype=float)
    idx = int(np.argmin(np.abs(sig_levels - 5.0)))
    stat = float(result.statistic)
    crit_5pct = float(crit_values[idx])
    decision = "fail_to_reject_normality" if stat < crit_5pct else "reject_normality"
    return stat, crit_5pct, decision


def summarize_sample(sample: np.ndarray, prefix: str) -> dict[str, object]:
    clean = np.asarray(sample, dtype=float).reshape(-1)
    clean = clean[np.isfinite(clean)]
    if clean.size < 3:
        raise ValueError(f"Need at least 3 finite values for {prefix}, got {clean.size}")

    shapiro_result = shapiro(clean)
    anderson_stat, anderson_crit_5pct, anderson_decision = anderson_5pct(clean)

    return {
        f"{prefix}_n": int(clean.size),
        f"{prefix}_shapiro_w": float(shapiro_result.statistic),
        f"{prefix}_shapiro_p": float(shapiro_result.pvalue),
        f"{prefix}_anderson_a2": anderson_stat,
        f"{prefix}_anderson_crit_5pct": anderson_crit_5pct,
        f"{prefix}_anderson_decision_5pct": anderson_decision,
        f"{prefix}_gaussian_mean": float(np.mean(clean)),
        f"{prefix}_gaussian_std": float(np.std(clean, ddof=1)),
    }


def format_value(column: str, value: object) -> str:
    if isinstance(value, float):
        if column.endswith("_p"):
            return f"{value:.3e}"
        return f"{value:.6f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    columns = [
        "dataset",
        "model",
        "epsilon",
        "run_name",
        "used_seeds",
        "losses_in_shapiro_w",
        "losses_in_shapiro_p",
        "losses_in_anderson_a2",
        "losses_in_anderson_crit_5pct",
        "losses_in_anderson_decision_5pct",
        "losses_in_gaussian_mean",
        "losses_in_gaussian_std",
        "losses_out_shapiro_w",
        "losses_out_shapiro_p",
        "losses_out_anderson_a2",
        "losses_out_anderson_crit_5pct",
        "losses_out_anderson_decision_5pct",
        "losses_out_gaussian_mean",
        "losses_out_gaussian_std",
    ]

    with path.open("w") as handle:
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            rendered = [format_value(column, row[column]) for column in columns]
            handle.write("| " + " | ".join(rendered) + " |\n")


def epsilon_sort_key(epsilon: str) -> tuple[int, float | str]:
    try:
        return (0, float(epsilon))
    except ValueError:
        return (1, epsilon)


def main() -> None:
    args = parse_args()
    results_root = args.results_root
    seeds = sorted(args.seeds)
    output_stem = args.output_stem or (results_root / f"normality_summary_seeds{'-'.join(map(str, seeds))}")

    run_names = get_run_names(results_root, seeds)
    if not run_names:
        raise FileNotFoundError(f"No run directories found under {results_root}")

    rows = []
    for run_name in run_names:
        losses_in, losses_out, used_seeds, missing_seeds = load_losses(results_root, run_name, seeds)
        dataset, model, epsilon = split_run_name(run_name)

        row = {
            "dataset": dataset,
            "model": model,
            "epsilon": epsilon,
            "run_name": run_name,
            "used_seeds": ",".join(map(str, used_seeds)),
            "missing_seeds": ",".join(map(str, missing_seeds)),
            "n_seeds": len(used_seeds),
        }
        row.update(summarize_sample(losses_in, "losses_in"))
        row.update(summarize_sample(losses_out, "losses_out"))
        rows.append(row)

    rows.sort(key=lambda row: (str(row["dataset"]), str(row["model"]), epsilon_sort_key(str(row["epsilon"])), str(row["run_name"])))

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_stem.with_suffix(".csv")
    md_path = output_stem.with_suffix(".md")
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)

    print(f"Saved {len(rows)} run summaries")
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
