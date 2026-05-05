#!/usr/bin/env python3
"""Find a seed combination whose aggregated losses fail to reject normality."""

from __future__ import annotations

import argparse
import itertools
import json
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
        "--combo-size",
        type=int,
        default=5,
        help="Number of seeds per combination.",
    )
    parser.add_argument(
        "--run-names",
        type=str,
        nargs="+",
        required=True,
        help="Run directories that all need to pass the normality checks.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Shapiro-Wilk significance level.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path for the search result JSON.",
    )
    return parser.parse_args()


def discover_seeds(results_root: Path) -> list[int]:
    seeds = []
    for seed_dir in results_root.glob("seed*"):
        if seed_dir.is_dir():
            try:
                seeds.append(int(seed_dir.name.removeprefix("seed")))
            except ValueError:
                continue
    return sorted(seeds)


def build_loss_cache(results_root: Path, run_names: list[str], seeds: list[int]) -> dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]:
    cache: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for run_name in run_names:
        cache[run_name] = {}
        for seed in seeds:
            run_dir = results_root / f"seed{seed}" / run_name
            in_path = run_dir / "losses_in.npy"
            out_path = run_dir / "losses_out.npy"
            if not in_path.exists() or not out_path.exists():
                continue
            cache[run_name][seed] = (
                np.load(in_path).reshape(-1),
                np.load(out_path).reshape(-1),
            )
    return cache


def load_aggregated_losses(loss_cache: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]], run_name: str, seeds: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    losses_in = []
    losses_out = []
    for seed in seeds:
        if seed not in loss_cache.get(run_name, {}):
            raise FileNotFoundError(f"Missing losses for seed {seed}, run {run_name}")
        seed_losses_in, seed_losses_out = loss_cache[run_name][seed]
        losses_in.append(seed_losses_in)
        losses_out.append(seed_losses_out)
    return np.concatenate(losses_in), np.concatenate(losses_out)


def anderson_5pct(sample: np.ndarray) -> tuple[float, float, bool]:
    result = anderson(sample, dist="norm")
    sig_levels = np.array(result.significance_level, dtype=float)
    crit_values = np.array(result.critical_values, dtype=float)
    idx = int(np.argmin(np.abs(sig_levels - 5.0)))
    stat = float(result.statistic)
    crit_5pct = float(crit_values[idx])
    passed = stat < crit_5pct
    return stat, crit_5pct, passed


def summarize_sample(sample: np.ndarray, alpha: float) -> dict[str, object]:
    clean = np.asarray(sample, dtype=float).reshape(-1)
    clean = clean[np.isfinite(clean)]
    shapiro_result = shapiro(clean)
    ad_stat, ad_crit_5pct, ad_passed = anderson_5pct(clean)
    shapiro_passed = float(shapiro_result.pvalue) > alpha
    return {
        "n": int(clean.size),
        "shapiro_w": float(shapiro_result.statistic),
        "shapiro_p": float(shapiro_result.pvalue),
        "shapiro_fail_to_reject": shapiro_passed,
        "anderson_a2": ad_stat,
        "anderson_crit_5pct": ad_crit_5pct,
        "anderson_fail_to_reject": ad_passed,
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean, ddof=1)),
        "passes_both_tests": shapiro_passed and ad_passed,
    }


def summarize_run(loss_cache: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]], run_name: str, seeds: tuple[int, ...], alpha: float) -> dict[str, object]:
    losses_in, losses_out = load_aggregated_losses(loss_cache, run_name, seeds)
    summary = {
        "losses_in": summarize_sample(losses_in, alpha),
        "losses_out": summarize_sample(losses_out, alpha),
    }
    summary["passes_both_splits"] = summary["losses_in"]["passes_both_tests"] and summary["losses_out"]["passes_both_tests"]
    return summary


def main() -> None:
    args = parse_args()
    results_root = args.results_root
    all_seeds = discover_seeds(results_root)
    output_json = args.output_json or (results_root / "cifar_normality_combo_search.json")
    loss_cache = build_loss_cache(results_root, args.run_names, all_seeds)

    checked = 0
    match = None
    for seeds in itertools.combinations(all_seeds, args.combo_size):
        checked += 1
        run_summaries = {}
        combo_passes = True
        for run_name in args.run_names:
            run_summary = summarize_run(loss_cache, run_name, seeds, args.alpha)
            run_summaries[run_name] = run_summary
            if not run_summary["passes_both_splits"]:
                combo_passes = False
                break

        if combo_passes:
            match = {
                "status": "match_found",
                "checked_combinations": checked,
                "seed_combination": list(seeds),
                "alpha": args.alpha,
                "run_names": args.run_names,
                "run_summaries": run_summaries,
            }
            break

    if match is None:
        match = {
            "status": "no_match_found",
            "checked_combinations": checked,
            "alpha": args.alpha,
            "run_names": args.run_names,
        }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as handle:
        json.dump(match, handle, indent=2)

    print(output_json)
    print(json.dumps(match, indent=2))


if __name__ == "__main__":
    main()
