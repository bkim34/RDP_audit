#!/usr/bin/env python3
"""Batch f-DP audit baseline using the repo's threshold-on-loss attack.

This keeps the Gaussian f-DP helper routines from the paper appendix, but
evaluates them on the same black-box attack family used elsewhere in this repo:
thresholding the saved target losses. It pools seeds, audits every available
run, and saves only the quantities needed for the alpha=2 RDP comparison.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import norm


def rh(inverse_blow_up_function, alpha, beta, j, m, k=2):
    h = [0.0 for _ in range(j + 1)]
    r = [0.0 for _ in range(j + 1)]
    h[j] = beta
    r[j] = alpha
    for i in range(j - 1, -1, -1):
        h[i] = max(h[i + 1], (k - 1) * inverse_blow_up_function(r[i + 1]))
        r[i] = r[i + 1] + (i / (m - i)) * (h[i] - h[i + 1])
    return r, h


def audit_rh(inverse_blow_up_function, m, c, threshold=0.05, k=2):
    alpha = threshold * c / m
    beta = threshold * (m - c) / m
    r, h = rh(inverse_blow_up_function, alpha, beta, c, m, k)
    return not (r[0] + h[0] > 1.0)


def gaussianDP_blow_up_inverse(noise):
    def blow_up_inverse_function(x):
        threshold = norm.ppf(x)
        blown_up_threshold = threshold - 1.0 / noise
        return norm.cdf(blown_up_threshold)

    return blow_up_inverse_function


def gaussian_fdp_noise_to_eps_rdp_alpha2(noise):
    noise = float(noise)
    if not np.isfinite(noise) or noise <= 0.0:
        return 0.0
    return 1.0 / (noise**2)


def empirical_noise_from_counts(m, c, candidate_noises, inverse_blow_up_functions, threshold=0.05, k=2):
    """Return the first rejected Gaussian f-DP noise hypothesis.

    `audit_rh(...)` is monotone in the Gaussian noise parameter: larger noise is
    a stronger privacy hypothesis and becomes harder for a fixed attack to
    satisfy. That lets us binary-search the first rejection instead of linearly
    scanning the whole candidate grid.
    """

    def passes(idx):
        return audit_rh(inverse_blow_up_functions[idx], m=m, c=int(c), threshold=threshold, k=k)

    if not passes(0):
        return {
            "emp_noise": float(candidate_noises[0]),
            "status": "rejected_at_smallest_noise_expand_grid_down",
        }

    last_idx = len(candidate_noises) - 1
    if passes(last_idx):
        return {
            "emp_noise": np.nan,
            "status": "no_rejection_in_grid",
        }

    lo = 0
    hi = last_idx
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if passes(mid):
            lo = mid
        else:
            hi = mid

    return {
        "emp_noise": float(candidate_noises[hi]),
        "status": "ok",
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/u/bdkim4/bb-audit-dpsgd-renyi/exp_data/max_grad_norm/1.0"),
        help="Root directory containing seed subdirectories.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional single run directory name under each seed. Defaults to all shared runs.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Seed ids to pool.",
    )
    parser.add_argument(
        "--audit-threshold",
        type=float,
        default=0.05,
        help="Threshold parameter used inside the appendix audit routines.",
    )
    parser.add_argument(
        "--candidate-noise-min",
        type=float,
        default=0.10,
        help="Smallest Gaussian f-DP noise hypothesis to scan.",
    )
    parser.add_argument(
        "--candidate-noise-max",
        type=float,
        default=8.0,
        help="Largest Gaussian f-DP noise hypothesis to scan.",
    )
    parser.add_argument(
        "--candidate-noise-count",
        type=int,
        default=3000,
        help="Number of Gaussian f-DP hypotheses to scan.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory to save pooled audit outputs. Defaults to a folder under results_root.",
    )
    return parser.parse_args()


def format_seeds_tag(seeds):
    if not seeds:
        return "none"
    if seeds == list(range(seeds[0], seeds[-1] + 1)):
        return f"{seeds[0]}-{seeds[-1]}"
    return "-".join(str(seed) for seed in seeds)


def discover_run_names(results_root: Path, seeds: list[int], requested_run_name: str | None):
    run_sets = []
    for seed in seeds:
        seed_dir = results_root / f"seed{seed}"
        if not seed_dir.is_dir():
            raise FileNotFoundError(f"Missing seed directory: {seed_dir}")
        run_sets.append({path.name for path in seed_dir.iterdir() if path.is_dir()})

    common_run_names = sorted(set.intersection(*run_sets))
    if requested_run_name is None:
        return common_run_names

    if requested_run_name not in common_run_names:
        raise FileNotFoundError(
            f"Run '{requested_run_name}' is not present for every requested seed in {results_root}"
        )
    return [requested_run_name]


def load_pooled_losses(results_root: Path, run_name: str, seeds: list[int]):
    losses_in = []
    losses_out = []
    for seed in seeds:
        run_dir = results_root / f"seed{seed}" / run_name
        losses_in.append(np.load(run_dir / "losses_in.npy").astype(float).ravel())
        losses_out.append(np.load(run_dir / "losses_out.npy").astype(float).ravel())
    return np.concatenate(losses_in), np.concatenate(losses_out)


def iter_threshold_rows(losses_in, losses_out):
    total_in = int(len(losses_in))
    total_out = int(len(losses_out))
    scores = np.concatenate([losses_in, losses_out])
    labels = np.concatenate(
        [np.ones(total_in, dtype=np.int8), np.zeros(total_out, dtype=np.int8)]
    )
    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    tp = 0
    fp = 0
    idx = 0
    total = len(sorted_scores)
    while idx < total:
        tau = float(sorted_scores[idx])
        group_tp = 0
        group_fp = 0
        while idx < total and sorted_scores[idx] == tau:
            if sorted_labels[idx]:
                group_tp += 1
            else:
                group_fp += 1
            idx += 1

        tp += group_tp
        fp += group_fp
        tn = total_out - fp
        fn = total_in - tp
        c = tp + tn
        yield {
            "tau": tau,
            "m": total_in + total_out,
            "c": int(c),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "raw_tpr": float(tp / total_in),
            "raw_fpr": float(fp / total_out),
            "attack_acc": float(c / (total_in + total_out)),
        }


def audit_run(
    losses_in,
    losses_out,
    candidate_noises,
    inverse_blow_up_functions,
    audit_threshold,
    cache_by_counts,
):
    threshold_count = 0
    best = None
    seen_c_values = set()

    for row in iter_threshold_rows(losses_in, losses_out):
        threshold_count += 1
        seen_c_values.add(row["c"])
        cache_key = (row["m"], row["c"])
        if cache_key not in cache_by_counts:
            cache_by_counts[cache_key] = empirical_noise_from_counts(
                m=row["m"],
                c=row["c"],
                candidate_noises=candidate_noises,
                inverse_blow_up_functions=inverse_blow_up_functions,
                threshold=audit_threshold,
                k=2,
            )

        result = {**row, **cache_by_counts[cache_key]}
        result["eps_rdp_alpha2"] = gaussian_fdp_noise_to_eps_rdp_alpha2(result["emp_noise"])

        if best is None or (result["eps_rdp_alpha2"], result["attack_acc"]) > (
            best["eps_rdp_alpha2"],
            best["attack_acc"],
        ):
            best = result

    if best is None:
        raise ValueError("No thresholds were generated from the provided losses.")

    best["threshold_count"] = threshold_count
    best["unique_c_count"] = len(seen_c_values)
    best["pooled_losses_in"] = int(len(losses_in))
    best["pooled_losses_out"] = int(len(losses_out))
    return best


def save_run_outputs(output_root: Path, run_name: str, result: dict):
    run_output_dir = output_root / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_output_dir / "fdp_emp_noise.npy", np.array([result["emp_noise"]], dtype=float))
    np.save(
        run_output_dir / "fdp_eps_rdp_alpha2.npy",
        np.array([result["eps_rdp_alpha2"]], dtype=float),
    )


def write_summary_csv(output_root: Path, rows: list[dict]):
    summary_path = output_root / "summary.csv"
    fieldnames = [
        "run_name",
        "pooled_losses_in",
        "pooled_losses_out",
        "threshold_count",
        "unique_c_count",
        "tau",
        "m",
        "c",
        "tp",
        "fp",
        "tn",
        "fn",
        "raw_tpr",
        "raw_fpr",
        "attack_acc",
        "emp_noise",
        "eps_rdp_alpha2",
        "status",
    ]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def main():
    args = parse_args()
    seeds = sorted(args.seeds)
    run_names = discover_run_names(args.results_root, seeds, args.run_name)
    seed_tag = format_seeds_tag(seeds)
    output_root = args.output_root
    if output_root is None:
        output_root = args.results_root / f"pooled_seed{seed_tag}_fdp_same_attack"
    output_root.mkdir(parents=True, exist_ok=True)

    candidate_noises = np.linspace(
        args.candidate_noise_min,
        args.candidate_noise_max,
        args.candidate_noise_count,
    )
    inverse_blow_up_functions = [gaussianDP_blow_up_inverse(noise) for noise in candidate_noises]
    cache_by_counts = {}

    print("Batching pooled f-DP audits")
    print("---------------------------")
    print(f"results_root         : {args.results_root}")
    print(f"run_count            : {len(run_names)}")
    print(f"seeds                : {seeds}")
    print(f"output_root          : {output_root}")
    print(f"candidate_noise_grid : {args.candidate_noise_min} .. {args.candidate_noise_max}")
    print(f"candidate_noise_count: {args.candidate_noise_count}")
    print()

    summary_rows = []
    for run_name in run_names:
        losses_in, losses_out = load_pooled_losses(args.results_root, run_name, seeds)
        best = audit_run(
            losses_in=losses_in,
            losses_out=losses_out,
            candidate_noises=candidate_noises,
            inverse_blow_up_functions=inverse_blow_up_functions,
            audit_threshold=args.audit_threshold,
            cache_by_counts=cache_by_counts,
        )
        best["run_name"] = run_name
        save_run_outputs(output_root, run_name, best)
        summary_rows.append(best)

        print(
            "{run_name}: emp_noise={emp_noise:.6f} eps_rdp_alpha2={eps_rdp_alpha2:.6f} "
            "tau={tau:.6f} acc={attack_acc:.6f} status={status}".format(**best)
        )

    write_summary_csv(output_root, summary_rows)

    print()
    print("Saved outputs")
    print("-------------")
    print(f"summary_csv          : {output_root / 'summary.csv'}")
    print("per-run files        : fdp_emp_noise.npy, fdp_eps_rdp_alpha2.npy")


if __name__ == "__main__":
    main()
