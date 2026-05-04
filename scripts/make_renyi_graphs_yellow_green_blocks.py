#!/usr/bin/env python3
"""Generate Renyi mean/std graphs for the yellow and green series."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_renyi_graphs import (
    available_eps_values,
    closed_form_from_samples,
    convert_empirical_dp_to_rdp,
    dataset_display_name,
    discover_dataset_prefixes,
    discover_group_roots,
    finite_max,
    format_tick_label,
    load_group_data,
)
from renyi_sweep import mu_from_eps_delta, rdp_eps_from_mu_alpha

Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

DEFAULT_SEED_BLOCKS = [
    [5, 6, 7, 8, 9],
    [10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19],
    [20, 21, 22, 23, 24],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build yellow/green Renyi plots with mean/std over seed blocks.")
    parser.add_argument("--exp-data-root", type=Path, default=Path("exp_data"))
    parser.add_argument("--output-root", type=Path, default=Path("alpha15_yellow_green_graphs"))
    parser.add_argument("--group-roots", nargs="*", type=Path, default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--eps-values", nargs="*", default=None)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--delta", type=float, default=1e-5)
    return parser.parse_args()


def alpha_label(alpha: float) -> str:
    return f"{alpha:g}".replace(".", "p")


def block_label(seed_block: list[int]) -> str:
    return f"{seed_block[0]}-{seed_block[-1]}"


def shared_eps_values(group_root: Path, dataset_prefix: str, seed_blocks: list[list[int]]) -> list[str]:
    eps_sets = [set(available_eps_values(group_root, dataset_prefix, seeds)) for seeds in seed_blocks]
    if not eps_sets:
        return []

    shared = set.intersection(*eps_sets)
    return sorted(shared, key=float)


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def build_block_rows(
    group_root: Path,
    dataset_prefix: str,
    eps_values: list[str],
    alpha: float,
    delta: float,
) -> list[dict[str, float | str]]:
    rows = []

    for eps_value in eps_values:
        dp_eps = float(eps_value)
        theoretical_rdp_eps = rdp_eps_from_mu_alpha(mu_from_eps_delta(dp_eps, delta), alpha)

        for seeds in DEFAULT_SEED_BLOCKS:
            print(f"{group_root} | {dataset_prefix} | alpha={alpha:g} | block={block_label(seeds)} | eps={eps_value}")
            losses_in, losses_out, emp_eps_seed_values = load_group_data(group_root, dataset_prefix, eps_value, seeds)

            nearly_tight_rdp_values = convert_empirical_dp_to_rdp(emp_eps_seed_values, delta, alpha)
            nearly_tight_dp_mean = float(np.mean(emp_eps_seed_values))
            nearly_tight_rdp_mean = float(np.mean(nearly_tight_rdp_values))

            closed_qp, params_qp = closed_form_from_samples(losses_in, losses_out, alpha)
            closed_pq, params_pq = closed_form_from_samples(losses_out, losses_in, alpha)
            closed_selected = finite_max([closed_qp, closed_pq])
            closed_scaled = float("nan") if not np.isfinite(closed_selected) else max(alpha * closed_selected, 0.0)

            rows.append(
                {
                    "group_root": str(group_root),
                    "dataset": dataset_prefix,
                    "alpha": alpha,
                    "seed_block": block_label(seeds),
                    "dp_eps": dp_eps,
                    "theoretical_rdp_eps": theoretical_rdp_eps,
                    "losses_in_count": int(losses_in.shape[0]),
                    "losses_out_count": int(losses_out.shape[0]),
                    "nearly_tight_dp_mean": nearly_tight_dp_mean,
                    "nearly_tight_rdp_mean": nearly_tight_rdp_mean,
                    "closed_qp_raw": closed_qp,
                    "closed_pq_raw": closed_pq,
                    "closed_selected_raw": closed_selected,
                    "closed_scaled_alpha": closed_scaled,
                    "closed_qp_mu0": params_qp[0],
                    "closed_qp_var0": params_qp[1],
                    "closed_qp_mu1": params_qp[2],
                    "closed_qp_var1": params_qp[3],
                    "closed_pq_mu0": params_pq[0],
                    "closed_pq_var0": params_pq[1],
                    "closed_pq_mu1": params_pq[2],
                    "closed_pq_var1": params_pq[3],
                }
            )

    return rows


def build_summary_rows(block_rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    grouped: dict[float, list[dict[str, float | str]]] = {}
    for row in block_rows:
        grouped.setdefault(float(row["dp_eps"]), []).append(row)

    summary_rows = []
    for dp_eps in sorted(grouped, key=float):
        rows = grouped[dp_eps]
        nearly_vals = [float(row["nearly_tight_rdp_mean"]) for row in rows]
        closed_vals = [float(row["closed_scaled_alpha"]) for row in rows]

        summary_rows.append(
            {
                "dp_eps": dp_eps,
                "theoretical_rdp_eps": float(rows[0]["theoretical_rdp_eps"]),
                "num_seed_blocks": len(rows),
                "nearly_tight_rdp_mean": float(np.mean(nearly_vals)),
                "nearly_tight_rdp_std": sample_std(nearly_vals),
                "closed_scaled_alpha_mean": float(np.mean(closed_vals)),
                "closed_scaled_alpha_std": sample_std(closed_vals),
            }
        )

    return summary_rows


def save_csv(csv_path: Path, rows: list[dict[str, float | str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_txt(txt_path: Path, title: str, summary_rows: list[dict[str, float | str]]) -> None:
    with txt_path.open("w", encoding="ascii") as f:
        f.write(title + "\n")
        f.write("=" * len(title) + "\n")
        f.write("Bars show mean across seed blocks 5-9, 10-14, 15-19, and 20-24.\n")
        f.write("Error bars show one standard deviation across the four seed blocks.\n\n")
        for row in summary_rows:
            f.write(
                "dp_eps={dp_eps:.6f} | theoretical_rdp_eps={theoretical_rdp_eps:.6f} | "
                "nearly_tight_mean={nearly_tight_rdp_mean:.6f} | nearly_tight_std={nearly_tight_rdp_std:.6f} | "
                "closed_mean={closed_scaled_alpha_mean:.6f} | closed_std={closed_scaled_alpha_std:.6f}\n".format(**row)
            )


def plot_summary(plot_path: Path, title: str, summary_rows: list[dict[str, float | str]], alpha: float) -> None:
    x = np.arange(len(summary_rows))
    theoretical = np.array([float(row["theoretical_rdp_eps"]) for row in summary_rows], dtype=np.float64)
    nearly_mean = np.array([float(row["nearly_tight_rdp_mean"]) for row in summary_rows], dtype=np.float64)
    nearly_std = np.array([float(row["nearly_tight_rdp_std"]) for row in summary_rows], dtype=np.float64)
    closed_mean = np.array([float(row["closed_scaled_alpha_mean"]) for row in summary_rows], dtype=np.float64)
    closed_std = np.array([float(row["closed_scaled_alpha_std"]) for row in summary_rows], dtype=np.float64)

    labels = [format_tick_label(v) for v in theoretical]
    width = 0.28

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hlines(
        theoretical,
        x - 1.2 * width,
        x + 1.2 * width,
        colors="red",
        linestyles="--",
        linewidth=1.5,
        label=f"Theoretical $\\varepsilon_{{{alpha:g}}}$",
    )
    ax.bar(
        x - width / 2,
        nearly_mean,
        width=width,
        yerr=nearly_std,
        capsize=4,
        color="#f6c54e",
        label="Nearly Tight Black-Box Auditing (mean +/- sd)",
    )
    ax.bar(
        x + width / 2,
        closed_mean,
        width=width,
        yerr=closed_std,
        capsize=4,
        color="#2ca02c",
        label="Closed Form Renyi (mean +/- sd)",
    )

    ax.set_title(title)
    ax.set_xlabel(f"Theoretical RDP $\\varepsilon_{{{alpha:g}}}$ (delta=1e-5)")
    ax.set_ylabel(f"Estimated RDP $\\hat{{\\varepsilon}}_{{{alpha:g}}}$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle=":", color="0.7")
    ax.legend(loc="upper left")
    ax.text(
        0.99,
        0.02,
        "Seed blocks: 5-9, 10-14, 15-19, 20-24\nError bars show 1 sd across seed blocks",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    group_roots = args.group_roots or discover_group_roots(args.exp_data_root)
    if not group_roots:
        raise RuntimeError(f"No experiment roots found under {args.exp_data_root}")

    for group_root in group_roots:
        dataset_prefixes = discover_dataset_prefixes(group_root, DEFAULT_SEED_BLOCKS[0])
        if args.datasets:
            dataset_prefixes = [prefix for prefix in dataset_prefixes if prefix in set(args.datasets)]
        if not dataset_prefixes:
            print(f"Skipping {group_root}: no dataset prefixes found for the first seed block")
            continue

        for dataset_prefix in dataset_prefixes:
            eps_values = shared_eps_values(group_root, dataset_prefix, DEFAULT_SEED_BLOCKS)
            if args.eps_values:
                eps_values = [eps for eps in eps_values if eps in set(args.eps_values)]
            if not eps_values:
                print(f"Skipping {group_root} / {dataset_prefix}: no shared eps values across all four seed blocks")
                continue

            block_rows = build_block_rows(group_root, dataset_prefix, eps_values, args.alpha, args.delta)
            if not block_rows:
                continue

            summary_rows = build_summary_rows(block_rows)
            rel_group = group_root.relative_to(args.exp_data_root)
            title = f"{dataset_display_name(dataset_prefix)} | {rel_group} | alpha={args.alpha:g}"
            stem = f"{dataset_prefix}_alpha{alpha_label(args.alpha)}_yellow_green_seed_blocks_renyi"
            output_dir = args.output_root / args.exp_data_root / rel_group

            detail_csv_path = output_dir / f"{stem}_blocks.csv"
            summary_csv_path = output_dir / f"{stem}_summary.csv"
            txt_path = output_dir / f"{stem}.txt"
            plot_path = output_dir / f"{stem}.png"

            save_csv(detail_csv_path, block_rows)
            save_csv(summary_csv_path, summary_rows)
            save_txt(txt_path, title, summary_rows)
            plot_summary(plot_path, title, summary_rows, args.alpha)

            print(f"Wrote {detail_csv_path}")
            print(f"Wrote {summary_csv_path}")
            print(f"Wrote {txt_path}")
            print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
