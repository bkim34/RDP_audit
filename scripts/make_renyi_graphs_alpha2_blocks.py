#!/usr/bin/env python3
"""Generate alpha=2 Renyi graphs with mean/std across seed blocks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

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
from renyi_sweep import make_loaders, mu_from_eps_delta, resolve_device, rdp_eps_from_mu_alpha, train_and_evaluate

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

ALPHA = 2.0
DEFAULT_SEED_BLOCKS = [
    [5, 6, 7, 8, 9],
    [10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19],
    [20, 21, 22, 23, 24],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build alpha=2 Renyi plots with mean/std over seed blocks.")
    parser.add_argument("--exp-data-root", type=Path, default=Path("exp_data"))
    parser.add_argument("--group-roots", nargs="*", type=Path, default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--eps-values", nargs="*", default=None)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--epochs-cifar", type=int, default=250)
    parser.add_argument("--epochs-mnist", type=int, default=750)
    parser.add_argument("--batch-size", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=100)
    parser.add_argument("--ema-rate", type=float, default=1.0)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def block_label(seed_block: list[int]) -> str:
    return f"{seed_block[0]}-{seed_block[-1]}"


def dv_epochs_for_dataset(dataset_prefix: str, args: argparse.Namespace) -> int:
    if dataset_prefix.startswith("cifar10"):
        return args.epochs_cifar
    if dataset_prefix.startswith("mnist"):
        return args.epochs_mnist
    return args.epochs_cifar


def shared_eps_values(group_root: Path, dataset_prefix: str, seed_blocks: list[list[int]]) -> list[str]:
    eps_sets = []
    for seed_block in seed_blocks:
        eps_sets.append(set(available_eps_values(group_root, dataset_prefix, seed_block)))

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
    args: argparse.Namespace,
) -> list[dict[str, float | str]]:
    rows = []
    dv_epochs = dv_epochs_for_dataset(dataset_prefix, args)

    for eps_index, eps_value in enumerate(eps_values):
        dp_eps = float(eps_value)
        theoretical_rdp_eps = rdp_eps_from_mu_alpha(mu_from_eps_delta(dp_eps, args.delta), ALPHA)

        for block_index, seeds in enumerate(DEFAULT_SEED_BLOCKS):
            print(f"{group_root} | {dataset_prefix} | alpha=2 | block={block_label(seeds)} | eps={eps_value}")
            losses_in, losses_out, emp_eps_seed_values = load_group_data(group_root, dataset_prefix, eps_value, seeds)

            nearly_tight_rdp_values = convert_empirical_dp_to_rdp(emp_eps_seed_values, args.delta, ALPHA)
            nearly_tight_dp_mean = float(np.mean(emp_eps_seed_values))
            nearly_tight_rdp_mean = float(np.mean(nearly_tight_rdp_values))

            train_loader, test_loader = make_loaders(losses_in, losses_out, args.batch_size, args.train_split)
            torch.manual_seed(args.training_seed + eps_index * 100 + block_index * 2 + 0)
            dv_qp = train_and_evaluate(
                train_loader=train_loader,
                test_loader=test_loader,
                device=resolve_device(args.device, args.require_cuda),
                lr=args.lr,
                epochs=dv_epochs,
                hidden=args.hidden,
                renyi_order=ALPHA,
                ema_rate=args.ema_rate,
            )[0]

            rev_train_loader, rev_test_loader = make_loaders(losses_out, losses_in, args.batch_size, args.train_split)
            torch.manual_seed(args.training_seed + eps_index * 100 + block_index * 2 + 1)
            dv_pq = train_and_evaluate(
                train_loader=rev_train_loader,
                test_loader=rev_test_loader,
                device=resolve_device(args.device, args.require_cuda),
                lr=args.lr,
                epochs=dv_epochs,
                hidden=args.hidden,
                renyi_order=ALPHA,
                ema_rate=args.ema_rate,
            )[0]

            dv_selected = finite_max([dv_qp, dv_pq])
            dv_scaled = float("nan") if not np.isfinite(dv_selected) else max(ALPHA * dv_selected, 0.0)

            closed_qp, params_qp = closed_form_from_samples(losses_in, losses_out, ALPHA)
            closed_pq, params_pq = closed_form_from_samples(losses_out, losses_in, ALPHA)
            closed_selected = finite_max([closed_qp, closed_pq])
            closed_scaled = float("nan") if not np.isfinite(closed_selected) else max(ALPHA * closed_selected, 0.0)

            rows.append(
                {
                    "group_root": str(group_root),
                    "dataset": dataset_prefix,
                    "alpha": ALPHA,
                    "seed_block": block_label(seeds),
                    "dp_eps": dp_eps,
                    "theoretical_rdp_eps": theoretical_rdp_eps,
                    "losses_in_count": int(losses_in.shape[0]),
                    "losses_out_count": int(losses_out.shape[0]),
                    "dv_epochs": dv_epochs,
                    "dv_batch_size": args.batch_size,
                    "dv_lr": args.lr,
                    "dv_ema_rate": args.ema_rate,
                    "nearly_tight_dp_mean": nearly_tight_dp_mean,
                    "nearly_tight_rdp_mean": nearly_tight_rdp_mean,
                    "dv_qp_raw": dv_qp,
                    "dv_pq_raw": dv_pq,
                    "dv_selected_raw": dv_selected,
                    "dv_scaled_alpha": dv_scaled,
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
        theoretical = float(rows[0]["theoretical_rdp_eps"])

        nearly_vals = [float(row["nearly_tight_rdp_mean"]) for row in rows]
        dv_vals = [float(row["dv_scaled_alpha"]) for row in rows]
        closed_vals = [float(row["closed_scaled_alpha"]) for row in rows]

        summary_rows.append(
            {
                "dp_eps": dp_eps,
                "theoretical_rdp_eps": theoretical,
                "num_seed_blocks": len(rows),
                "nearly_tight_rdp_mean": float(np.mean(nearly_vals)),
                "nearly_tight_rdp_std": sample_std(nearly_vals),
                "dv_scaled_alpha_mean": float(np.mean(dv_vals)),
                "dv_scaled_alpha_std": sample_std(dv_vals),
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
                "dv_mean={dv_scaled_alpha_mean:.6f} | dv_std={dv_scaled_alpha_std:.6f} | "
                "closed_mean={closed_scaled_alpha_mean:.6f} | closed_std={closed_scaled_alpha_std:.6f}\n".format(**row)
            )


def plot_summary(plot_path: Path, title: str, summary_rows: list[dict[str, float | str]]) -> None:
    x = np.arange(len(summary_rows))
    theoretical = np.array([float(row["theoretical_rdp_eps"]) for row in summary_rows], dtype=np.float64)
    nearly_mean = np.array([float(row["nearly_tight_rdp_mean"]) for row in summary_rows], dtype=np.float64)
    nearly_std = np.array([float(row["nearly_tight_rdp_std"]) for row in summary_rows], dtype=np.float64)
    dv_mean = np.array([float(row["dv_scaled_alpha_mean"]) for row in summary_rows], dtype=np.float64)
    dv_std = np.array([float(row["dv_scaled_alpha_std"]) for row in summary_rows], dtype=np.float64)
    closed_mean = np.array([float(row["closed_scaled_alpha_mean"]) for row in summary_rows], dtype=np.float64)
    closed_std = np.array([float(row["closed_scaled_alpha_std"]) for row in summary_rows], dtype=np.float64)

    labels = [format_tick_label(v) for v in theoretical]
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hlines(
        theoretical,
        x - 1.5 * width,
        x + 1.5 * width,
        colors="red",
        linestyles="--",
        linewidth=1.5,
        label="Theoretical $\\varepsilon_2$",
    )
    ax.bar(
        x - width,
        nearly_mean,
        width=width,
        yerr=nearly_std,
        capsize=4,
        color="#f6c54e",
        label="Nearly Tight Black-Box Auditing (mean +/- sd)",
    )
    ax.bar(
        x,
        dv_mean,
        width=width,
        yerr=dv_std,
        capsize=4,
        color="#2d7db6",
        label="DV Renyi Estimate (mean +/- sd)",
    )
    ax.bar(
        x + width,
        closed_mean,
        width=width,
        yerr=closed_std,
        capsize=4,
        color="#2ca02c",
        label="Closed Form Renyi (mean +/- sd)",
    )

    ax.set_title(title)
    ax.set_xlabel("Theoretical RDP $\\varepsilon_2$ (delta=1e-5)")
    ax.set_ylabel("Estimated RDP $\\hat{\\varepsilon}_2$")
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
    device = resolve_device(args.device, args.require_cuda)
    print(f"Using device: {device}")

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

            block_rows = build_block_rows(group_root, dataset_prefix, eps_values, args)
            if not block_rows:
                continue

            summary_rows = build_summary_rows(block_rows)
            title = f"{dataset_display_name(dataset_prefix)} | {group_root.relative_to(args.exp_data_root)} | alpha=2"
            stem = f"{dataset_prefix}_alpha2_seed_blocks_renyi"

            detail_csv_path = group_root / f"{stem}_blocks.csv"
            summary_csv_path = group_root / f"{stem}_summary.csv"
            txt_path = group_root / f"{stem}.txt"
            plot_path = group_root / f"{stem}.png"

            save_csv(detail_csv_path, block_rows)
            save_csv(summary_csv_path, summary_rows)
            save_txt(txt_path, title, summary_rows)
            plot_summary(plot_path, title, summary_rows)

            print(f"Wrote {detail_csv_path}")
            print(f"Wrote {summary_csv_path}")
            print(f"Wrote {txt_path}")
            print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
