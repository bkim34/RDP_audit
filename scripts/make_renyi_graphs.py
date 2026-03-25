#!/usr/bin/env python3
"""Generate alpha-specific Renyi graphs and tables from exp_data."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from renyi_sweep import make_loaders, mu_from_eps_delta, resolve_device, rdp_eps_from_mu_alpha, train_and_evaluate

Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Renyi alpha=3/5 plots for exp_data folders.")
    parser.add_argument("--exp-data-root", type=Path, default=Path("exp_data"))
    parser.add_argument("--group-roots", nargs="*", type=Path, default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--eps-values", nargs="*", default=None)
    parser.add_argument("--alphas", nargs="+", type=float, default=[3.0, 5.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[5, 6, 7, 8, 9])
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=100)
    parser.add_argument("--ema-rate", type=float, default=0.25)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def discover_group_roots(exp_data_root: Path) -> list[Path]:
    roots = []
    for candidate in sorted(exp_data_root.glob("*/*")):
        if candidate.is_dir() and any((candidate / f"seed{seed}").is_dir() for seed in [5, 6, 7, 8, 9]):
            roots.append(candidate)
    return roots


def discover_dataset_prefixes(group_root: Path, seeds: Iterable[int]) -> list[str]:
    prefixes = set()
    for seed in seeds:
        seed_dir = group_root / f"seed{seed}"
        if not seed_dir.is_dir():
            continue
        for run_dir in seed_dir.iterdir():
            if run_dir.is_dir() and "_eps" in run_dir.name:
                prefixes.add(run_dir.name.rsplit("_eps", 1)[0])
    return sorted(prefixes)


def available_eps_values(group_root: Path, dataset_prefix: str, seeds: Iterable[int]) -> list[str]:
    eps_sets = []
    for seed in seeds:
        seed_dir = group_root / f"seed{seed}"
        if not seed_dir.is_dir():
            return []

        curr = set()
        for run_dir in seed_dir.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith(f"{dataset_prefix}_eps"):
                for needed in ["losses_in.npy", "losses_out.npy", "emp_eps_loss.npy"]:
                    if not (run_dir / needed).exists():
                        break
                else:
                    curr.add(run_dir.name.rsplit("_eps", 1)[1])
        eps_sets.append(curr)

    if not eps_sets:
        return []

    shared = set.intersection(*eps_sets)
    return sorted(shared, key=float)


def fit_gaussian_mle(x: np.ndarray, var_floor: float = 1e-12) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("No finite samples available.")
    mu = float(np.mean(x))
    var = float(np.mean((x - mu) ** 2))
    return mu, max(var, var_floor)


def renyi_gaussian_closed_form(mu0: float, var0: float, mu1: float, var1: float, alpha: float) -> float:
    sigma_alpha_sq = (1.0 - alpha) * var0 + alpha * var1
    if sigma_alpha_sq <= 0:
        return float("inf")

    sigma0 = math.sqrt(var0)
    sigma1 = math.sqrt(var1)
    sigma_alpha = math.sqrt(sigma_alpha_sq)

    mean_term = alpha * (mu1 - mu0) ** 2 / (2.0 * sigma_alpha_sq)
    scale_term = (1.0 / (1.0 - alpha)) * math.log(
        sigma_alpha / ((sigma0 ** (1.0 - alpha)) * (sigma1 ** alpha))
    )
    return float(mean_term + scale_term)


def closed_form_from_samples(samples0: np.ndarray, samples1: np.ndarray, alpha: float) -> tuple[float, tuple[float, float, float, float]]:
    mu0, var0 = fit_gaussian_mle(samples0)
    mu1, var1 = fit_gaussian_mle(samples1)
    return renyi_gaussian_closed_form(mu0, var0, mu1, var1, alpha), (mu0, var0, mu1, var1)


def finite_max(values: Iterable[float]) -> float:
    finite_vals = [float(v) for v in values if np.isfinite(v)]
    if not finite_vals:
        return float("nan")
    return max(finite_vals)


def load_group_data(group_root: Path, dataset_prefix: str, eps_value: str, seeds: Iterable[int]) -> tuple[np.ndarray, np.ndarray, list[float]]:
    losses_in = []
    losses_out = []
    emp_eps = []

    for seed in seeds:
        run_dir = group_root / f"seed{seed}" / f"{dataset_prefix}_eps{eps_value}"
        losses_in.append(np.load(run_dir / "losses_in.npy").reshape(-1))
        losses_out.append(np.load(run_dir / "losses_out.npy").reshape(-1))
        emp_eps.append(float(np.load(run_dir / "emp_eps_loss.npy").reshape(-1)[0]))

    return np.concatenate(losses_in), np.concatenate(losses_out), emp_eps


def convert_empirical_dp_to_rdp(emp_eps_values: Iterable[float], delta: float, alpha: float) -> list[float]:
    converted = []
    for eps in emp_eps_values:
        mu = mu_from_eps_delta(float(max(eps, 0.0)), delta)
        converted.append(rdp_eps_from_mu_alpha(mu, alpha))
    return converted


def dataset_display_name(dataset_prefix: str) -> str:
    if dataset_prefix.startswith("cifar10"):
        return "CIFAR-10"
    if dataset_prefix.startswith("mnist"):
        return "MNIST"
    return dataset_prefix


def format_tick_label(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def build_rows(
    group_root: Path,
    dataset_prefix: str,
    eps_values: Iterable[str],
    alpha: float,
    seeds: Iterable[int],
    delta: float,
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict[str, float | str]]:
    rows = []

    for eps_index, eps_value in enumerate(eps_values):
        print(f"{group_root} | {dataset_prefix} | alpha={alpha:g} | eps={eps_value}")
        losses_in, losses_out, emp_eps_seed_values = load_group_data(group_root, dataset_prefix, eps_value, seeds)

        empirical_rdp_seed_values = convert_empirical_dp_to_rdp(emp_eps_seed_values, delta, alpha)
        nearly_tight_rdp_mean = float(np.mean(empirical_rdp_seed_values))
        nearly_tight_dp_mean = float(np.mean(emp_eps_seed_values))

        train_loader, test_loader = make_loaders(losses_in, losses_out, args.batch_size, args.train_split)

        torch.manual_seed(args.training_seed + int(alpha) * 100 + eps_index * 10 + 0)
        dv_qp = train_and_evaluate(
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            lr=args.lr,
            epochs=args.epochs,
            hidden=args.hidden,
            renyi_order=alpha,
            ema_rate=args.ema_rate,
        )[0]

        rev_train_loader, rev_test_loader = make_loaders(losses_out, losses_in, args.batch_size, args.train_split)
        torch.manual_seed(args.training_seed + int(alpha) * 100 + eps_index * 10 + 1)
        dv_pq = train_and_evaluate(
            train_loader=rev_train_loader,
            test_loader=rev_test_loader,
            device=device,
            lr=args.lr,
            epochs=args.epochs,
            hidden=args.hidden,
            renyi_order=alpha,
            ema_rate=args.ema_rate,
        )[0]

        dv_selected = finite_max([dv_qp, dv_pq])
        dv_scaled = float("nan") if not np.isfinite(dv_selected) else max(alpha * dv_selected, 0.0)

        closed_qp, params_qp = closed_form_from_samples(losses_in, losses_out, alpha)
        closed_pq, params_pq = closed_form_from_samples(losses_out, losses_in, alpha)
        closed_selected = finite_max([closed_qp, closed_pq])
        closed_scaled = float("nan") if not np.isfinite(closed_selected) else max(alpha * closed_selected, 0.0)

        dp_eps = float(eps_value)
        theoretical_rdp_eps = rdp_eps_from_mu_alpha(mu_from_eps_delta(dp_eps, delta), alpha)

        rows.append(
            {
                "group_root": str(group_root),
                "dataset": dataset_prefix,
                "alpha": alpha,
                "dp_eps": dp_eps,
                "theoretical_rdp_eps": theoretical_rdp_eps,
                "losses_in_count": int(losses_in.shape[0]),
                "losses_out_count": int(losses_out.shape[0]),
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


def save_csv(csv_path: Path, rows: list[dict[str, float | str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_txt(txt_path: Path, rows: list[dict[str, float | str]], title: str) -> None:
    with txt_path.open("w", encoding="ascii") as f:
        f.write(title + "\n")
        f.write("=" * len(title) + "\n\n")
        for row in rows:
            f.write(
                "dp_eps={dp_eps:.6f} | theoretical_rdp_eps={theoretical_rdp_eps:.6f} | "
                "nearly_tight_rdp_mean={nearly_tight_rdp_mean:.6f} | "
                "dv_scaled_alpha={dv_scaled_alpha:.6f} | "
                "closed_scaled_alpha={closed_scaled_alpha:.6f}\n".format(**row)
            )


def plot_rows(plot_path: Path, rows: list[dict[str, float | str]], title: str, alpha: float) -> None:
    x = np.arange(len(rows))
    theoretical = np.array([float(row["theoretical_rdp_eps"]) for row in rows], dtype=np.float64)
    nearly_tight = np.array([float(row["nearly_tight_rdp_mean"]) for row in rows], dtype=np.float64)
    dv = np.array([float(row["dv_scaled_alpha"]) for row in rows], dtype=np.float64)
    closed = np.array([float(row["closed_scaled_alpha"]) for row in rows], dtype=np.float64)

    labels = [format_tick_label(v) for v in theoretical]
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, theoretical, color="red", linestyle="--", linewidth=1.5, label="Theoretical $\\varepsilon_\\alpha$")
    ax.bar(x - width, nearly_tight, width=width, color="#f6c54e", label="Nearly Tight Black-Box Auditing")
    ax.bar(x, dv, width=width, color="#2d7db6", label="DV Renyi Estimate")
    ax.bar(x + width, closed, width=width, color="#2ca02c", label="Closed Form Renyi")

    ax.set_title(title)
    ax.set_xlabel(f"Theoretical RDP $\\varepsilon_{{{int(alpha)}}}$ (delta=1e-5)")
    ax.set_ylabel(f"Estimated RDP $\\hat{{\\varepsilon}}_{{{int(alpha)}}}$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle=":", color="0.7")
    ax.legend(loc="upper left")
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
        dataset_prefixes = discover_dataset_prefixes(group_root, args.seeds)
        if args.datasets:
            dataset_prefixes = [prefix for prefix in dataset_prefixes if prefix in set(args.datasets)]
        if not dataset_prefixes:
            print(f"Skipping {group_root}: no dataset prefixes found for seeds {args.seeds}")
            continue

        for dataset_prefix in dataset_prefixes:
            eps_values = available_eps_values(group_root, dataset_prefix, args.seeds)
            if args.eps_values:
                eps_values = [eps for eps in eps_values if eps in set(args.eps_values)]
            if not eps_values:
                print(f"Skipping {group_root} / {dataset_prefix}: no shared eps values across seeds {args.seeds}")
                continue

            for alpha in args.alphas:
                rows = build_rows(
                    group_root=group_root,
                    dataset_prefix=dataset_prefix,
                    eps_values=eps_values,
                    alpha=alpha,
                    seeds=args.seeds,
                    delta=args.delta,
                    device=device,
                    args=args,
                )
                if not rows:
                    continue

                alpha_label = int(alpha) if float(alpha).is_integer() else alpha
                title = f"{dataset_display_name(dataset_prefix)} | {group_root.relative_to(args.exp_data_root)} | alpha={alpha_label}"
                stem = f"{dataset_prefix}_alpha{alpha_label}_renyi"

                csv_path = group_root / f"{stem}.csv"
                txt_path = group_root / f"{stem}.txt"
                plot_path = group_root / f"{stem}.png"

                save_csv(csv_path, rows)
                save_txt(txt_path, rows, title)
                plot_rows(plot_path, rows, title, alpha)

                print(f"Wrote {csv_path}")
                print(f"Wrote {txt_path}")
                print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
