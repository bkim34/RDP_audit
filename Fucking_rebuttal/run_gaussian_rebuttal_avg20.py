#!/usr/bin/env python3
"""Run the Gaussian Re'nyi rebuttal sweep and average across multiple runs."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gaussian Renyi rebuttal sweep averaged across runs.")
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--train-size", type=int, default=40000)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--require-cuda", action="store_true", help="Fail fast unless CUDA is available.")
    parser.add_argument("--method", type=str, default="DV_Renyi", choices=["DV_Renyi"])
    parser.add_argument("--rescaled", action="store_true", help="Optimize the alpha-scaled objective.")
    parser.add_argument("--deltas", nargs="+", type=float, required=True)
    parser.add_argument("--alphas", nargs="+", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def exact_renyi(alpha: float, delta: float) -> float:
    return 0.5 * alpha * delta**2


def exact_rescaled_renyi(alpha: float, delta: float) -> float:
    return alpha * exact_renyi(alpha, delta)


class Critic(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_dataset(
    dim: int,
    train_size: int,
    test_size: int,
    delta: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    total_size = train_size + test_size
    generator = torch.Generator().manual_seed(seed)

    q = torch.randn(total_size, dim, generator=generator)
    p = torch.randn(total_size, dim, generator=generator)
    p[:, 0] += delta

    q_train = q[:train_size].clone()
    p_train = p[:train_size].clone()
    q_test = q[train_size:].clone()
    p_test = p[train_size:].clone()
    return q_train, p_train, q_test, p_test


def make_minibatches(
    q_train: torch.Tensor,
    p_train: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
):
    n = min(q_train.shape[0], p_train.shape[0])
    permutation = torch.randperm(n, generator=generator)
    for start in range(0, n, batch_size):
        batch_indices = permutation[start : start + batch_size]
        yield q_train[batch_indices], p_train[batch_indices]


def compute_dv_objective(
    critic: nn.Module,
    q_samples: torch.Tensor,
    p_samples: torch.Tensor,
    alpha: float,
    rescaled: bool,
) -> torch.Tensor:
    alpha_scaling = alpha if rescaled else 1.0
    p_data = critic(p_samples).reshape(-1)
    q_data = critic(q_samples).reshape(-1)

    p_max = torch.max((alpha - 1.0) * p_data)
    q_max = torch.max(alpha * q_data)

    p_term = (
        alpha_scaling
        / (alpha - 1.0)
        * torch.log(torch.mean(torch.exp((((alpha - 1.0) * p_data) - p_max) / alpha_scaling)))
        + p_max / (alpha - 1.0)
    )
    q_term = (
        q_max / alpha
        + alpha_scaling / alpha * torch.log(torch.mean(torch.exp(((alpha * q_data) - q_max) / alpha_scaling)))
    )
    return p_term - q_term


def evaluate(
    critic: nn.Module,
    q_test: torch.Tensor,
    p_test: torch.Tensor,
    alpha: float,
    rescaled: bool,
    device: torch.device,
) -> float:
    critic.eval()
    with torch.no_grad():
        objective = compute_dv_objective(
            critic=critic,
            q_samples=q_test.to(device),
            p_samples=p_test.to(device),
            alpha=alpha,
            rescaled=rescaled,
        )
    critic.train()
    return float(objective.detach().cpu().item())


def train_one_run(
    q_train: torch.Tensor,
    p_train: torch.Tensor,
    q_test: torch.Tensor,
    p_test: torch.Tensor,
    dim: int,
    hidden_dim: int,
    alpha: float,
    batch_size: int,
    epochs: int,
    lr: float,
    rescaled: bool,
    seed: int,
    device: torch.device,
) -> tuple[list[float], float, int]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    critic = Critic(dim=dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
    batch_generator = torch.Generator().manual_seed(seed)

    q_test = q_test.to(device)
    p_test = p_test.to(device)

    estimates_by_epoch = [evaluate(critic, q_test, p_test, alpha, rescaled, device)]
    best_estimate = estimates_by_epoch[0]
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        for q_batch, p_batch in make_minibatches(q_train, p_train, batch_size, batch_generator):
            q_batch = q_batch.to(device)
            p_batch = p_batch.to(device)

            optimizer.zero_grad()
            objective = compute_dv_objective(
                critic=critic,
                q_samples=q_batch,
                p_samples=p_batch,
                alpha=alpha,
                rescaled=rescaled,
            )
            loss = -objective
            loss.backward()
            optimizer.step()

        epoch_estimate = evaluate(critic, q_test, p_test, alpha, rescaled, device)
        estimates_by_epoch.append(epoch_estimate)
        if epoch_estimate > best_estimate:
            best_estimate = epoch_estimate
            best_epoch = epoch

    return estimates_by_epoch, best_estimate, best_epoch


def make_final_plot(summary_df: pd.DataFrame, plot_path: Path, rescaled: bool) -> None:
    alphas = sorted(summary_df["alpha"].unique())
    fig, axes = plt.subplots(1, len(alphas), figsize=(5 * len(alphas), 4), squeeze=False)
    axes = axes.ravel()

    target_col = "exact_rescaled" if rescaled else "exact_renyi"
    target_label = "exact scaled" if rescaled else "exact"

    for ax, alpha in zip(axes, alphas):
        sub = summary_df[summary_df["alpha"] == alpha].sort_values("delta")
        ax.errorbar(
            sub["delta"],
            sub["mean_final_estimate"],
            yerr=sub["std_final_estimate"],
            marker="o",
            capsize=4,
            label="mean final estimate",
        )
        ax.plot(sub["delta"], sub[target_col], linestyle="--", label=target_label)
        ax.set_title(f"alpha={alpha}")
        ax.set_xlabel("delta")
        ax.set_ylabel("Renyi objective")
        ax.grid(alpha=0.3)

    axes[0].legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if args.require_cuda and (device.type != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("CUDA was required for this run, but no CUDA-enabled PyTorch device is available.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    config_path = args.output_dir / "config.json"
    config = vars(args).copy()
    config["output_dir"] = str(args.output_dir)
    config["device_resolved"] = str(device)
    config["torch_version"] = torch.__version__
    config["cuda_available"] = bool(torch.cuda.is_available())
    config["cuda_device_count"] = int(torch.cuda.device_count())
    config["exact_target_used"] = "alpha^2 * delta^2 / 2" if args.rescaled else "alpha * delta^2 / 2"
    pd.Series(config).to_json(config_path, indent=2)

    print(f"Using device: {device}")
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(
        "Settings: "
        f"dim={args.dim}, train_size={args.train_size}, test_size={args.test_size}, "
        f"batch_size={args.batch_size}, epochs={args.epochs}, lr={args.lr}, "
        f"hidden_dim={args.hidden_dim}, num_runs={args.num_runs}, rescaled={args.rescaled}"
    )
    print(f"Output directory: {args.output_dir}")

    per_epoch_rows: list[dict[str, float | int]] = []
    final_rows: list[dict[str, float | int]] = []

    for delta_index, delta in enumerate(args.deltas):
        for run in range(args.num_runs):
            data_seed = args.base_seed + 10_000 * delta_index + run
            q_train, p_train, q_test, p_test = make_dataset(
                dim=args.dim,
                train_size=args.train_size,
                test_size=args.test_size,
                delta=delta,
                seed=data_seed,
            )

            for alpha_index, alpha in enumerate(args.alphas):
                train_seed = args.base_seed + 100_000 * delta_index + 1_000 * alpha_index + run
                estimates_by_epoch, best_estimate, best_epoch = train_one_run(
                    q_train=q_train,
                    p_train=p_train,
                    q_test=q_test,
                    p_test=p_test,
                    dim=args.dim,
                    hidden_dim=args.hidden_dim,
                    alpha=alpha,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    lr=args.lr,
                    rescaled=args.rescaled,
                    seed=train_seed,
                    device=device,
                )

                exact = exact_renyi(alpha, delta)
                exact_scaled = exact_rescaled_renyi(alpha, delta)
                target_exact = exact_scaled if args.rescaled else exact
                final_estimate = estimates_by_epoch[-1]

                print(
                    f"delta={delta}, alpha={alpha}, run={run}: "
                    f"final={final_estimate:.6f}, best={best_estimate:.6f}, "
                    f"target={target_exact:.6f}, best_epoch={best_epoch}"
                )

                for epoch, estimate in enumerate(estimates_by_epoch):
                    per_epoch_rows.append(
                        {
                            "delta": float(delta),
                            "alpha": float(alpha),
                            "run": int(run),
                            "epoch": int(epoch),
                            "estimate": float(estimate),
                            "exact_renyi": float(exact),
                            "exact_rescaled": float(exact_scaled),
                        }
                    )

                final_rows.append(
                    {
                        "delta": float(delta),
                        "alpha": float(alpha),
                        "run": int(run),
                        "final_estimate": float(final_estimate),
                        "best_estimate": float(best_estimate),
                        "best_epoch": int(best_epoch),
                        "exact_renyi": float(exact),
                        "exact_rescaled": float(exact_scaled),
                    }
                )

    per_epoch_df = pd.DataFrame(per_epoch_rows).sort_values(["delta", "alpha", "run", "epoch"]).reset_index(drop=True)
    final_df = pd.DataFrame(final_rows).sort_values(["delta", "alpha", "run"]).reset_index(drop=True)

    summary_curve_df = (
        per_epoch_df.groupby(["delta", "alpha", "epoch"], as_index=False)
        .agg(
            mean_estimate=("estimate", "mean"),
            std_estimate=("estimate", "std"),
            exact_renyi=("exact_renyi", "first"),
            exact_rescaled=("exact_rescaled", "first"),
        )
        .sort_values(["delta", "alpha", "epoch"])
        .reset_index(drop=True)
    )

    summary_final_df = (
        final_df.groupby(["delta", "alpha"], as_index=False)
        .agg(
            mean_final_estimate=("final_estimate", "mean"),
            std_final_estimate=("final_estimate", "std"),
            mean_best_estimate=("best_estimate", "mean"),
            std_best_estimate=("best_estimate", "std"),
            mean_best_epoch=("best_epoch", "mean"),
            exact_renyi=("exact_renyi", "first"),
            exact_rescaled=("exact_rescaled", "first"),
        )
        .sort_values(["delta", "alpha"])
        .reset_index(drop=True)
    )

    target_col = "exact_rescaled" if args.rescaled else "exact_renyi"
    summary_final_df["target_exact"] = summary_final_df[target_col]
    summary_final_df["final_abs_error"] = (summary_final_df["mean_final_estimate"] - summary_final_df["target_exact"]).abs()
    summary_final_df["best_abs_error"] = (summary_final_df["mean_best_estimate"] - summary_final_df["target_exact"]).abs()
    summary_final_df["final_rel_error"] = (
        (summary_final_df["mean_final_estimate"] - summary_final_df["target_exact"]) / summary_final_df["target_exact"]
    )
    summary_final_df["best_rel_error"] = (
        (summary_final_df["mean_best_estimate"] - summary_final_df["target_exact"]) / summary_final_df["target_exact"]
    )

    per_epoch_path = args.output_dir / "per_epoch_estimates.csv"
    per_run_path = args.output_dir / "per_run_summary.csv"
    summary_curve_path = args.output_dir / "summary_curve.csv"
    summary_final_path = args.output_dir / "summary_final.csv"
    plot_path = args.output_dir / "summary_final.png"

    per_epoch_df.to_csv(per_epoch_path, index=False)
    final_df.to_csv(per_run_path, index=False)
    summary_curve_df.to_csv(summary_curve_path, index=False)
    summary_final_df.to_csv(summary_final_path, index=False)
    make_final_plot(summary_final_df, plot_path, rescaled=args.rescaled)

    print(f"\nSaved per-epoch estimates to: {per_epoch_path}")
    print(f"Saved per-run summary to: {per_run_path}")
    print(f"Saved averaged curves to: {summary_curve_path}")
    print(f"Saved final averaged summary to: {summary_final_path}")
    print(f"Saved plot to: {plot_path}")
    print("\nAveraged final results:")
    print(summary_final_df.to_string(index=False))


if __name__ == "__main__":
    main()
