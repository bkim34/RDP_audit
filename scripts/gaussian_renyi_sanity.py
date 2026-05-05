#!/usr/bin/env python3
"""Cheap Gaussian Renyi sanity-check sweep with fresh resampling."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn


def exact_renyi(alpha: float, delta: float) -> float:
    return 0.5 * alpha * delta**2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cheap Gaussian Renyi sanity-check experiment.")
    parser.add_argument(
        "--dim",
        type=int,
        default=100,
        help="Reference Gaussian dimension. Training uses only the sufficient-statistic coordinate x[:, 0].",
    )
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-samples", type=int, default=200000)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--deltas", nargs="+", type=float, default=[0.5, 2.0, 5.0])
    parser.add_argument("--alphas", nargs="+", type=float, default=[2.0, 5.0, 10.0])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gaussian_renyi_sanity"))
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def logmeanexp(x: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(x, dim=0) - math.log(x.numel())


class BiaslessLinearCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    @property
    def weight_value(self) -> float:
        return float(self.linear.weight.detach().cpu().item())


def draw_fresh_gaussian_batch(delta: float, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.randn(batch_size, 1, device=device)
    p = torch.randn(batch_size, 1, device=device)
    p[:, 0] += delta
    return q, p


def compute_raw_objective(
    critic: nn.Module,
    q_samples: torch.Tensor,
    p_samples: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    p_scores = critic(p_samples).reshape(-1)
    q_scores = critic(q_samples).reshape(-1)

    p_term = logmeanexp((alpha - 1.0) * p_scores) / (alpha - 1.0)
    q_term = logmeanexp(alpha * q_scores) / alpha
    return p_term - q_term


def analytic_raw_objective(weight: float, delta: float) -> float:
    return weight * delta - 0.5 * weight * weight


def train_single_model(
    delta: float,
    alpha: float,
    batch_size: int,
    steps: int,
    lr: float,
    eval_samples: int,
    log_every: int,
    seed: int,
    device: torch.device,
) -> dict[str, float | int]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    critic = BiaslessLinearCritic().to(device)
    optimizer = torch.optim.Adam(critic.parameters(), lr=lr)

    for step in range(1, steps + 1):
        q_batch, p_batch = draw_fresh_gaussian_batch(delta=delta, batch_size=batch_size, device=device)

        optimizer.zero_grad()
        raw_objective = compute_raw_objective(critic=critic, q_samples=q_batch, p_samples=p_batch, alpha=alpha)
        loss = -raw_objective
        loss.backward()
        optimizer.step()

        if log_every > 0 and (step == 1 or step % log_every == 0 or step == steps):
            print(
                f"step={step}/{steps} "
                f"raw_batch_obj={raw_objective.detach().cpu().item():.6f} "
                f"scaled_batch_obj={alpha * raw_objective.detach().cpu().item():.6f} "
                f"weight={critic.weight_value:.6f}"
            )

    critic.eval()
    with torch.no_grad():
        q_eval, p_eval = draw_fresh_gaussian_batch(delta=delta, batch_size=eval_samples, device=device)
        raw_mc_estimate = float(
            compute_raw_objective(critic=critic, q_samples=q_eval, p_samples=p_eval, alpha=alpha).detach().cpu().item()
        )

    learned_weight = critic.weight_value
    raw_analytic_estimate = analytic_raw_objective(weight=learned_weight, delta=delta)

    return {
        "steps": int(steps),
        "learned_weight": float(learned_weight),
        "raw_mc_estimate": float(raw_mc_estimate),
        "estimated_renyi": float(alpha * raw_mc_estimate),
        "raw_analytic_estimate": float(raw_analytic_estimate),
        "analytic_renyi": float(alpha * raw_analytic_estimate),
    }


def make_plot(results_df: pd.DataFrame, deltas: list[float], plot_path: Path) -> None:
    fig, axes = plt.subplots(1, len(deltas), figsize=(5 * len(deltas), 4), squeeze=False)
    axes = axes.ravel()

    for ax, delta in zip(axes, deltas):
        sub = results_df[results_df["delta"] == delta].sort_values("alpha")
        ax.plot(sub["alpha"], sub["estimated_renyi"], marker="o", label="MC estimate")
        ax.plot(sub["alpha"], sub["analytic_renyi"], marker="s", linestyle=":", label="analytic critic")
        ax.plot(sub["alpha"], sub["theoretical_renyi"], linestyle="--", label="theoretical")
        ax.set_title(f"delta={delta}")
        ax.set_xlabel("alpha")
        ax.set_ylabel("Renyi divergence")
        ax.grid(alpha=0.3)

    axes[0].legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(
        "Settings: "
        f"dim={args.dim}, "
        f"steps={args.steps}, "
        f"batch_size={args.batch_size}, "
        f"lr={args.lr}, "
        f"eval_samples={args.eval_samples}"
    )
    print("Training uses fresh Gaussian resampling each step on the 1D sufficient statistic x[:, 0].")

    results: list[dict[str, float | int]] = []

    for delta in args.deltas:
        for alpha in args.alphas:
            print(f"\nRunning delta={delta}, alpha={alpha}")
            run_stats = train_single_model(
                delta=delta,
                alpha=alpha,
                batch_size=args.batch_size,
                steps=args.steps,
                lr=args.lr,
                eval_samples=args.eval_samples,
                log_every=args.log_every,
                seed=args.seed,
                device=device,
            )

            theoretical = exact_renyi(alpha, delta)
            row = {
                "delta": float(delta),
                "alpha": float(alpha),
                "learned_weight": float(run_stats["learned_weight"]),
                "raw_mc_estimate": float(run_stats["raw_mc_estimate"]),
                "estimated_renyi": float(run_stats["estimated_renyi"]),
                "raw_analytic_estimate": float(run_stats["raw_analytic_estimate"]),
                "analytic_renyi": float(run_stats["analytic_renyi"]),
                "theoretical_renyi": float(theoretical),
                "mc_abs_error": float(abs(float(run_stats["estimated_renyi"]) - theoretical)),
                "analytic_abs_error": float(abs(float(run_stats["analytic_renyi"]) - theoretical)),
                "steps": int(run_stats["steps"]),
                "batch_size": int(args.batch_size),
                "eval_samples": int(args.eval_samples),
                "lr": float(args.lr),
                "seed": int(args.seed),
            }
            results.append(row)

            print(
                f"delta={delta}, alpha={alpha}: "
                f"mc_scaled={row['estimated_renyi']:.6f}, "
                f"analytic_scaled={row['analytic_renyi']:.6f}, "
                f"theoretical={theoretical:.6f}, "
                f"weight={row['learned_weight']:.6f}"
            )

    results_df = pd.DataFrame(results).sort_values(["delta", "alpha"]).reset_index(drop=True)

    csv_path = args.output_dir / "estimated_vs_theoretical.csv"
    plot_path = args.output_dir / "estimated_vs_theoretical.png"
    results_df.to_csv(csv_path, index=False)
    make_plot(results_df, args.deltas, plot_path)

    print(f"\nSaved CSV to: {csv_path}")
    print(f"Saved plot to: {plot_path}")
    print("\nResults:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
