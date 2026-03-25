#!/usr/bin/env python3
"""Queue-friendly Renyi sweep derived from RENYI.ipynb."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

EPS = 1e-6


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def delta_from_eps_mu(eps: float, mu: float) -> float:
    if mu <= 0:
        raise ValueError("mu must be > 0")
    term1 = normal_cdf(-eps / mu + mu / 2.0)
    term2 = math.exp(eps) * normal_cdf(-eps / mu - mu / 2.0)
    delta = term1 - term2
    return max(0.0, min(1.0, delta))


def mu_from_eps_delta(eps: float, delta_target: float, tol: float = 1e-12, max_iter: int = 300) -> float:
    if eps < 0:
        raise ValueError("eps must be >= 0")
    if not (0 < delta_target < 1):
        raise ValueError("delta_target must be in (0,1)")

    lo = 1e-12
    hi = 1.0

    while delta_from_eps_mu(eps, hi) < delta_target:
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError(f"Could not bracket mu for eps={eps}, delta={delta_target}")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = delta_from_eps_mu(eps, mid)
        if abs(val - delta_target) < tol:
            return mid
        if val < delta_target:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


def rdp_eps_from_mu_alpha(mu: float, alpha: float) -> float:
    if mu <= 0:
        raise ValueError("mu must be > 0")
    if alpha <= 1:
        raise ValueError("alpha must be > 1")
    return alpha * (mu**2) / 2.0


class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, running_ema: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1)
        ctx.save_for_backward(x, running_ema)
        return torch.logsumexp(x, dim=0) - math.log(x.shape[0])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        x, running_mean = ctx.saved_tensors
        x = x.reshape(-1)

        m = x.max().detach()
        exp_shift = torch.exp(x - m).detach()
        denom = (running_mean * torch.exp(-m) + EPS) * x.shape[0]

        grad = grad_output * exp_shift / denom
        return grad, None


def logmeanexp(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return torch.logsumexp(x, dim=dim) - math.log(x.shape[dim])


def ema_loss(x: torch.Tensor, running_mean: torch.Tensor, ema_rate: float) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.reshape(-1)
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()

    if running_mean.item() == 0.0:
        running_mean = t_exp
    else:
        running_mean = ema_rate * t_exp + (1.0 - ema_rate) * running_mean

    t_log = EMALoss.apply(x, running_mean)
    return t_log, running_mean


class PairDataset(Dataset):
    def __init__(self, q: torch.Tensor, p: torch.Tensor):
        self.q = q
        self.p = p

    def __len__(self) -> int:
        return min(len(self.q), len(self.p))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q[index], self.p[index]


class TNet(nn.Module):
    def __init__(self, hidden: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.net(x)


class RenyiObjective(nn.Module):
    def __init__(self, scorer: nn.Module, renyi_order: float, ema_rate: float = 0.25):
        super().__init__()
        if renyi_order <= 1:
            raise ValueError("renyi_order must be > 1")

        self.scorer = scorer
        self.renyi_order = float(renyi_order)
        self.ema_rate = float(ema_rate)
        self.register_buffer("running_mean_q", torch.tensor(0.0))
        self.register_buffer("running_mean_p", torch.tensor(0.0))

    def forward(self, q_batch: torch.Tensor, p_batch: torch.Tensor, update_ema: bool = True) -> torch.Tensor:
        alpha = self.renyi_order
        t_q = self.scorer(q_batch).reshape(-1)
        t_p = self.scorer(p_batch).reshape(-1)

        if update_ema:
            log_mq, m_q = ema_loss((alpha - 1.0) * t_q, self.running_mean_q, self.ema_rate)
            log_mp, m_p = ema_loss(alpha * t_p, self.running_mean_p, self.ema_rate)
            self.running_mean_q.copy_(m_q)
            self.running_mean_p.copy_(m_p)
        else:
            log_mq = logmeanexp((alpha - 1.0) * t_q, dim=0)
            log_mp = logmeanexp(alpha * t_p, dim=0)

        renyi_lb = (log_mq / (alpha - 1.0)) - (log_mp / alpha)
        return -renyi_lb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Renyi notebook workflow as a batch script.")
    parser.add_argument("--results-root", type=Path, default=Path("exp_data/max_grad_norm/1.0"))
    parser.add_argument("--run-template", type=str, default="cifar10_half_cnn_eps{eps}")
    parser.add_argument("--eps-values", nargs="+", default=["2.0", "4.38", "6.57", "10.0", "17.85"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[5, 6, 7, 8, 9])
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--renyi-order", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--hidden", type=int, default=100)
    parser.add_argument("--ema-rate", type=float, default=0.25)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0, help="Training seed for the Renyi estimator.")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=Path("renyi_results/max_grad_norm_1.0_cifar10_half_cnn_seeds5_9.txt"),
    )
    return parser.parse_args()


def resolve_device(device_arg: str, require_cuda: bool) -> torch.device:
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)

    if require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required for this run, but no GPU is visible.")

    return device


def load_losses(results_root: Path, run_template: str, eps_value: str, seeds: Iterable[int]) -> tuple[np.ndarray, np.ndarray, float]:
    all_losses_in = []
    all_losses_out = []
    seed_emp_means = []

    for seed in seeds:
        base = results_root / f"seed{seed}" / run_template.format(eps=eps_value)
        in_path = base / "losses_in.npy"
        out_path = base / "losses_out.npy"

        if not in_path.exists() or not out_path.exists():
            raise FileNotFoundError(f"Missing losses for seed {seed}: {base}")

        seed_losses_in = np.load(in_path).reshape(-1)
        seed_losses_out = np.load(out_path).reshape(-1)

        print(f"eps {eps_value} seed {seed}: losses_in {seed_losses_in.shape}, losses_out {seed_losses_out.shape}")

        all_losses_in.append(seed_losses_in)
        all_losses_out.append(seed_losses_out)
        seed_emp_means.append(float(np.mean(np.concatenate([seed_losses_in, seed_losses_out]))))

    return (
        np.concatenate(all_losses_in),
        np.concatenate(all_losses_out),
        float(np.mean(seed_emp_means)),
    )


def make_loaders(losses_in: np.ndarray, losses_out: np.ndarray, batch_size: int, train_split: float) -> tuple[DataLoader, DataLoader]:
    x = torch.tensor(losses_in, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(losses_out, dtype=torch.float32).view(-1, 1)

    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    n_train = int(train_split * n)
    x_train, y_train = x[:n_train], y[:n_train]
    x_test, y_test = x[n_train:], y[n_train:]

    train_loader = DataLoader(PairDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(PairDataset(x_test, y_test), batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader


def train_and_evaluate(
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    lr: float,
    epochs: int,
    hidden: int,
    renyi_order: float,
    ema_rate: float,
) -> tuple[float, list[float]]:
    model = RenyiObjective(TNet(hidden=hidden).to(device), renyi_order=renyi_order, ema_rate=ema_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epoch_bounds = []

    for epoch in range(epochs):
        model.train()
        train_total = 0.0
        train_count = 0

        for q_batch, p_batch in train_loader:
            q_batch = q_batch.to(device)
            p_batch = p_batch.to(device)

            optimizer.zero_grad()
            loss = model(q_batch, p_batch, update_ema=True)
            loss.backward()
            optimizer.step()

            batch_size = q_batch.shape[0]
            train_total += (-loss.detach()).item() * batch_size
            train_count += batch_size

        train_avg = train_total / max(train_count, 1)
        epoch_bounds.append(train_avg)
        print(f"epoch {epoch + 1}/{epochs}: train_renyi_lb={train_avg:.6f}")

    model.eval()
    test_total = 0.0
    test_count = 0

    with torch.no_grad():
        for q_batch, p_batch in test_loader:
            q_batch = q_batch.to(device)
            p_batch = p_batch.to(device)
            loss = model(q_batch, p_batch, update_ema=False)
            batch_size = q_batch.shape[0]
            test_total += (-loss).item() * batch_size
            test_count += batch_size

    test_avg = test_total / max(test_count, 1)
    return test_avg, epoch_bounds


def format_summary_line(
    eps_value: str,
    mean_emp_loss: float,
    mu_value: float,
    rdp_value: float,
    test_bound: float,
    losses_in: np.ndarray,
    losses_out: np.ndarray,
) -> str:
    return (
        f"eps={eps_value} | "
        f"losses_in={losses_in.shape[0]} | "
        f"losses_out={losses_out.shape[0]} | "
        f"mean_emp_loss={mean_emp_loss:.10f} | "
        f"mu={mu_value:.10f} | "
        f"theory_rdp_alpha={rdp_value:.10f} | "
        f"estimated_renyi_lb={test_bound:.10f}"
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device(args.device, args.require_cuda)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")

    summaries = []

    for eps_value in args.eps_values:
        print()
        print(f"=== epsilon {eps_value} ===")
        losses_in, losses_out, mean_emp_loss = load_losses(args.results_root, args.run_template, eps_value, args.seeds)
        train_loader, test_loader = make_loaders(losses_in, losses_out, args.batch_size, args.train_split)

        test_bound, epoch_bounds = train_and_evaluate(
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            lr=args.lr,
            epochs=args.epochs,
            hidden=args.hidden,
            renyi_order=args.renyi_order,
            ema_rate=args.ema_rate,
        )

        mu_value = mu_from_eps_delta(float(eps_value), args.delta)
        rdp_value = rdp_eps_from_mu_alpha(mu_value, args.renyi_order)
        summary_line = format_summary_line(
            eps_value=eps_value,
            mean_emp_loss=mean_emp_loss,
            mu_value=mu_value,
            rdp_value=rdp_value,
            test_bound=test_bound,
            losses_in=losses_in,
            losses_out=losses_out,
        )

        summaries.append(summary_line)
        print(summary_line)
        print(f"epoch_bounds={', '.join(f'{x:.6f}' for x in epoch_bounds)}")

    args.output_txt.parent.mkdir(parents=True, exist_ok=True)
    with args.output_txt.open("w", encoding="ascii") as f:
        f.write("Renyi sweep summary\n")
        f.write(f"results_root={args.results_root}\n")
        f.write(f"run_template={args.run_template}\n")
        f.write(f"eps_values={args.eps_values}\n")
        f.write(f"seeds={args.seeds}\n")
        f.write(f"delta={args.delta}\n")
        f.write(f"renyi_order={args.renyi_order}\n")
        f.write(f"lr={args.lr}\n")
        f.write(f"epochs={args.epochs}\n")
        f.write(f"batch_size={args.batch_size}\n")
        f.write(f"device={device}\n")
        f.write("\n")
        for line in summaries:
            f.write(line + "\n")

    print()
    print(f"Wrote summary to {args.output_txt}")


if __name__ == "__main__":
    main()
