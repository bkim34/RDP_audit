#!/usr/bin/env python3
"""Report losses_in/losses_out counts for audit runs and flag incomplete ones."""

from __future__ import annotations

import argparse
import ast
import math
import struct
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("exp_data/max_grad_norm/1.0"),
        help="Root directory containing seed subdirectories.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Seed ids to inspect.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=100,
        help="Expected number of losses in each completed .npy file.",
    )
    parser.add_argument(
        "--only-incomplete",
        action="store_true",
        help="Only print runs that are missing files or have unexpected counts.",
    )
    return parser.parse_args()


def split_run_name(run_name: str) -> tuple[str, str, str]:
    if "_eps" not in run_name:
        return run_name, "", ""

    prefix, epsilon = run_name.rsplit("_eps", 1)
    if "_" not in prefix:
        return prefix, "", epsilon

    dataset, model = prefix.rsplit("_", 1)
    return dataset, model, epsilon


def epsilon_sort_key(epsilon: str) -> tuple[int, float | str]:
    try:
        return (0, float(epsilon))
    except ValueError:
        return (1, epsilon)


def get_run_names(results_root: Path, seeds: list[int]) -> list[str]:
    run_names = set()
    for seed in seeds:
        seed_dir = results_root / f"seed{seed}"
        if not seed_dir.exists():
            continue
        for run_dir in seed_dir.iterdir():
            if run_dir.is_dir():
                run_names.add(run_dir.name)

    return sorted(
        run_names,
        key=lambda name: (
            split_run_name(name)[0],
            split_run_name(name)[1],
            epsilon_sort_key(split_run_name(name)[2]),
            name,
        ),
    )


def load_npy_count(path: Path) -> int:
    with path.open("rb") as handle:
        if handle.read(6) != b"\x93NUMPY":
            raise ValueError(f"{path} is not a valid .npy file")

        major, minor = struct.unpack("BB", handle.read(2))
        if major == 1:
            header_len = struct.unpack("<H", handle.read(2))[0]
        elif major in (2, 3):
            header_len = struct.unpack("<I", handle.read(4))[0]
        else:
            raise ValueError(f"Unsupported .npy version {major}.{minor} in {path}")

        header = handle.read(header_len).decode("latin1")
        header_dict = ast.literal_eval(header)
        shape = header_dict["shape"]

    if shape == ():
        return 1
    return int(math.prod(shape))


def load_count(path: Path) -> int | None:
    if not path.exists():
        return None
    return load_npy_count(path)


def status_for(in_count: int | None, out_count: int | None, expected_count: int) -> str:
    if in_count is None and out_count is None:
        return "missing_both"
    if in_count is None:
        return "missing_in"
    if out_count is None:
        return "missing_out"
    if in_count == expected_count and out_count == expected_count:
        return "complete"

    issues = []
    if in_count != expected_count:
        issues.append(f"in={in_count}")
    if out_count != expected_count:
        issues.append(f"out={out_count}")
    return "incomplete:" + ",".join(issues)


def render_count(count: int | None) -> str:
    return "-" if count is None else str(count)


def main() -> None:
    args = parse_args()
    seeds = sorted(args.seeds)
    run_names = get_run_names(args.results_root, seeds)

    if not run_names:
        raise FileNotFoundError(f"No run directories found under {args.results_root}")

    all_rows = []
    for run_name in run_names:
        for seed in seeds:
            run_dir = args.results_root / f"seed{seed}" / run_name
            in_count = load_count(run_dir / "losses_in.npy")
            out_count = load_count(run_dir / "losses_out.npy")
            status = status_for(in_count, out_count, args.expected_count)
            all_rows.append((seed, run_name, in_count, out_count, status))

    rows = all_rows
    if args.only_incomplete:
        rows = [row for row in all_rows if row[4] != "complete"]

    if not rows:
        print("No matching rows to display.")
        return

    seed_width = max(len("seed"), max(len(str(row[0])) for row in rows))
    run_width = max(len("run_name"), max(len(row[1]) for row in rows))
    in_width = max(len("losses_in"), max(len(render_count(row[2])) for row in rows))
    out_width = max(len("losses_out"), max(len(render_count(row[3])) for row in rows))
    status_width = max(len("status"), max(len(row[4]) for row in rows))

    header = (
        f"{'seed':<{seed_width}}  "
        f"{'run_name':<{run_width}}  "
        f"{'losses_in':>{in_width}}  "
        f"{'losses_out':>{out_width}}  "
        f"{'status':<{status_width}}"
    )
    print(header)
    print("-" * len(header))
    for seed, run_name, in_count, out_count, status in rows:
        print(
            f"{seed:<{seed_width}}  "
            f"{run_name:<{run_width}}  "
            f"{render_count(in_count):>{in_width}}  "
            f"{render_count(out_count):>{out_width}}  "
            f"{status:<{status_width}}"
        )

    incomplete_rows = [row for row in all_rows if row[4] != "complete"]
    print()
    print(
        f"Summary: {len(all_rows) - len(incomplete_rows)} complete, "
        f"{len(incomplete_rows)} incomplete, "
        f"{len(rows)} total rows shown"
    )


if __name__ == "__main__":
    main()
