#!/usr/bin/env python3
"""Histogram of solution-trace lengths from the physics hypergraph.

Trace length = number of formula steps in ``trace["path"]`` (same as
``len(trace["path"])`` in ``generate_dataset.py`` / ``traversal.py``).

The hypergraph JSON stores nodes and hyperedges only; traces are sampled by
backward traversal from each producible target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from traversal import HyperGraph

HERE = Path(__file__).parent
DEFAULT_GRAPH = HERE / "data" / "physics_hypergraph.json"


def sample_trace_lengths(
    g: HyperGraph,
    *,
    max_depth: int,
    single_domain: bool,
    samples_per_target: int,
    seed: int,
) -> tuple[list[int], list[str], int]:
    """Return (lengths, targets_per_sample, num_failed_traversals)."""
    import random

    rng = random.Random(seed)
    targets = g.producible()
    lengths: list[int] = []
    targets_used: list[str] = []
    failed = 0

    for target in targets:
        for _ in range(samples_per_target):
            trace = g.traverse(
                target,
                max_depth=max_depth,
                seed=rng.randint(0, 2**31 - 1),
                single_domain=single_domain,
            )
            if trace is None or not trace.get("path"):
                failed += 1
                continue
            lengths.append(len(trace["path"]))
            targets_used.append(target)

    return lengths, targets_used, failed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    ap.add_argument(
        "--output",
        type=Path,
        default=HERE / "data" / "plots" / "trace_lengths.png",
        help="Histogram PNG path.",
    )
    ap.add_argument("--max-depth", type=int, default=5,
                    help="Max backward-DFS depth (same default as dataset gen).")
    ap.add_argument(
        "--samples-per-target",
        type=int,
        default=32,
        help="Random traversals per producible target (default 32).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--single-domain", action="store_true", default=True)
    ap.add_argument("--no-single-domain", dest="single_domain", action="store_false")
    ap.add_argument(
        "--save-lengths",
        type=Path,
        default=None,
        help="Optional JSON file listing all sampled lengths.",
    )
    args = ap.parse_args()

    g = HyperGraph(str(args.graph))
    lengths, _, failed = sample_trace_lengths(
        g,
        max_depth=args.max_depth,
        single_domain=args.single_domain,
        samples_per_target=args.samples_per_target,
        seed=args.seed,
    )

    if not lengths:
        raise SystemExit("No traces sampled; check graph path and traversal settings.")

    arr = np.array(lengths, dtype=int)
    n_targets = len(g.producible())
    print(f"Graph: {args.graph}")
    print(f"Producible targets: {n_targets}")
    print(f"Samples per target: {args.samples_per_target}")
    print(f"Successful traces: {len(lengths)}  (failed traversals: {failed})")
    print(f"Length min={arr.min()} max={arr.max()} mean={arr.mean():.2f} "
          f"median={np.median(arr):.1f} std={arr.std():.2f}")

    if args.save_lengths is not None:
        out_json = Path(args.save_lengths)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "graph": str(args.graph),
                    "max_depth": args.max_depth,
                    "single_domain": args.single_domain,
                    "samples_per_target": args.samples_per_target,
                    "lengths": lengths,
                },
                f,
                indent=2,
            )
        print(f"Wrote lengths to {out_json}")

    import matplotlib.pyplot as plt

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(arr.min() - 0.5, arr.max() + 1.5, 1)
    ax.hist(arr, bins=bins, color="#4a90e2", edgecolor="white", linewidth=1.0, alpha=0.9)
    ax.set_xlabel("Trace length (number of formula steps in path)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Trace length distribution ({len(lengths)} samples, "
        f"{n_targets} targets × {args.samples_per_target} draws)"
    )
    ax.axvline(arr.mean(), color="#c0392b", linestyle="--", linewidth=1.5,
               label=f"mean = {arr.mean():.2f}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"Saved histogram to {args.output}")


if __name__ == "__main__":
    main()
