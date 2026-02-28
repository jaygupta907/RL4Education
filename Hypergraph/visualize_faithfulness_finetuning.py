"""
Visualize faithfulness scores stored during RL finetuning (faithfulness_scores.jsonl).

Usage:
  python visualize_faithfulness_finetuning.py path/to/faithfulness_scores.jsonl
  python visualize_faithfulness_finetuning.py path/to/run_dir   # looks for faithfulness_scores.jsonl inside
  python visualize_faithfulness_finetuning.py                  # uses checkpoints/logs/latest or default
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    MATPLOTLIB_AVAILABLE = True
    rcParams["font.size"] = 11
    rcParams["axes.labelsize"] = 12
    rcParams["axes.titlesize"] = 14
    rcParams["figure.titlesize"] = 16
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_faithfulness_jsonl(file_path: str):
    """Load records from faithfulness_scores.jsonl. Returns list of dicts."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skip invalid line: {e}", file=sys.stderr)
    return data


def filter_valid_scores(data):
    """Return only records with a numeric faithfulness_score."""
    return [r for r in data if r.get("faithfulness_score") is not None]


def plot_faithfulness_over_episodes(data, output_path: str, window: int = 20):
    """Scatter of faithfulness vs episode + rolling mean line."""
    valid = filter_valid_scores(data)
    if not valid:
        print("No valid faithfulness scores to plot.")
        return
    episodes = [r["episode"] for r in valid]
    scores = [r["faithfulness_score"] for r in valid]
    # Sort by episode for rolling mean
    order = np.argsort(episodes)
    ep = np.array(episodes)[order]
    sc = np.array(scores)[order]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(episodes, scores, alpha=0.4, s=12, c="#2E86AB", label="Per question")
    if len(ep) >= window:
        roll = np.convolve(sc, np.ones(window) / window, mode="valid")
        ep_roll = ep[window - 1 :]
        ax.plot(ep_roll, roll, color="#A23B72", linewidth=2, label=f"Rolling mean (n={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Faithfulness score (1–10)")
    ax.set_title("Faithfulness score during finetuning")
    ax.set_ylim(0, 10.5)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_faithfulness_vs_length(data, output_path: str):
    """Box plot or scatter: faithfulness by trace length."""
    valid = filter_valid_scores(data)
    if not valid:
        return
    by_length = defaultdict(list)
    for r in valid:
        by_length[r["length"]].append(r["faithfulness_score"])
    lengths = sorted(by_length.keys())
    if not lengths:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    boxes = [by_length[L] for L in lengths]
    bp = ax.boxplot(boxes, positions=lengths, widths=0.4, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#2E86AB")
        patch.set_alpha(0.7)
    ax.set_xlabel("Trace length (number of formulas)")
    ax.set_ylabel("Faithfulness score (1–10)")
    ax.set_title("Faithfulness score by trace length")
    ax.set_xticks(lengths)
    ax.set_ylim(0, 10.5)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_mean_faithfulness_by_length(data, output_path: str):
    """Line plot: mean faithfulness score for each trace length (with count and range)."""
    valid = filter_valid_scores(data)
    if not valid:
        return
    by_length = defaultdict(list)
    for r in valid:
        by_length[r["length"]].append(r["faithfulness_score"])
    lengths = sorted(by_length.keys())
    if not lengths:
        return
    means = [np.mean(by_length[L]) for L in lengths]
    counts = [len(by_length[L]) for L in lengths]
    mins = [min(by_length[L]) for L in lengths]
    maxs = [max(by_length[L]) for L in lengths]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lengths, means, marker="o", linestyle="-", linewidth=2, markersize=10,
            color="#2E86AB", label="Mean faithfulness")
    ax.fill_between(lengths, mins, maxs, alpha=0.2, color="#2E86AB", label="Min–max range")
    for i, (L, m, n) in enumerate(zip(lengths, means, counts)):
        ax.annotate(f"n={n}", xy=(L, m), xytext=(0, 8), textcoords="offset points",
                    fontsize=9, ha="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.set_xlabel("Trace length (number of formulas)")
    ax.set_ylabel("Mean faithfulness score (1–10)")
    ax.set_title("Mean faithfulness score by trace length")
    ax.set_xticks(lengths)
    ax.set_ylim(0, 10.5)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_faithfulness_histogram(data, output_path: str):
    """Histogram of faithfulness scores."""
    valid = filter_valid_scores(data)
    if not valid:
        return
    scores = [r["faithfulness_score"] for r in valid]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=np.arange(0.5, 11.5, 1), color="#2E86AB", alpha=0.8, edgecolor="white")
    ax.set_xlabel("Faithfulness score (1–10)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of faithfulness scores")
    ax.set_xlim(0, 11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_score_vs_faithfulness(data, output_path: str):
    """Scatter: judge (difficulty) score vs faithfulness score."""
    valid = filter_valid_scores(data)
    if not valid:
        return
    judge = [r["score"] for r in valid]
    faith = [r["faithfulness_score"] for r in valid]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(judge, faith, alpha=0.5, s=20, c="#2E86AB")
    ax.set_xlabel("Judge score (difficulty, 0–10)")
    ax.set_ylabel("Faithfulness score (1–10)")
    ax.set_title("Judge score vs faithfulness score")
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(0, 10.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize faithfulness scores from RL finetuning (faithfulness_scores.jsonl)"
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=None,
        help="Path to faithfulness_scores.jsonl or to a run directory containing it",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Directory to save plots (default: same as input file)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Rolling mean window for episode plot (default: 20)",
    )
    args = parser.parse_args()

    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib required. pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    # Resolve input file
    if args.input_path is None:
        # Default: look for latest run in checkpoints/logs
        base = os.path.join(os.path.dirname(__file__), "checkpoints", "logs")
        if os.path.isdir(base):
            runs = sorted([d for d in os.listdir(base) if d.startswith("run_")], reverse=True)
            if runs:
                candidate = os.path.join(base, runs[0], "faithfulness_scores.jsonl")
                if os.path.isfile(candidate):
                    args.input_path = candidate
        if args.input_path is None:
            args.input_path = os.path.join(base, "faithfulness_scores.jsonl")
            if not os.path.isfile(args.input_path):
                print("No input path given and no default faithfulness_scores.jsonl found.", file=sys.stderr)
                print("Usage: python visualize_faithfulness_finetuning.py <path_to_jsonl_or_run_dir>", file=sys.stderr)
                sys.exit(1)

    if os.path.isdir(args.input_path):
        jsonl_path = os.path.join(args.input_path, "faithfulness_scores.jsonl")
        if not os.path.isfile(jsonl_path):
            print(f"Not found: {jsonl_path}", file=sys.stderr)
            sys.exit(1)
        input_path = jsonl_path
        if args.output_dir is None:
            args.output_dir = args.input_path
    else:
        input_path = args.input_path
        if args.output_dir is None:
            args.output_dir = os.path.dirname(os.path.abspath(input_path))

    os.makedirs(args.output_dir, exist_ok=True)

    data = load_faithfulness_jsonl(input_path)
    valid = filter_valid_scores(data)
    print(f"Loaded {len(data)} records, {len(valid)} with valid faithfulness_score from {input_path}")

    if not valid:
        print("No valid faithfulness scores. Exiting.")
        sys.exit(0)

    prefix = os.path.join(args.output_dir, "faithfulness")
    plot_faithfulness_over_episodes(data, f"{prefix}_over_episodes.png", window=args.window)
    plot_faithfulness_vs_length(data, f"{prefix}_vs_length.png")
    plot_mean_faithfulness_by_length(data, f"{prefix}_mean_by_length.png")
    plot_faithfulness_histogram(data, f"{prefix}_histogram.png")
    plot_score_vs_faithfulness(data, f"{prefix}_vs_judge_score.png")
    print("Done.")


if __name__ == "__main__":
    main()
