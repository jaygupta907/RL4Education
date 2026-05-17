"""Metrics on the eval results.

Reads `eval_results.json` records, each of which already carries the two
Claude judgments stored at eval time:
  - claude_difficulty:    integer 1..10
  - claude_faithfulness:  {leaf_hits, target_present, variable_coverage,
                            faithful}

This script does no judging itself; it only aggregates and plots:
  i)   Confusion matrix of requested vs Claude-judged difficulty
       (raw 10x10 and bucketed easy/medium/hard).
  ii)  Faithfulness: aggregate Claude-judged faithfulness rates.
  iii) Distractors: physics keywords in the question that aren't in the trace.
  iv)  Difficulty alignment: MAE / Pearson r / per-target monotonicity.
  v)   Chapter coverage and question length.
"""
import argparse
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from traversal import HyperGraph

HERE = Path(__file__).parent


# ----- plotting --------------------------------------------------------------

def _setup_mpl_style():
    import matplotlib.pyplot as plt
    use_tex = bool(shutil.which("latex"))
    rc = {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 110,
    }
    if use_tex:
        rc["text.usetex"] = True
        rc["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{bm}"
    plt.rcParams.update(rc)
    return use_tex


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _bold(use_tex):
    return (lambda s: r"\textbf{" + s + "}") if use_tex else (lambda s: s)


def plot_difficulty_alignment(rows, output_path):
    import matplotlib.pyplot as plt
    buckets = defaultdict(list)
    for r in rows:
        d = r.get("requested_difficulty")
        c = r.get("claude_difficulty")
        if isinstance(d, int) and isinstance(c, int) and 1 <= c <= 10:
            buckets[d].append(c)
    if not buckets:
        return False

    xs = sorted(buckets)
    means = np.array([np.mean(buckets[d]) for d in xs])
    sems = np.array([
        np.std(buckets[d], ddof=1) / math.sqrt(len(buckets[d]))
        if len(buckets[d]) > 1 else 0.0
        for d in xs
    ])
    n_total = sum(len(v) for v in buckets.values())

    use_tex = _setup_mpl_style()
    bold = _bold(use_tex)

    fig, ax = plt.subplots(figsize=(7, 6.2))
    band_x = np.linspace(1, 10, 100)
    ax.fill_between(band_x, band_x - 1, band_x + 1, color="0.7",
                    alpha=0.18, linewidth=0,
                    label=r"$\pm 1$ tolerance band")
    ax.plot([1, 10], [1, 10], linestyle="--", color="0.40",
            linewidth=1.3, label=r"$y = x$ (perfect alignment)")
    ax.errorbar(xs, means, yerr=sems, fmt="o-", capsize=4,
                color="#1f4e79", linewidth=2.2, markersize=8,
                markerfacecolor="#1f77b4", markeredgecolor="white",
                markeredgewidth=1.2,
                label=r"mean Claude score $\pm$ SEM")
    for d, m in zip(xs, means):
        ax.annotate(f"{m:.1f}", (d, m), textcoords="offset points",
                    xytext=(8, 7), fontsize=10, color="#1f4e79")

    ax.set_xticks(range(1, 11))
    ax.set_yticks(range(1, 11))
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.5, 10.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Requested difficulty $d_{\mathrm{req}}$")
    ax.set_ylabel(r"Mean Claude difficulty $\bar{d}_{\mathrm{claude}}$")
    ax.set_title(bold(r"Requested vs.\ Claude-judged difficulty") +
                 "\n" + rf"($n = {n_total}$ questions, "
                 rf"{len(rows)} eval records)")
    ax.legend(loc="lower right", frameon=False)
    _save(fig, output_path)
    return True


def plot_confusion(cm_dict, output_path, title, xlabel, ylabel,
                   row_normalise=True):
    import matplotlib.pyplot as plt
    labels = cm_dict["labels"]
    M = np.array(cm_dict["matrix"], dtype=float)
    if M.sum() == 0:
        return False
    if row_normalise:
        with np.errstate(invalid="ignore", divide="ignore"):
            row_sum = M.sum(axis=1, keepdims=True)
            P = np.divide(M, row_sum, where=row_sum > 0)
    else:
        P = M / M.max()

    use_tex = _setup_mpl_style()
    bold = _bold(use_tex)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(5.5, 0.55 * n + 3),
                                    max(5.0, 0.55 * n + 2.5)))
    im = ax.imshow(P, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(bold(title))
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            count = int(M[i, j])
            if count == 0:
                continue
            color = "white" if P[i, j] > 0.55 else "0.15"
            ax.text(j, i, f"{count}", ha="center", va="center",
                    fontsize=10, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("row-normalised frequency" if row_normalise
                   else r"count / $\max$")
    _save(fig, output_path)
    return True


def plot_distractors_by_difficulty(rows, output_path):
    import matplotlib.pyplot as plt
    by = defaultdict(list)
    for r in rows:
        d = r.get("requested_difficulty")
        if isinstance(d, int):
            by[d].append(r["distractor_count"])
    if not by:
        return False
    xs = sorted(by)
    means = [np.mean(by[d]) for d in xs]
    sems = [np.std(by[d], ddof=1) / math.sqrt(len(by[d])) if len(by[d]) > 1
            else 0.0 for d in xs]

    use_tex = _setup_mpl_style()
    bold = _bold(use_tex)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(xs, means, yerr=sems, capsize=4, color="#3a7bd5",
           edgecolor="white", linewidth=1.0, alpha=0.9,
           label=r"mean $\pm$ SEM")
    for d, m in zip(xs, means):
        ax.text(d, m, f"{m:.1f}", ha="center", va="bottom",
                fontsize=10, color="#1f4e79")
    ax.set_xticks(xs)
    ax.set_xlabel(r"Requested difficulty $d_{\mathrm{req}}$")
    ax.set_ylabel(r"Mean distractor count")
    ax.set_title(bold("Distractors per requested difficulty"))
    ax.legend(loc="upper left", frameon=False)
    _save(fig, output_path)
    return True


def plot_distractor_distribution(rows, output_path):
    import matplotlib.pyplot as plt
    counts = [r["distractor_count"] for r in rows]
    if not counts:
        return False
    use_tex = _setup_mpl_style()
    bold = _bold(use_tex)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.arange(-0.5, max(counts) + 1.5, 1)
    ax.hist(counts, bins=bins, color="#4a90e2", edgecolor="white",
            linewidth=1.2, alpha=0.9)
    mu = float(np.mean(counts))
    md = float(np.median(counts))
    ax.axvline(mu, color="#c0392b", linestyle="--", linewidth=1.4,
               label=rf"mean $= {mu:.2f}$")
    ax.axvline(md, color="#27ae60", linestyle=":", linewidth=1.4,
               label=rf"median $= {md:.0f}$")
    ax.set_xlabel(r"Distractor count per question")
    ax.set_ylabel(r"Number of questions")
    ax.set_title(bold("Distractor count distribution"))
    ax.legend(frameon=False)
    _save(fig, output_path)
    return True


def plot_faithfulness(rows, output_path):
    import matplotlib.pyplot as plt
    if not rows:
        return False
    by = defaultdict(list)
    for r in rows:
        d = r.get("requested_difficulty")
        if isinstance(d, int):
            by[d].append(r["claude_faithfulness"])
    xs = sorted(by)
    cov = [np.mean([f["variable_coverage"] for f in by[d]]) for d in xs]
    tgt = [np.mean([f["target_present"] for f in by[d]]) for d in xs]
    full = [np.mean([f["faithful"] for f in by[d]]) for d in xs]

    cov_all = float(np.mean([r["claude_faithfulness"]["variable_coverage"]
                             for r in rows]))
    tgt_all = float(np.mean([r["claude_faithfulness"]["target_present"]
                             for r in rows]))
    full_all = float(np.mean([r["claude_faithfulness"]["faithful"]
                              for r in rows]))

    use_tex = _setup_mpl_style()
    bold = _bold(use_tex)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5),
                                   gridspec_kw={"width_ratios": [1, 1.6]})
    bars = ax1.bar(
        ["variable\ncoverage", "target\npresent", "fully\nfaithful"],
        [cov_all, tgt_all, full_all],
        color=["#3a7bd5", "#27ae60", "#9b59b6"],
        edgecolor="white", linewidth=1.0, alpha=0.92)
    for b, v in zip(bars, [cov_all, tgt_all, full_all]):
        ax1.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=10, color="0.15")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel(r"Rate")
    ax1.set_title(bold("Overall faithfulness (Claude semantic)"))

    ax2.plot(xs, cov, "o-", linewidth=2, label=r"variable coverage",
             color="#3a7bd5")
    ax2.plot(xs, tgt, "s--", linewidth=2, label=r"target present",
             color="#27ae60")
    ax2.plot(xs, full, "^-.", linewidth=2, label=r"fully faithful",
             color="#9b59b6")
    ax2.set_xticks(xs)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel(r"Requested difficulty $d_{\mathrm{req}}$")
    ax2.set_ylabel(r"Rate")
    ax2.set_title(bold("Faithfulness by requested difficulty"))
    ax2.legend(loc="lower left", frameon=False, fontsize=10)
    _save(fig, output_path)
    return True


def plot_chapter_coverage(summary, output_path):
    import matplotlib.pyplot as plt
    cov = summary["chapter_coverage"]
    if not cov:
        return False
    items = sorted(cov.items(), key=lambda kv: kv[1])
    chapters = [k.replace("_", " ") for k, _ in items]
    counts = [v for _, v in items]

    use_tex = _setup_mpl_style()
    bold = _bold(use_tex)
    fig, ax = plt.subplots(figsize=(7.5, max(4, 0.32 * len(chapters) + 2)))
    bars = ax.barh(chapters, counts, color="#1f77b4",
                   edgecolor="white", linewidth=1.0, alpha=0.9)
    for b, v in zip(bars, counts):
        ax.text(v, b.get_y() + b.get_height() / 2, f" {v}",
                va="center", ha="left", fontsize=10, color="0.15")
    ax.set_xlabel(r"Number of eval records touching the chapter")
    ax.set_title(bold("Chapter coverage across eval set"))
    _save(fig, output_path)
    return True


def plot_question_length(rows, output_path):
    import matplotlib.pyplot as plt
    lengths = [r["question_length_words"] for r in rows]
    if not lengths:
        return False
    by = defaultdict(list)
    for r in rows:
        d = r.get("requested_difficulty")
        if isinstance(d, int):
            by[d].append(r["question_length_words"])
    xs = sorted(by)
    means = [np.mean(by[d]) for d in xs]
    sems = [np.std(by[d], ddof=1) / math.sqrt(len(by[d])) if len(by[d]) > 1
            else 0.0 for d in xs]

    use_tex = _setup_mpl_style()
    bold = _bold(use_tex)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.hist(lengths, bins=20, color="#4a90e2", edgecolor="white",
             linewidth=1.2, alpha=0.9)
    mu = float(np.mean(lengths))
    ax1.axvline(mu, color="#c0392b", linestyle="--", linewidth=1.4,
                label=rf"mean $= {mu:.1f}$ words")
    ax1.set_xlabel(r"Words per question")
    ax1.set_ylabel(r"Number of questions")
    ax1.set_title(bold("Question length distribution"))
    ax1.legend(frameon=False)

    ax2.errorbar(xs, means, yerr=sems, fmt="o-", capsize=4,
                 color="#1f4e79", linewidth=2.2, markersize=7,
                 markerfacecolor="#1f77b4", markeredgecolor="white",
                 markeredgewidth=1.0,
                 label=r"mean $\pm$ SEM")
    ax2.set_xticks(xs)
    ax2.set_xlabel(r"Requested difficulty $d_{\mathrm{req}}$")
    ax2.set_ylabel(r"Mean words per question")
    ax2.set_title(bold("Question length vs requested difficulty"))
    ax2.legend(loc="lower right", frameon=False)
    _save(fig, output_path)
    return True


def plot_alignment_summary(summary, output_path):
    """Bar chart of MAE / Pearson r / monotonicity."""
    import matplotlib.pyplot as plt
    da = summary.get("difficulty_alignment", {})
    mae = da.get("mae")
    pr = da.get("pearson_r")
    mono = da.get("monotonicity_per_target")
    if mae is None and pr is None and mono is None:
        return False

    use_tex = _setup_mpl_style()
    bold = _bold(use_tex)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    labels, values, colors = [], [], []
    if pr is not None:
        labels.append(r"Pearson $r$"); values.append(pr); colors.append("#3a7bd5")
    if mono is not None:
        labels.append(r"per-target monotonicity"); values.append(mono); colors.append("#27ae60")
    if mae is not None:
        labels.append(r"MAE / 9 (lower is better)")
        values.append(mae / 9.0); colors.append("#c0392b")

    bars = ax.bar(labels, values, color=colors, edgecolor="white",
                  linewidth=1.0, alpha=0.92)
    for b, v, raw in zip(bars, values,
                         [pr, mono, mae][: len(values)]):
        label = f"{raw:.2f}" if raw is not None else "n/a"
        ax.text(b.get_x() + b.get_width() / 2, v, label,
                ha="center", va="bottom", fontsize=11, color="0.15")
    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.set_ylim(min(0, min(values) - 0.05),
                max(1.05, max(values) + 0.1))
    ax.set_ylabel(r"score (normalised to $[0, 1]$ where applicable)")
    ax.set_title(bold("Difficulty-alignment summary"))
    _save(fig, output_path)
    return True


def _feasibility_summary(rows):
    valid = [r for r in rows if isinstance(r.get("claude_feasibility"), int)
             and r["claude_feasibility"] >= 1]
    if not valid:
        return None
    scores = np.array([r["claude_feasibility"] for r in valid])
    by_d = defaultdict(list)
    for r in valid:
        by_d[r["requested_difficulty"]].append(r["claude_feasibility"])
    return {
        "n_valid": len(valid),
        "mean": float(scores.mean()),
        "median": float(np.median(scores)),
        "frac_infeasible_le_3": float(np.mean(scores <= 3)),
        "frac_feasible_ge_7": float(np.mean(scores >= 7)),
        "by_requested_difficulty": {
            str(d): float(np.mean(v)) for d, v in sorted(by_d.items())
        },
    }


def plot_feasibility(rows, output_path):
    import matplotlib.pyplot as plt
    valid = [r for r in rows if isinstance(r.get("claude_feasibility"), int)
             and r["claude_feasibility"] >= 1]
    if not valid:
        return False
    scores = [r["claude_feasibility"] for r in valid]
    by = defaultdict(list)
    for r in valid:
        by[r["requested_difficulty"]].append(r["claude_feasibility"])
    xs = sorted(by)
    means = [np.mean(by[d]) for d in xs]
    sems = [np.std(by[d], ddof=1) / math.sqrt(len(by[d])) if len(by[d]) > 1
            else 0.0 for d in xs]

    use_tex = _setup_mpl_style()
    bold = _bold(use_tex)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    bins = np.arange(0.5, 11.5, 1)
    ax1.hist(scores, bins=bins, color="#4a90e2", edgecolor="white",
             linewidth=1.2, alpha=0.9)
    mu = float(np.mean(scores))
    md = float(np.median(scores))
    ax1.axvline(mu, color="#c0392b", linestyle="--", linewidth=1.4,
                label=rf"mean $= {mu:.2f}$")
    ax1.axvline(md, color="#27ae60", linestyle=":", linewidth=1.4,
                label=rf"median $= {md:.0f}$")
    ax1.set_xticks(range(1, 11))
    ax1.set_xlim(0.5, 10.5)
    ax1.set_xlabel(r"Claude feasibility score (1--10)")
    ax1.set_ylabel(r"Number of questions")
    ax1.set_title(bold("Physical-feasibility distribution"))
    ax1.legend(frameon=False)

    ax2.errorbar(xs, means, yerr=sems, fmt="o-", capsize=4,
                 color="#1f4e79", linewidth=2.2, markersize=8,
                 markerfacecolor="#1f77b4", markeredgecolor="white",
                 markeredgewidth=1.2,
                 label=r"mean feasibility $\pm$ SEM")
    ax2.axhline(7, color="0.55", linestyle=":", linewidth=1.0,
                label=r"plausible threshold ($\geq 7$)")
    for d, m in zip(xs, means):
        ax2.annotate(f"{m:.1f}", (d, m), textcoords="offset points",
                     xytext=(8, 7), fontsize=10, color="#1f4e79")
    ax2.set_xticks(xs)
    ax2.set_yticks(range(1, 11))
    ax2.set_ylim(0.5, 10.5)
    ax2.set_xlabel(r"Requested difficulty $d_{\mathrm{req}}$")
    ax2.set_ylabel(r"Mean feasibility")
    ax2.set_title(bold("Feasibility vs requested difficulty"))
    ax2.legend(loc="lower left", frameon=False, fontsize=10)
    _save(fig, output_path)
    return True


def plot_rl_training_metrics(rows: List[Dict], output_path):
    """Training curves for policy-gradient RL (same matplotlib style as ``plot_feasibility``).

    Expects JSONL rows with ``reward_raw`` / ``rewards`` (scaled in ``[-1,1]``) or
    legacy ``faithfulness_raw`` / ``mean_faithfulness_raw``. Optional ``reward_kind``
    is ``faithfulness`` or ``feasibility`` (for axis titles).
    """
    import matplotlib.pyplot as plt

    if not rows:
        return False

    first = rows[0]
    reward_kind = str(first.get("reward_kind", "faithfulness"))

    def _raw_list(r: Dict) -> List[float]:
        if "reward_raw" in r and r["reward_raw"] is not None:
            return [float(x) for x in r["reward_raw"]]
        fr = r.get("faithfulness_raw")
        if fr is not None:
            return [float(x) for x in fr]
        return []

    def _mean_raw(r: Dict) -> float:
        if "mean_reward_raw" in r:
            return float(r["mean_reward_raw"])
        if "mean_faithfulness_raw" in r:
            return float(r["mean_faithfulness_raw"])
        return float(r["mean_reward"])

    legacy = "reward_raw" not in first and "faithfulness_raw" not in first
    all_raw: List[float] = []
    all_scaled: List[float] = []
    for r in rows:
        rs = r.get("rewards") or []
        if not legacy:
            all_raw.extend(_raw_list(r))
            all_scaled.extend(float(x) for x in rs)
        else:
            all_raw.extend(float(x) for x in rs)
            all_scaled.extend(float(x) - 1.0 for x in rs)

    if not all_raw and not all_scaled:
        return False

    steps_r = [int(r["step"]) for r in rows]
    if not legacy:
        means_raw = [_mean_raw(r) for r in rows]
        means_scaled = [
            float(r.get("mean_reward_scaled", r["mean_reward"])) for r in rows
        ]
    else:
        means_raw = [float(r["mean_reward"]) for r in rows]
        means_scaled = [m - 1.0 for m in means_raw]

    pairs = [(int(r["step"]), float(r["loss"]))
              for r in rows
              if r.get("loss") is not None]
    use_tex = _setup_mpl_style()
    bold = _bold(use_tex)

    if reward_kind == "feasibility":
        raw_xlabel = r"Raw score (feasibility $1$--$10$ $\rightarrow$ $[0,2]$)"
        raw_title = "Raw feasibility (all rollouts)"
        mean_raw_ylabel = r"Mean raw (feasibility-mapped)"
        mean_raw_plot_title = bold("Mean raw feasibility vs step")
        supt = (
            f"RL training ({reward_kind}): raw, scaled reward, and PG loss "
            f"({len(rows)} steps, {len(all_scaled)} rollouts)"
        )
    else:
        raw_xlabel = r"Raw faithfulness score (Claude composite, $\approx [0,2]$)"
        raw_title = "Raw faithfulness (all rollouts)"
        mean_raw_ylabel = r"Mean raw faithfulness"
        mean_raw_plot_title = bold("Mean raw faithfulness vs step")
        supt = (
            f"RL training ({reward_kind}): raw, scaled reward, and PG loss "
            f"({len(rows)} steps, {len(all_scaled)} rollouts)"
        )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.8))

    ax00 = axes[0, 0]
    ax00.hist(
        all_raw,
        bins=24,
        color="#4a90e2",
        edgecolor="white",
        linewidth=1.2,
        alpha=0.9,
    )
    mu_r = float(np.mean(all_raw))
    md_r = float(np.median(all_raw))
    ax00.axvline(mu_r, color="#c0392b", linestyle="--", linewidth=1.4,
                 label=rf"mean $= {mu_r:.3f}$")
    ax00.axvline(md_r, color="#27ae60", linestyle=":", linewidth=1.4,
                 label=rf"median $= {md_r:.3f}$")
    ax00.set_xlim(-0.05, 2.15)
    ax00.set_xlabel(raw_xlabel)
    ax00.set_ylabel(r"Number of rollouts")
    ax00.set_title(bold(raw_title))
    ax00.legend(frameon=False)

    ax01 = axes[0, 1]
    ax01.hist(
        all_scaled,
        bins=28,
        color="#2e86ab",
        edgecolor="white",
        linewidth=1.2,
        alpha=0.9,
    )
    mu_s = float(np.mean(all_scaled))
    md_s = float(np.median(all_scaled))
    ax01.axvline(mu_s, color="#c0392b", linestyle="--", linewidth=1.4,
                 label=rf"mean $= {mu_s:.3f}$")
    ax01.axvline(md_s, color="#27ae60", linestyle=":", linewidth=1.4,
                 label=rf"median $= {md_s:.3f}$")
    ax01.set_xlim(-1.05, 1.05)
    ax01.set_xlabel(r"Scaled PG reward (linear map to $[-1,1]$)")
    ax01.set_ylabel(r"Number of rollouts")
    ax01.set_title(bold("Scaled reward (PG / advantage signal)"))
    ax01.legend(frameon=False)

    ax10 = axes[1, 0]
    ax10.plot(
        steps_r,
        means_raw,
        color="#1f4e79",
        linewidth=2.0,
        label=r"mean raw / step",
    )
    if len(means_raw) >= 8:
        w = min(31, max(5, len(means_raw) // 10))
        ker = np.ones(w, dtype=float) / w
        sm = np.convolve(means_raw, ker, mode="same")
        ax10.plot(
            steps_r,
            sm,
            color="#7fb3d5",
            linewidth=1.5,
            linestyle="--",
            label=rf"rolling mean ($w={w}$)",
        )
    ax10.set_xlabel(r"Training step")
    ax10.set_ylabel(mean_raw_ylabel)
    ax10.set_title(mean_raw_plot_title)
    ax10.legend(loc="lower right", frameon=False, fontsize=9)

    ax11 = axes[1, 1]
    ax11.plot(
        steps_r,
        means_scaled,
        color="#1f4e79",
        linewidth=2.0,
        label=r"mean scaled / step",
    )
    if len(means_scaled) >= 8:
        w2 = min(31, max(5, len(means_scaled) // 10))
        ker2 = np.ones(w2, dtype=float) / w2
        sm2 = np.convolve(means_scaled, ker2, mode="same")
        ax11.plot(
            steps_r,
            sm2,
            color="#7fb3d5",
            linewidth=1.5,
            linestyle="--",
            label=rf"rolling mean ($w={w2}$)",
        )
    ax11.set_xlabel(r"Training step")
    ax11.set_ylabel(r"Mean scaled reward ($[-1,1]$)")
    ax11.set_title(bold("Mean scaled reward vs step"))

    if pairs:
        st_l, ls = zip(*pairs)
        ax11_t = ax11.twinx()
        ax11_t.plot(
            st_l,
            ls,
            color="#c0392b",
            linewidth=1.8,
            alpha=0.95,
            label=r"PG loss ($-\hat A \sum \log \pi$)",
        )
        ax11_t.set_ylabel(r"PG loss")
        ax11_t.spines["top"].set_visible(False)
        h1, l1 = ax11.get_legend_handles_labels()
        h2, l2 = ax11_t.get_legend_handles_labels()
        ax11.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False, fontsize=9)
    else:
        ax11.legend(loc="lower right", frameon=False, fontsize=9)

    fig.suptitle(supt, fontsize=13, y=0.995)
    _save(fig, output_path)
    return True


def plot_rl_combined_training_metrics(rows: List[Dict], output_path):
    """RL training curves when ``reward_kind`` is ``combined``: faithfulness raw,
    feasibility raw (mapped to ``[0,2]``), combined raw, scaled combined reward,
    and PG loss. Same matplotlib style as ``plot_feasibility``."""
    import matplotlib.pyplot as plt

    if not rows or str(rows[0].get("reward_kind")) != "combined":
        return False

    def _roll(y: List[float], w: int) -> List[float]:
        if len(y) < w or w < 2:
            return y
        ker = np.ones(w, dtype=float) / w
        return list(np.convolve(np.array(y, dtype=float), ker, mode="same"))

    steps = [int(r["step"]) for r in rows]
    all_f: List[float] = []
    all_e: List[float] = []
    all_c: List[float] = []
    for r in rows:
        all_f.extend(float(x) for x in (r.get("faithfulness_raw_rollout") or []))
        all_e.extend(float(x) for x in (r.get("feasibility_raw_rollout") or []))
        all_c.extend(float(x) for x in (r.get("combined_raw_rollout") or r.get("reward_raw") or []))

    if not all_c and not all_f and not all_e:
        return False

    mf = [float(r.get("mean_faithfulness_raw", 0.0)) for r in rows]
    me = [float(r.get("mean_feasibility_raw", 0.0)) for r in rows]
    m_scaled = [float(r.get("mean_reward_scaled", r["mean_reward"])) for r in rows]
    pairs = [(int(r["step"]), float(r["loss"]))
              for r in rows if r.get("loss") is not None]

    use_tex = _setup_mpl_style()
    bold = _bold(use_tex)
    w0 = rows[0].get("combined_faith_weight", 0.5)
    w1 = rows[0].get("combined_feas_weight", 0.5)

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.8))

    def _hist(ax, data, title, xlabel, color="#4a90e2"):
        if not data:
            ax.set_visible(False)
            return
        ax.hist(data, bins=22, color=color, edgecolor="white", linewidth=1.1, alpha=0.9)
        mu, md = float(np.mean(data)), float(np.median(data))
        ax.axvline(mu, color="#c0392b", linestyle="--", linewidth=1.3, label=rf"mean $= {mu:.3f}$")
        ax.axvline(md, color="#27ae60", linestyle=":", linewidth=1.3, label=rf"median $= {md:.3f}$")
        ax.set_xlim(-0.05, 2.15)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Rollouts")
        ax.set_title(bold(title))
        ax.legend(frameon=False, fontsize=8)

    _hist(
        axes[0, 0],
        all_f,
        "Faithfulness raw",
        r"Faithfulness composite $\approx [0,2]$",
        "#4a90e2",
    )
    _hist(
        axes[0, 1],
        all_e,
        "Feasibility raw",
        r"Feasibility ($1$--$10$) $\rightarrow [0,2]$",
        "#27ae60",
    )
    _hist(
        axes[0, 2],
        all_c,
        "Combined raw (PG)",
        rf"Weighted sum ($w_f={w0:.2f}$, $w_e={w1:.2f}$) in $[0,2]$",
        "#8e44ad",
    )

    def _line(ax, ys, ylab, ttl, color="#1f4e79"):
        ax.plot(steps, ys, color=color, linewidth=2.0, label=r"mean / step")
        if len(ys) >= 8:
            w = min(31, max(5, len(ys) // 10))
            ax.plot(steps, _roll(ys, w), color="#7fb3d5", linewidth=1.4, linestyle="--",
                    label=rf"rolling ($w={w}$)")
        ax.set_xlabel(r"Training step")
        ax.set_ylabel(ylab)
        ax.set_title(bold(ttl))
        ax.legend(loc="lower right", frameon=False, fontsize=8)

    _line(axes[1, 0], mf, r"Mean faithfulness raw", "Faithfulness vs step")
    _line(axes[1, 1], me, r"Mean feasibility raw", "Feasibility vs step", "#1d6f42")

    axc = axes[1, 2]
    axc.plot(steps, m_scaled, color="#1f4e79", linewidth=2.0, label=r"mean scaled combined")
    if len(m_scaled) >= 8:
        w2 = min(31, max(5, len(m_scaled) // 10))
        axc.plot(steps, _roll(m_scaled, w2), color="#7fb3d5", linewidth=1.4, linestyle="--",
                 label=rf"rolling ($w={w2}$)")
    axc.set_xlabel(r"Training step")
    axc.set_ylabel(r"Mean scaled combined ($[-1,1]$)")
    axc.set_title(bold("Combined reward (PG signal) vs step"))
    if pairs:
        st_l, ls = zip(*pairs)
        ax_t = axc.twinx()
        ax_t.plot(st_l, ls, color="#c0392b", linewidth=1.7, alpha=0.95, label=r"PG loss")
        ax_t.set_ylabel(r"PG loss")
        ax_t.spines["top"].set_visible(False)
        h1, l1 = axc.get_legend_handles_labels()
        h2, l2 = ax_t.get_legend_handles_labels()
        axc.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False, fontsize=8)
    else:
        axc.legend(loc="lower right", frameon=False, fontsize=8)

    fig.suptitle(
        bold("RL combined: faithfulness, feasibility, and weighted PG reward")
        + f"\n({len(rows)} steps, {len(all_c)} rollouts)",
        fontsize=12,
        y=0.995,
    )
    _save(fig, output_path)
    return True


def make_all_plots(rows, summary, plot_dir):
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    plans = [
        ("difficulty_alignment.png",
         lambda p: plot_difficulty_alignment(rows, p)),
        ("alignment_summary.png",
         lambda p: plot_alignment_summary(summary, p)),
        ("confusion_10x10.png",
         lambda p: plot_confusion(
             summary["confusion_raw_10x10"], p,
             title=r"Confusion matrix (1--10 raw)",
             xlabel=r"Claude-judged difficulty",
             ylabel=r"Requested difficulty")),
        ("confusion_3x3.png",
         lambda p: plot_confusion(
             summary["confusion_bucket_3x3"], p,
             title=r"Confusion matrix (easy / medium / hard)",
             xlabel=r"Claude-judged bucket",
             ylabel=r"Requested bucket")),
        ("distractors_by_difficulty.png",
         lambda p: plot_distractors_by_difficulty(rows, p)),
        ("distractor_distribution.png",
         lambda p: plot_distractor_distribution(rows, p)),
        ("faithfulness.png",
         lambda p: plot_faithfulness(rows, p)),
        ("feasibility.png",
         lambda p: plot_feasibility(rows, p)),
        ("chapter_coverage.png",
         lambda p: plot_chapter_coverage(summary, p)),
        ("question_length.png",
         lambda p: plot_question_length(rows, p)),
    ]
    for name, fn in plans:
        path = plot_dir / name
        try:
            ok = fn(path)
        except Exception as e:
            print(f"[warn] {name} failed: {e}")
            ok = False
        if ok:
            written[name] = str(path)
    return written


# ----- helpers ---------------------------------------------------------------

def tokens(text: str):
    return set(re.findall(r"[a-zA-Z]+", text.lower()))


def variable_keywords(var: str):
    """Words that should plausibly appear in the question for this variable.

    e.g. 'final_velocity' -> {'final', 'velocity'} ; 'mass1' -> {'mass'}
    """
    parts = re.findall(r"[a-zA-Z]+", var)
    return {p.lower() for p in parts if len(p) > 2}


def bucket(score: int) -> int:
    if score <= 3: return 0
    if score <= 6: return 1
    return 2


# ----- per-record metrics ----------------------------------------------------

def distractor_count(item: Dict, all_vars: set):
    """Distinct physics keyword tokens that appear in the question but are
    not part of the trace's expected variables (target + leafs + intermediates).
    """
    toks = tokens(item["question"])
    expected = set(item["trace"]["leafs"]) | {item["target"]} | {
        e["output"] for e in item["trace"]["path"]
    } | {i for e in item["trace"]["path"] for i in e["inputs"]}
    physics_kw = {kw for v in all_vars for kw in variable_keywords(v)}
    expected_kw = set()
    for v in expected:
        expected_kw |= variable_keywords(v)
    extras = sorted((toks & physics_kw) - expected_kw)
    return len(extras), extras


# ----- aggregate -------------------------------------------------------------

def confusion(items: List[Dict], bucketed: bool):
    if bucketed:
        n = 3
        labels = ["easy(1-3)", "med(4-6)", "hard(7-10)"]
        idx = bucket
    else:
        n = 10
        labels = [str(i) for i in range(1, 11)]
        idx = lambda x: max(0, min(9, x - 1))
    cm = np.zeros((n, n), int)
    for it in items:
        if it["claude_difficulty"] < 1 or it["requested_difficulty"] < 1:
            continue
        cm[idx(it["requested_difficulty"]), idx(it["claude_difficulty"])] += 1
    return {"labels": labels, "matrix": cm.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "data" / "eval_results.json"))
    ap.add_argument("--graph", default=str(HERE / "data" / "physics_hypergraph.json"))
    ap.add_argument("--output", default=str(HERE / "data" / "metrics.json"))
    ap.add_argument("--plot_dir", default=str(HERE / "data" / "plots"))
    args = ap.parse_args()

    g = HyperGraph(args.graph)
    all_vars = set(g.nodes)

    with open(args.results) as f:
        items = json.load(f)
    if not items:
        raise SystemExit("eval_results.json is empty - run eval_pipeline.py first.")

    rows = []
    for it in items:
        n_dist, dist_list = distractor_count(it, all_vars)
        rows.append({**it,
                     "distractor_count": n_dist,
                     "distractor_variables": dist_list,
                     "question_length_words": len(it["question"].split())})

    valid = [r for r in rows if r["claude_difficulty"] >= 1]
    req = np.array([r["requested_difficulty"] for r in valid])
    got = np.array([r["claude_difficulty"] for r in valid])

    # monotonicity per target
    by_target = defaultdict(list)
    for r in rows:
        by_target[r["target"]].append((r["requested_difficulty"], r["claude_difficulty"]))
    mono = []
    for tgt, pairs in by_target.items():
        pairs.sort()
        scores = [p[1] for p in pairs if p[1] >= 1]
        if len(scores) >= 2:
            ups = sum(scores[i+1] >= scores[i] for i in range(len(scores)-1))
            mono.append(ups / (len(scores)-1))
    monotonicity = float(np.mean(mono)) if mono else None

    summary = {
        "n_records": len(rows),
        "n_valid_scores": len(valid),
        "confusion_raw_10x10": confusion(items, bucketed=False),
        "confusion_bucket_3x3": confusion(items, bucketed=True),
        "faithfulness": {
            "mean_variable_coverage": float(np.mean(
                [r["claude_faithfulness"]["variable_coverage"] for r in rows])),
            "target_present_rate": float(np.mean(
                [r["claude_faithfulness"]["target_present"] for r in rows])),
            "fully_faithful_rate": float(np.mean(
                [r["claude_faithfulness"]["faithful"] for r in rows])),
        },
        "distractors": {
            "mean": float(np.mean([r["distractor_count"] for r in rows])),
            "median": float(np.median([r["distractor_count"] for r in rows])),
            "max": int(np.max([r["distractor_count"] for r in rows])),
            "by_requested_difficulty": {
                str(d): float(np.mean([r["distractor_count"]
                                       for r in rows
                                       if r["requested_difficulty"] == d]))
                for d in sorted({r["requested_difficulty"] for r in rows})
            },
        },
        "difficulty_alignment": {
            "mae": float(np.mean(np.abs(req - got))) if len(valid) else None,
            "pearson_r": float(np.corrcoef(req, got)[0, 1])
            if len(valid) > 1 and req.std() > 0 and got.std() > 0 else None,
            "monotonicity_per_target": monotonicity,
            "mean_score_by_requested": {
                str(d): float(np.mean([r["claude_difficulty"]
                                       for r in valid
                                       if r["requested_difficulty"] == d]))
                for d in sorted(set(req.tolist()))
            },
        },
        "feasibility": _feasibility_summary(rows),
        "chapter_coverage": dict(Counter(
            ch for r in rows for ch in r["trace"]["chapters"])),
        "avg_question_length_words": float(np.mean(
            [r["question_length_words"] for r in rows])),
    }

    with open(args.output, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote per-record details + summary to {args.output}")

    written = make_all_plots(rows, summary, args.plot_dir)
    if written:
        print(f"\nWrote {len(written)} plots to {args.plot_dir}/:")
        for name, path in written.items():
            print(f"  - {name}")


if __name__ == "__main__":
    main()
