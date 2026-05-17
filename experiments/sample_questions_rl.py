"""Generate sample physics questions with an RL (or SFT) LoRA checkpoint.

Samples random solution traces from the physics hypergraph (same traversal
semantics as ``eval_pipeline.py``), builds the same chat prompts as training /
eval, and runs the merged model to produce questions. No Claude calls unless
you pass ``--with-judges``.

Example::

    cd experiments
    python sample_questions_rl.py \\
        --sft_dir /mnt/storage/ae21b026/rl_lora \\
        --num-targets 5 --difficulties 2 5 8 \\
        --output data/rl/sample_questions.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from cot_utils import strip_reasoning_prefix
from eval_pipeline import generate_question, load_model, tokens_for_difficulty
from prompts import build_sft_chat_messages, format_trace
from traversal import HyperGraph

HERE = Path(__file__).parent
DATA = HERE / "data"


def _trace_to_json(trace: dict) -> dict:
    return {
        "leafs": trace["leafs"],
        "chapters": trace["chapters"],
        "subdomains": trace.get("subdomains", trace["chapters"]),
        "hypernym": trace.get("hypernym"),
        "path": [
            {
                "id": e["id"],
                "output": e["output"],
                "inputs": e["inputs"],
                "label": e["label"],
                "domain": e["domain"],
            }
            for e in trace["path"]
        ],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", default=str(DATA / "physics_hypergraph.json"))
    ap.add_argument(
        "--sft_dir",
        default="/mnt/storage/ae21b026/rl_lora",
        help="LoRA adapter (RL or SFT); merged for generation like eval_pipeline.",
    )
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument(
        "--output",
        default=str(DATA / "rl" / "sample_questions.json"),
        help="JSON file with traces + model outputs.",
    )
    ap.add_argument("--num-targets", type=int, default=5,
                    help="How many distinct random traces to sample.")
    ap.add_argument(
        "--difficulties",
        type=int,
        nargs="+",
        default=[2, 5, 8],
        help="Requested difficulty per trace (default: 2 5 8).",
    )
    ap.add_argument("--max_depth", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--exclude_targets_from",
        default="",
        help="Optional JSON (e.g. dataset.json) whose `target` keys are skipped.",
    )
    ap.add_argument("--gen_tokens_base", type=int, default=350)
    ap.add_argument("--gen_tokens_scale", type=int, default=80)
    ap.add_argument("--gen_temperature", type=float, default=0.6)
    ap.add_argument("--gen_top_p", type=float, default=0.95)
    ap.add_argument(
        "--with-cot",
        action="store_true",
        help="Match CoT SFT/RL prompts (<reasoning> then question).",
    )
    ap.add_argument("--single-domain", action="store_true", default=True)
    ap.add_argument("--no-single-domain", dest="single_domain", action="store_false")
    ap.add_argument(
        "--with-judges",
        action="store_true",
        help="Call Claude difficulty / faithfulness / feasibility (needs API key).",
    )
    ap.add_argument(
        "--print",
        action="store_true",
        help="Pretty-print each generated question to stdout.",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    g = HyperGraph(args.graph)
    pool = g.producible()
    if args.exclude_targets_from and Path(args.exclude_targets_from).exists():
        with open(args.exclude_targets_from) as f:
            ex = {it["target"] for it in json.load(f)}
        pool = [c for c in pool if c not in ex] or pool

    model, tok = load_model(args.base_model, args.sft_dir)
    judges = None
    if args.with_judges:
        from claude_client import ClaudeClient
        from judges import judge_difficulty, judge_faithfulness, judge_feasibility

        judges = ClaudeClient()

    records = []
    n, attempts = 0, 0
    while n < args.num_targets and attempts < args.num_targets * 8:
        attempts += 1
        target = rng.choice(pool)
        trace = g.traverse(
            target,
            max_depth=args.max_depth,
            seed=rng.randint(0, 10**9),
            single_domain=args.single_domain,
        )
        if trace is None or not trace["path"]:
            continue
        if args.single_domain and (
            len(trace["chapters"]) != 1
            or len(trace.get("subdomains", [])) > 1
        ):
            continue
        n += 1
        trace_str = format_trace(trace)
        chapter = trace["chapters"][0] if trace["chapters"] else ""
        sub = (trace.get("subdomains") or [chapter])[0]

        for d in args.difficulties:
            messages = build_sft_chat_messages(
                trace_str,
                target,
                trace["leafs"],
                d,
                domain=chapter,
                subdomain=sub,
                expect_chain_of_thought=args.with_cot,
            )
            mnt = tokens_for_difficulty(d, args.gen_tokens_base, args.gen_tokens_scale)
            if args.with_cot:
                mnt += 512
            raw_out = generate_question(
                model,
                tok,
                messages,
                max_new_tokens=mnt,
                temperature=args.gen_temperature,
                top_p=args.gen_top_p,
            )
            question = strip_reasoning_prefix(raw_out)
            rec = {
                "target": target,
                "requested_difficulty": d,
                "trace_str": trace_str,
                "trace": _trace_to_json(trace),
                "raw_model_output": raw_out,
                "question": question,
            }
            if judges:
                rec["claude_difficulty"] = judge_difficulty(
                    judges, trace_str, trace["leafs"], target, question
                )
                rec["claude_faithfulness"] = judge_faithfulness(
                    judges, trace_str, trace["leafs"], target, question
                )
                rec["claude_feasibility"] = judge_feasibility(
                    judges, trace_str, trace["leafs"], target, question
                )
            records.append(rec)
            if args.print:
                print(
                    f"\n--- target={target} d={d} chapter={chapter}/{sub} ---\n",
                    question,
                    "\n",
                    flush=True,
                )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "meta": {
                    "graph": args.graph,
                    "sft_dir": args.sft_dir,
                    "base_model": args.base_model,
                    "num_targets_sampled": n,
                    "difficulties": list(args.difficulties),
                    "with_cot": args.with_cot,
                    "with_judges": args.with_judges,
                    "seed": args.seed,
                },
                "samples": records,
            },
            f,
            indent=2,
        )
    print(f"Wrote {len(records)} samples to {out_path}", flush=True)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
