"""Generate a (prompt, response, difficulty) dataset using Claude.

For each randomly chosen target we build a solution trace and ask Claude to
write a physics question at multiple requested difficulties. By default the
generator must first emit chain-of-thought inside <reasoning>...</reasoning>,
then a JSON array of {difficulty, question} objects; use --no-cot for the
legacy JSON-only output. Accepted rows may store chain_of_thought in the
output JSON. Claude judges (feasibility, faithfulness) still run separately.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

from claude_client import ClaudeClient
from judges import judge_faithfulness, judge_feasibility
from prompts import (
    SYSTEM_GEN,
    SYSTEM_GEN_WITH_COT,
    USER_GEN,
    USER_GEN_WITH_COT,
    format_trace,
)
from traversal import HyperGraph

HERE = Path(__file__).parent
DATA = HERE / "data"


_OBJ_RE = re.compile(
    r'\{\s*"difficulty"\s*:\s*(\d+)\s*,\s*"question"\s*:\s*"((?:\\.|[^"\\])*)"\s*\}',
    re.DOTALL,
)

_REASONING_RE = re.compile(
    r"<reasoning>\s*(.*?)\s*</reasoning>",
    re.IGNORECASE | re.DOTALL,
)


def extract_reasoning(text: str) -> tuple[str, str]:
    """Strip optional <reasoning>...</reasoning> block; return (cot, rest)."""
    m = _REASONING_RE.search(text)
    if not m:
        return "", text
    return m.group(1).strip(), (text[: m.start()] + text[m.end() :]).strip()


def _validate(items):
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        d = it.get("difficulty")
        q = it.get("question")
        if isinstance(d, int) and 1 <= d <= 10 and isinstance(q, str) and q.strip():
            out.append((d, q.strip()))
    return out


def parse_question_list(text: str) -> tuple[list[tuple[int, str]], str]:
    """Extract a JSON array of {difficulty, question} objects from Claude's
    output. Tolerates code fences, leading/trailing prose, a leading
    <reasoning>...</reasoning> block, and a truncated final element by
    falling back to per-object regex extraction.

    Returns (pairs, chain_of_thought_text). chain_of_thought_text is empty
    if no <reasoning> block was present.
    """
    cot, body = extract_reasoning(text.strip())
    s = body.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)

    lo, hi = s.find("["), s.rfind("]")
    if lo != -1 and hi != -1 and hi > lo:
        try:
            return _validate(json.loads(s[lo:hi + 1])), cot
        except json.JSONDecodeError:
            pass

    out = []
    for m in _OBJ_RE.finditer(s):
        try:
            d = int(m.group(1))
            q = json.loads(f'"{m.group(2)}"')
        except (ValueError, json.JSONDecodeError):
            continue
        if 1 <= d <= 10 and q.strip():
            out.append((d, q.strip()))
    return out, cot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=str(DATA / "physics_hypergraph.json"))
    ap.add_argument("--output", default=str(DATA / "dataset.json"))
    ap.add_argument("--num_targets", type=int, default=80,
                    help="number of (target, trace) instances")
    ap.add_argument("--difficulties", type=int, nargs="+",
                    default=[2, 5, 8])
    ap.add_argument("--max_depth", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_trace", type=int, default=1)
    ap.add_argument("--single_domain", action="store_true", default=True,
                    help="restrict every trace to one physics chapter")
    ap.add_argument("--no_single_domain", dest="single_domain",
                    action="store_false")
    ap.add_argument("--min_feasibility", type=int, default=7,
                    help="drop questions whose Claude feasibility score is below this (1-10)")
    ap.add_argument("--min_coverage", type=float, default=1.0,
                    help="drop questions whose Claude leaf-coverage is below this (0..1)")
    ap.add_argument("--require_target_present", action="store_true", default=True,
                    help="drop questions where the Claude judge does not see the target")
    ap.add_argument("--no_require_target_present", dest="require_target_present",
                    action="store_false")
    ap.add_argument("--max_regens", type=int, default=2,
                    help="max regeneration attempts per (target, trace) when a quality filter fails")
    ap.add_argument(
        "--no-cot",
        action="store_true",
        help="disable chain-of-thought: use plain JSON-only generator prompt",
    )
    args = ap.parse_args()
    use_cot = not args.no_cot
    print(
        f"Chain-of-thought prompting: {'on (default)' if use_cot else 'off (--no-cot)'}",
        flush=True,
    )

    rng = random.Random(args.seed)
    g = HyperGraph(args.graph)
    client = ClaudeClient()

    candidates = g.producible()
    out = []
    seen_targets = []
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    sys_gen = SYSTEM_GEN_WITH_COT if use_cot else SYSTEM_GEN
    usr_gen = USER_GEN_WITH_COT if use_cot else USER_GEN

    def gen_pairs(trace_str, target, leafs, domain, difficulties):
        """One Claude call -> (pairs, raw, chain_of_thought)."""
        raw = client.complete(
            sys_gen,
            usr_gen.format(
                trace_str=trace_str,
                target=target,
                leafs=", ".join(leafs) or "(none)",
                domain=domain,
                difficulties=difficulties,
            ),
            max_tokens=8192,
            temperature=0.9,
        )
        pairs, cot = parse_question_list(raw)
        return pairs, raw, cot

    i = 0
    attempts = 0
    while i < args.num_targets and attempts < args.num_targets * 5:
        attempts += 1
        target = rng.choice(candidates)
        trace = g.traverse(target, max_depth=args.max_depth,
                           seed=rng.randint(0, 10**9),
                           single_domain=args.single_domain)
        if trace is None or len(trace["path"]) < args.min_trace:
            continue
        if args.single_domain and (len(trace["chapters"]) != 1
                                   or len(trace.get("subdomains", [])) > 1):
            continue  # belt-and-braces: enforce single chapter and subdomain
        i += 1
        seen_targets.append(target)
        trace_str = format_trace(trace)
        domain = trace["chapters"][0] if trace["chapters"] else trace["hypernym"]
        sub = (trace.get("subdomains") or [domain])[0]
        print(f"[{i}/{args.num_targets}] target={target} steps={len(trace['path'])} "
              f"domain={domain} sub={sub}", flush=True)

        # Per-difficulty regeneration loop driven by feasibility AND coverage.
        remaining = list(args.difficulties)
        kept = {}  # difficulty -> dict(question, feasibility, faithfulness)
        for attempt in range(args.max_regens + 1):
            if not remaining:
                break
            try:
                pairs, raw, cot = gen_pairs(
                    trace_str, target, trace["leafs"], domain, remaining
                )
            except Exception as e:
                print(f"  GEN FAILED: {e}", file=sys.stderr)
                break
            if not pairs:
                print("  GEN FAILED: could not parse JSON list", file=sys.stderr)
                print("  --- raw response (first 400 chars) ---", file=sys.stderr)
                print(raw[:400], file=sys.stderr)
                print("  --- end ---", file=sys.stderr)
                break
            for d, question in pairs:
                if d not in remaining:
                    continue
                try:
                    feas = judge_feasibility(client, trace_str,
                                             trace["leafs"], target, question)
                except Exception as e:
                    print(f"  d={d} FEAS JUDGE FAILED: {e}", file=sys.stderr)
                    feas = -1
                try:
                    faith = judge_faithfulness(client, trace_str,
                                               trace["leafs"], target, question)
                except Exception as e:
                    print(f"  d={d} FAITH JUDGE FAILED: {e}", file=sys.stderr)
                    faith = {"variable_coverage": 0.0,
                             "target_present": False,
                             "leaf_hits": {},
                             "faithful": False}
                cov = faith["variable_coverage"]
                tgt_ok = faith["target_present"] or not args.require_target_present
                feas_ok = feas >= args.min_feasibility
                cov_ok = cov >= args.min_coverage
                if feas_ok and cov_ok and tgt_ok:
                    kept[d] = {
                        "question": question,
                        "feasibility": feas,
                        "faithfulness": faith,
                        "chain_of_thought": cot if use_cot else "",
                    }
                    remaining.remove(d)
                    print(f"  d={d} kept "
                          f"(feasibility={feas}, coverage={cov:.2f}, "
                          f"target_present={faith['target_present']})",
                          flush=True)
                else:
                    reason = []
                    if not feas_ok:
                        reason.append(f"feas={feas}<{args.min_feasibility}")
                    if not cov_ok:
                        reason.append(f"cov={cov:.2f}<{args.min_coverage}")
                    if not tgt_ok:
                        reason.append("target_missing")
                    print(f"  d={d} dropped ({', '.join(reason)}), "
                          f"attempt {attempt + 1}", flush=True)
            if remaining and attempt < args.max_regens:
                print(f"  regenerating difficulties {remaining}", flush=True)

        if remaining:
            print(f"  giving up on difficulties {remaining} after "
                  f"{args.max_regens + 1} attempts", flush=True)

        for d in sorted(kept):
            rec = kept[d]
            row = {
                "target": target,
                "trace": {
                    "target": trace["target"],
                    "leafs": trace["leafs"],
                    "chapters": trace["chapters"],
                    "subdomains": trace.get("subdomains", trace["chapters"]),
                    "hypernym": trace["hypernym"],
                    "path": [{"id": e["id"], "output": e["output"],
                              "inputs": e["inputs"], "label": e["label"],
                              "domain": e["domain"],
                              "subdomain": e.get("subdomain", e["domain"])}
                             for e in trace["path"]],
                },
                "trace_str": trace_str,
                "domain": domain,
                "subdomain": sub,
                "requested_difficulty": d,
                "claude_difficulty": d,
                "claude_feasibility": rec["feasibility"],
                "claude_faithfulness": rec["faithfulness"],
                "question": rec["question"],
            }
            cot_text = (rec.get("chain_of_thought") or "").strip()
            if cot_text:
                row["chain_of_thought"] = cot_text
            out.append(row)
        tmp = Path(args.output).with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(out, f, indent=2)
        tmp.replace(args.output)

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {len(out)} (prompt, response, difficulty) records to {args.output}")


if __name__ == "__main__":
    main()
