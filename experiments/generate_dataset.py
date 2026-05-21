"""Generate a (prompt, response, difficulty) dataset using an LLM (Claude or OpenAI).

For each randomly chosen target we build a solution trace and ask Claude to
write one physics question per API call (one requested difficulty at a time). By default the
generator must first emit chain-of-thought inside <reasoning>...</reasoning>,
then a JSON array of {difficulty, question} objects; use --no-cot for the
legacy JSON-only output. Accepted rows may store chain_of_thought in the
output JSON. Feasibility and faithfulness judges run in parallel per question.
Failures are appended to ``<output_stem>_failed.json`` (or ``--failed-output``) with scores when available.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from judges import judge_faithfulness, judge_feasibility
from llm_client import add_llm_cli, describe_llm_client, llm_client_from_args
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


def _faith_judge_failed() -> dict:
    return {
        "variable_coverage": 0.0,
        "target_present": False,
        "leaf_hits": {},
        "faithful": False,
    }


def judge_feasibility_and_faithfulness(
    client,
    trace_str: str,
    leafs: list,
    target: str,
    question: str,
) -> tuple[int, dict]:
    """Run feasibility and faithfulness judges concurrently (two API calls)."""
    feas: int = -1
    faith: dict = _faith_judge_failed()

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_feas = pool.submit(
            judge_feasibility, client, trace_str, leafs, target, question
        )
        fut_faith = pool.submit(
            judge_faithfulness, client, trace_str, leafs, target, question
        )
        try:
            feas = fut_feas.result()
        except Exception as e:
            print(f"  FEAS JUDGE FAILED: {e}", file=sys.stderr)
        try:
            faith = fut_faith.result()
        except Exception as e:
            print(f"  FAITH JUDGE FAILED: {e}", file=sys.stderr)

    return feas, faith


def _trace_json(trace: dict) -> dict:
    return {
        "target": trace["target"],
        "leafs": trace["leafs"],
        "chapters": trace["chapters"],
        "subdomains": trace.get("subdomains", trace["chapters"]),
        "hypernym": trace["hypernym"],
        "path": [
            {
                "id": e["id"],
                "output": e["output"],
                "inputs": e["inputs"],
                "label": e["label"],
                "domain": e["domain"],
                "subdomain": e.get("subdomain", e["domain"]),
            }
            for e in trace["path"]
        ],
    }


def _build_row(
    *,
    target: str,
    trace: dict,
    trace_str: str,
    domain: str,
    sub: str,
    difficulty: int,
    question: str,
    feas: int,
    faith: dict,
    cot: str,
    use_cot: bool,
) -> dict:
    row = {
        "target": target,
        "trace": _trace_json(trace),
        "trace_str": trace_str,
        "domain": domain,
        "subdomain": sub,
        "requested_difficulty": difficulty,
        "claude_difficulty": difficulty,
        "claude_feasibility": feas,
        "claude_faithfulness": faith,
        "question": question,
    }
    cot_text = (cot or "").strip()
    if use_cot and cot_text:
        row["chain_of_thought"] = cot_text
    return row


def _gen_max_tokens_for_call(
    *,
    use_cot: bool,
    base: int,
    extra: int = 0,
) -> int:
    """Completion budget for a single-difficulty generation call."""
    cot_pad = 700 if use_cot else 0
    scaled = int(base) + cot_pad + int(extra)
    return min(scaled, 16384)


def _flush_json(rows: list, output_path: Path) -> None:
    tmp = output_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    tmp.replace(output_path)


def _failed_output_path(output_path: Path, explicit: str) -> Path:
    if explicit.strip():
        p = Path(explicit).expanduser()
        return p if p.is_absolute() else (Path.cwd() / p).resolve()
    return output_path.with_name(output_path.stem + "_failed.json")


def _build_failed_row(
    *,
    target: str,
    trace: dict,
    trace_str: str,
    domain: str,
    sub: str,
    difficulty: int,
    failure_kind: str,
    attempt: int,
    question: str | None = None,
    feas: int | None = None,
    faith: dict | None = None,
    drop_reason: str = "",
    cot: str = "",
    raw_generation: str = "",
    max_gen_tokens: int | None = None,
    error: str = "",
    parsed_difficulties: list | None = None,
) -> dict:
    row: dict = {
        "target": target,
        "trace": _trace_json(trace),
        "trace_str": trace_str,
        "domain": domain,
        "subdomain": sub,
        "requested_difficulty": int(difficulty),
        "failure_kind": failure_kind,
        "attempt": int(attempt),
        "question": question,
        "drop_reason": drop_reason,
        "claude_feasibility": feas,
        "claude_faithfulness": faith,
        "faithfulness_coverage": (
            float(faith["variable_coverage"]) if faith else None
        ),
        "target_present": (
            bool(faith["target_present"]) if faith else None
        ),
        "max_gen_tokens": max_gen_tokens,
        "parsed_difficulties": parsed_difficulties,
        "error": error or None,
    }
    if raw_generation:
        row["raw_generation"] = raw_generation[:8000]
    cot_text = (cot or "").strip()
    if cot_text:
        row["chain_of_thought"] = cot_text
    return row


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
    ap.add_argument(
        "--gen-max-tokens",
        type=int,
        default=2048,
        help="Max completion tokens per generation call (lower = faster; default 2048).",
    )
    ap.add_argument(
        "--gen-temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for question generation (default 0.7).",
    )
    ap.add_argument(
        "--failed-output",
        dest="failed_output",
        default="",
        help="JSON path for failed generations and judge drops "
        "(default: <output_stem>_failed.json).",
    )
    add_llm_cli(ap)
    args = ap.parse_args()
    use_cot = not args.no_cot
    print(
        f"Chain-of-thought prompting: {'on (default)' if use_cot else 'off (--no-cot)'}; "
        f"one difficulty per API call; gen_max_tokens={args.gen_max_tokens}, "
        f"gen_temperature={args.gen_temperature}",
        flush=True,
    )

    rng = random.Random(args.seed)
    g = HyperGraph(args.graph)
    client = llm_client_from_args(args)
    print(f"LLM backend: {describe_llm_client(client)}", flush=True)

    candidates = g.producible()
    out_path = Path(args.output)
    failed_path = _failed_output_path(out_path, args.failed_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            out = json.load(f)
        print(f"Resuming: {len(out)} existing records in {out_path}", flush=True)
    else:
        out = []
    if failed_path.exists():
        with open(failed_path, encoding="utf-8") as f:
            failed_out = json.load(f)
        print(f"Resuming: {len(failed_out)} failed records in {failed_path}", flush=True)
    else:
        failed_out = []
    print(f"Failed log: {failed_path}", flush=True)
    seen_targets = []

    def record_failure(**kwargs) -> None:
        failed_out.append(_build_failed_row(**kwargs))
        _flush_json(failed_out, failed_path)

    sys_gen = SYSTEM_GEN_WITH_COT if use_cot else SYSTEM_GEN
    usr_gen = USER_GEN_WITH_COT if use_cot else USER_GEN

    def gen_one(
        trace_str,
        target,
        leafs,
        domain,
        difficulty: int,
        *,
        token_extra: int = 0,
    ):
        """One API call for a single difficulty -> (pairs, raw, cot, max_tokens)."""
        d = int(difficulty)
        max_t = _gen_max_tokens_for_call(
            use_cot=use_cot,
            base=args.gen_max_tokens,
            extra=token_extra,
        )
        raw = client.complete(
            sys_gen,
            usr_gen.format(
                trace_str=trace_str,
                target=target,
                leafs=", ".join(leafs) or "(none)",
                domain=domain,
                difficulty=d,
            ),
            max_tokens=max_t,
            temperature=args.gen_temperature,
        )
        pairs, cot = parse_question_list(raw)
        return pairs, raw, cot, max_t

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

        # One API call per difficulty level.
        remaining = list(args.difficulties)
        for attempt in range(args.max_regens + 1):
            if not remaining:
                break
            token_extra = attempt * 600
            for d in sorted(list(remaining)):
                try:
                    pairs, raw, cot, max_t = gen_one(
                        trace_str,
                        target,
                        trace["leafs"],
                        domain,
                        d,
                        token_extra=token_extra,
                    )
                except Exception as e:
                    print(f"  d={d} GEN FAILED: {e}", file=sys.stderr)
                    record_failure(
                        target=target,
                        trace=trace,
                        trace_str=trace_str,
                        domain=domain,
                        sub=sub,
                        difficulty=d,
                        failure_kind="gen_exception",
                        attempt=attempt + 1,
                        error=str(e),
                        max_gen_tokens=_gen_max_tokens_for_call(
                            use_cot=use_cot,
                            base=args.gen_max_tokens,
                            extra=token_extra,
                        ),
                    )
                    continue
                if not pairs:
                    print(
                        f"  d={d} GEN FAILED: could not parse JSON (max_tokens={max_t})",
                        file=sys.stderr,
                    )
                    print("  --- raw response (first 500 chars) ---", file=sys.stderr)
                    print(raw[:500], file=sys.stderr)
                    print("  --- end ---", file=sys.stderr)
                    record_failure(
                        target=target,
                        trace=trace,
                        trace_str=trace_str,
                        domain=domain,
                        sub=sub,
                        difficulty=d,
                        failure_kind="parse_error",
                        attempt=attempt + 1,
                        raw_generation=raw,
                        max_gen_tokens=max_t,
                        cot=cot if use_cot else "",
                    )
                    continue
                got = {pd: q for pd, q in pairs}
                if d not in got:
                    print(
                        f"  d={d} GEN FAILED: JSON had difficulties {list(got.keys())}, "
                        f"expected {d}",
                        file=sys.stderr,
                    )
                    record_failure(
                        target=target,
                        trace=trace,
                        trace_str=trace_str,
                        domain=domain,
                        sub=sub,
                        difficulty=d,
                        failure_kind="wrong_difficulty",
                        attempt=attempt + 1,
                        question=got.get(d),
                        raw_generation=raw,
                        max_gen_tokens=max_t,
                        cot=cot if use_cot else "",
                        parsed_difficulties=sorted(got.keys()),
                        drop_reason=f"expected d={d}, got {sorted(got.keys())}",
                    )
                    continue
                question = got[d]
                feas, faith = judge_feasibility_and_faithfulness(
                    client, trace_str, trace["leafs"], target, question
                )
                cov = faith["variable_coverage"]
                tgt_ok = faith["target_present"] or not args.require_target_present
                feas_ok = feas >= args.min_feasibility
                cov_ok = cov >= args.min_coverage
                if feas_ok and cov_ok and tgt_ok:
                    remaining.remove(d)
                    row = _build_row(
                        target=target,
                        trace=trace,
                        trace_str=trace_str,
                        domain=domain,
                        sub=sub,
                        difficulty=d,
                        question=question,
                        feas=feas,
                        faith=faith,
                        cot=cot if use_cot else "",
                        use_cot=use_cot,
                    )
                    out.append(row)
                    _flush_json(out, out_path)
                    print(
                        f"  d={d} kept "
                        f"(feasibility={feas}, coverage={cov:.2f}, "
                        f"target_present={faith['target_present']}) "
                        f"-> saved ({len(out)} total)",
                        flush=True,
                    )
                else:
                    reason = []
                    if not feas_ok:
                        reason.append(f"feas={feas}<{args.min_feasibility}")
                    if not cov_ok:
                        reason.append(f"cov={cov:.2f}<{args.min_coverage}")
                    if not tgt_ok:
                        reason.append("target_missing")
                    drop_reason = ", ".join(reason)
                    print(
                        f"  d={d} dropped ({drop_reason}), "
                        f"attempt {attempt + 1}",
                        flush=True,
                    )
                    record_failure(
                        target=target,
                        trace=trace,
                        trace_str=trace_str,
                        domain=domain,
                        sub=sub,
                        difficulty=d,
                        failure_kind="quality_filter",
                        attempt=attempt + 1,
                        question=question,
                        feas=feas,
                        faith=faith,
                        drop_reason=drop_reason,
                        cot=cot if use_cot else "",
                        max_gen_tokens=max_t,
                    )
            if remaining and attempt < args.max_regens:
                print(f"  regenerating difficulties {remaining}", flush=True)

        if remaining:
            print(f"  giving up on difficulties {remaining} after "
                  f"{args.max_regens + 1} attempts", flush=True)
            for d in sorted(remaining):
                record_failure(
                    target=target,
                    trace=trace,
                    trace_str=trace_str,
                    domain=domain,
                    sub=sub,
                    difficulty=d,
                    failure_kind="gave_up",
                    attempt=args.max_regens + 1,
                    drop_reason=f"no passing question after {args.max_regens + 1} attempts",
                )

    print(f"\nSaved {len(out)} (prompt, response, difficulty) records to {out_path}")
    print(f"Saved {len(failed_out)} failed records to {failed_path}")


if __name__ == "__main__":
    main()
