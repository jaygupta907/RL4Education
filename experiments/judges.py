"""Independent LLM judges used during evaluation (Claude or OpenAI via ``llm_client``).

Three separate API calls per generated question, each with its own
system prompt:
  - judge_difficulty:    SYSTEM_SCORE / USER_SCORE -> int 1..10
  - judge_faithfulness:  SYSTEM_FAITH / USER_FAITH -> {leaf_hits,
                                                       target_present,
                                                       variable_coverage,
                                                       faithful}
  - judge_feasibility:   SYSTEM_FEAS  / USER_FEAS  -> int 1..10
The judges share no state.
"""
import json
import re
from typing import Dict, List

from prompts import (SYSTEM_FAITH, SYSTEM_FEAS, SYSTEM_SCORE,
                     USER_FAITH, USER_FEAS, USER_SCORE)


def parse_score(text: str) -> int:
    for tok in text.replace(",", " ").split():
        try:
            v = int(tok.strip("().:/"))
            if 1 <= v <= 10:
                return v
        except ValueError:
            continue
    return -1


_FAITH_DEFAULT = lambda leafs: {
    "leaf_hits": {leaf: False for leaf in leafs},
    "target_present": False,
}


def parse_faith_json(text: str, leafs: List[str]) -> Dict:
    """Robustly parse a Claude faithfulness response. Never raises - on any
    parse failure (empty response, no JSON, malformed JSON, missing keys),
    returns the safe default of all-leaves-missing and target-absent so the
    eval loop can continue and the failure is visibly reflected in coverage."""
    if not text:
        return _FAITH_DEFAULT(leafs)
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    lo = s.find("{")
    if lo == -1:
        return _FAITH_DEFAULT(leafs)
    try:
        obj, _ = json.JSONDecoder().raw_decode(s[lo:])
    except json.JSONDecodeError:
        return _FAITH_DEFAULT(leafs)
    if not isinstance(obj, dict):
        return _FAITH_DEFAULT(leafs)
    raw_hits = obj.get("leaf_hits") or {}
    if not isinstance(raw_hits, dict):
        raw_hits = {}
    return {
        "leaf_hits": {leaf: bool(raw_hits.get(leaf, False)) for leaf in leafs},
        "target_present": bool(obj.get("target_present", False)),
    }


def judge_difficulty(client, trace_str: str, leafs: List[str], target: str,
                     question: str) -> int:
    try:
        raw = client.complete(
            SYSTEM_SCORE,
            USER_SCORE.format(
                trace_str=trace_str,
                leafs=", ".join(leafs) or "(none)",
                target=target, question=question,
            ),
            max_tokens=64, temperature=0.0,
        )
    except Exception:
        return -1
    return parse_score(raw)


def judge_faithfulness(client, trace_str: str, leafs: List[str], target: str,
                       question: str) -> Dict:
    try:
        raw = client.complete(
            SYSTEM_FAITH,
            USER_FAITH.format(
                trace_str=trace_str,
                leafs=", ".join(leafs) or "(none)",
                target=target, question=question,
            ),
            max_tokens=256, temperature=0.0,
        )
    except Exception:
        raw = ""
    parsed = parse_faith_json(raw, leafs)
    n = max(len(leafs), 1)
    var_hits = sum(parsed["leaf_hits"].values())
    coverage = var_hits / n
    return {
        "leaf_hits": parsed["leaf_hits"],
        "target_present": parsed["target_present"],
        "variable_coverage": coverage,
        "faithful": bool(parsed["target_present"] and coverage >= 1.0),
    }


def judge_feasibility(client, trace_str: str, leafs: List[str], target: str,
                      question: str) -> int:
    try:
        raw = client.complete(
            SYSTEM_FEAS,
            USER_FEAS.format(
                trace_str=trace_str,
                leafs=", ".join(leafs) or "(none)",
                target=target, question=question,
            ),
            max_tokens=64, temperature=0.0,
        )
    except Exception:
        return -1
    return parse_score(raw)
