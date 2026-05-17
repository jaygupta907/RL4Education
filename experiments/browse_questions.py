#!/usr/bin/env python3
"""Streamlit UI to browse training dataset and base / SFT evaluation JSON.

    pip install streamlit
    streamlit run browse_questions.py

JSON paths are fixed below (under ``experiments/data/``). Edit those
constants if your files live elsewhere.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import streamlit as st

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# --- Hardcoded JSON locations (edit if needed) --------------------------------

DATASET_JSON = DATA / "dataset.json"
DATASET_DEEPER_JSON = DATA / "dataset_deeper.json"
DATASET_COT_JSON = DATA / "dataset_cot.json"

# Evaluation pairs: (base_eval.json, sft_eval.json)
EVAL_CLAUDE_FAITHFULNESS: Tuple[Path, Path] = (
    DATA / "claude_faithfulness" / "eval_base.json",
    DATA / "claude_faithfulness" / "eval_sft.json",
)
EVAL_CLAUDE_FAITHFULNESS_DEEPER: Tuple[Path, Path] = (
    DATA / "claude_faithfulness_deeper" / "eval_base.json",
    DATA / "claude_faithfulness_deeper" / "eval_sft.json",
)
EVAL_WITHOUT_CLAUDE_FAITHFULNESS: Tuple[Path, Path] = (
    DATA / "without_claude_faithfulness" / "eval_base.json",
    DATA / "without_claude_faithfulness" / "eval_sft.json",
)


@st.cache_data(show_spinner=False)
def load_records(path_str: str) -> List[Dict[str, Any]]:
    p = Path(path_str)
    if not p.is_file():
        return []
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def chapter_of(rec: Dict[str, Any]) -> str:
    t = rec.get("trace") or {}
    ch = t.get("chapters") or []
    return str(ch[0]) if ch else ""


def faith_block(rec: Dict[str, Any]) -> Dict[str, Any]:
    return rec.get("claude_faithfulness") or {}


def filter_records(
    rows: List[Dict[str, Any]],
    difficulties: Optional[Set[int]],
    chapters: Optional[Set[str]],
    target_substr: str,
    question_substr: str,
    faithful_only: bool,
    target_present_only: bool,
    cot_substr: str = "",
    has_cot_only: bool = False,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    tq = target_substr.strip().lower()
    qq = question_substr.strip().lower()
    cq = cot_substr.strip().lower()
    for r in rows:
        d = r.get("requested_difficulty")
        if difficulties is not None and d not in difficulties:
            continue
        ch = chapter_of(r)
        if chapters and ch not in chapters:
            continue
        if tq and tq not in str(r.get("target", "")).lower():
            continue
        q = str(r.get("question", ""))
        if qq and qq not in q.lower():
            continue
        fb = faith_block(r)
        if faithful_only and not fb.get("faithful"):
            continue
        if target_present_only and not fb.get("target_present"):
            continue
        cot = str(r.get("chain_of_thought") or "").strip()
        if has_cot_only and not cot:
            continue
        if cq and cq not in cot.lower():
            continue
        out.append(r)
    return out


def show_record(
    rec: Dict[str, Any],
    key_prefix: str,
    *,
    expand_cot: bool = False,
) -> None:
    fb = faith_block(rec)
    st.markdown(
        f"**Target:** `{rec.get('target', '')}`  \n"
        f"**Requested difficulty:** {rec.get('requested_difficulty', '—')}  \n"
        f"**Claude difficulty:** {rec.get('claude_difficulty', '—')}  \n"
        f"**Feasibility:** {rec.get('claude_feasibility', '—')}  \n"
        f"**Variable coverage:** {fb.get('variable_coverage', '—')}  \n"
        f"**Target present:** {fb.get('target_present', '—')}  \n"
        f"**Faithful:** {fb.get('faithful', '—')}"
    )
    if rec.get("model"):
        st.caption(f"Model tag: `{rec['model']}`")
    dom = rec.get("domain") or chapter_of(rec)
    sub = rec.get("subdomain")
    if dom or sub:
        st.caption(f"Domain: {dom or '—'} / {sub or '—'}")

    cot = str(rec.get("chain_of_thought") or "").strip()
    if cot:
        with st.expander(
            "Chain of thought (dataset generation)",
            expanded=expand_cot,
        ):
            st.markdown(cot)

    st.subheader("Question")
    st.text_area("question", value=str(rec.get("question", "")), height=220, key=f"{key_prefix}_q", disabled=True)

    st.subheader("Trace (formatted)")
    st.code(rec.get("trace_str") or "", language=None)

    hits = fb.get("leaf_hits") or {}
    if hits:
        st.subheader("Leaf hits")
        cols = st.columns(min(4, max(1, len(hits))))
        for i, (leaf, ok) in enumerate(sorted(hits.items())):
            cols[i % len(cols)].markdown(f"- `{leaf}`: **{ok}**")


def main() -> None:
    st.set_page_config(page_title="Question browser", layout="wide")
    st.title("Question browser")
    st.caption("Training dataset · Base eval · SFT eval (paths fixed in browse_questions.py)")

    with st.sidebar:
        st.header("Dataset file")
        dataset_choice = st.radio(
            "Training JSON",
            (
                ("dataset.json", DATASET_JSON),
                ("dataset_deeper.json", DATASET_DEEPER_JSON),
                ("dataset_cot.json", DATASET_COT_JSON),
            ),
            format_func=lambda x: x[0],
        )
        dataset_path = dataset_choice[1]

        st.header("Evaluation JSON pair")
        eval_choice = st.radio(
            "Eval run",
            (
                ("claude_faithfulness/", EVAL_CLAUDE_FAITHFULNESS),
                ("claude_faithfulness_deeper/", EVAL_CLAUDE_FAITHFULNESS_DEEPER),
                ("without_claude_faithfulness/", EVAL_WITHOUT_CLAUDE_FAITHFULNESS),
            ),
            format_func=lambda x: x[0],
        )
        base_path, sft_path = eval_choice[1]

        st.header("View")
        view = st.radio(
            "Source",
            ("Dataset", "Base evaluation", "SFT evaluation", "Compare base vs SFT (same index)"),
            horizontal=False,
        )

        st.header("Filters")
        faithful_only = st.checkbox("Fully faithful only", value=False)
        target_present_only = st.checkbox("Target present only", value=False)
        target_substr = st.text_input("Target contains", "")
        question_substr = st.text_input("Question contains", "")
        cot_substr = st.text_input("Chain-of-thought contains", "")
        has_cot_only = st.checkbox(
            "Only rows with stored chain-of-thought",
            value=False,
        )

    ds = load_records(str(dataset_path.resolve()))
    base = load_records(str(base_path.resolve()))
    sft = load_records(str(sft_path.resolve()))

    st.caption(
        f"Dataset: `{dataset_path.relative_to(HERE)}` · "
        f"Eval: `{base_path.parent.relative_to(HERE)}/`"
    )

    if view == "Dataset":
        rows, label = ds, "dataset"
    elif view == "Base evaluation":
        rows, label = base, "base"
    elif view == "SFT evaluation":
        rows, label = sft, "sft"
    else:
        rows, label = [], "compare"

    if view != "Compare base vs SFT (same index)":
        if not rows:
            path_hint = {
                "dataset": dataset_path,
                "base": base_path,
                "sft": sft_path,
            }[label]
            st.error(f"No records loaded for **{label}**. File missing or empty: `{path_hint}`")
            st.stop()

        all_d = sorted({int(r["requested_difficulty"]) for r in rows if isinstance(r.get("requested_difficulty"), int)})
        all_ch = sorted({chapter_of(r) for r in rows if chapter_of(r)})

        with st.sidebar:
            difficulties = st.multiselect("Requested difficulty", options=all_d, default=all_d)
            chapters = st.multiselect("Chapter", options=all_ch, default=all_ch)

        if not difficulties or not chapters:
            st.warning("Select at least one difficulty and one chapter to show records.")
            filtered = []
        else:
            filtered = filter_records(
                rows,
                set(difficulties),
                set(chapters),
                target_substr,
                question_substr,
                faithful_only,
                target_present_only,
                cot_substr=cot_substr,
                has_cot_only=has_cot_only,
            )

        st.metric("Records (filtered)", len(filtered), delta=f"of {len(rows)} total")

        expand_cot_panel = (
            label == "dataset" and dataset_path.name == "dataset_cot.json"
        )

        page_size = st.slider("Page size", 1, 50, 10)
        n_pages = max(1, (len(filtered) + page_size - 1) // page_size)
        page = st.number_input("Page (0-based)", min_value=0, max_value=n_pages - 1, value=0, step=1)
        start = page * page_size
        chunk = filtered[start : start + page_size]

        for j, rec in enumerate(chunk):
            idx = start + j
            with st.expander(
                f"#{idx} · d={rec.get('requested_difficulty')} · "
                f"{rec.get('target', '')} · {chapter_of(rec) or '?'}"[:120],
                expanded=(page_size <= 3),
            ):
                show_record(
                    rec,
                    f"{label}_{idx}",
                    expand_cot=expand_cot_panel,
                )

        if chunk:
            st.subheader("Page summary")
            summary = []
            for j, rec in enumerate(chunk):
                idx = start + j
                fb = faith_block(rec)
                q = str(rec.get("question", ""))[:100].replace("\n", " ")
                cot_n = len(str(rec.get("chain_of_thought") or ""))
                summary.append(
                    {
                        "#": idx,
                        "d_req": rec.get("requested_difficulty"),
                        "d_claude": rec.get("claude_difficulty"),
                        "feas": rec.get("claude_feasibility"),
                        "faith": fb.get("faithful"),
                        "cov": fb.get("variable_coverage"),
                        "cot_chars": cot_n,
                        "target": rec.get("target"),
                        "chapter": chapter_of(rec),
                        "preview": q + ("…" if len(str(rec.get("question", ""))) > 100 else ""),
                    }
                )
            st.dataframe(summary, use_container_width=True, hide_index=True)
        st.stop()

    if not base or not sft:
        st.error("Compare mode needs both base and SFT eval files to load for the selected eval run.")
        st.stop()

    n = min(len(base), len(sft))
    st.metric("Paired records", n, delta=f"base={len(base)}, sft={len(sft)}")

    with st.sidebar:
        compare_idx = st.number_input("Record index (paired order)", min_value=0, max_value=max(0, n - 1), value=0, step=1)

    b = base[compare_idx]
    s = sft[compare_idx]
    same_pair = b.get("target") == s.get("target") and b.get("requested_difficulty") == s.get("requested_difficulty")
    if not same_pair:
        st.warning("Warning: base and SFT rows at this index do not share the same (target, requested_difficulty). Files may differ.")

    col1, col2 = st.columns(2)
    with col1:
        st.header("Base")
        show_record(b, f"cmp_base_{compare_idx}")
    with col2:
        st.header("SFT")
        show_record(s, f"cmp_sft_{compare_idx}")


if __name__ == "__main__":
    main()
