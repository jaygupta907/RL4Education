#!/usr/bin/env python3
"""Streamlit UI: enumerate and visualize all solution traces for a hypergraph target.

    pip install streamlit networkx matplotlib
    streamlit run visualize_traces.py

Uses NetworkX for dependency graphs (variables → formulas → outputs).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

from traversal import HyperGraph, format_trace

HERE = Path(__file__).resolve().parent
DEFAULT_GRAPH = HERE / "data" / "physics_hypergraph_deeper.json"
GRAPH_OPTIONS = {
    "physics_hypergraph_deeper.json": HERE / "data" / "physics_hypergraph_deeper.json",
    "physics_hypergraph.json": HERE / "data" / "physics_hypergraph.json",
}

NODE_COLORS = {
    "target": "#e74c3c",
    "leaf": "#27ae60",
    "derived": "#3498db",
    "formula": "#f39c12",
}


@st.cache_resource(show_spinner="Loading hypergraph…")
def load_graph(path_str: str) -> HyperGraph:
    return HyperGraph(path_str)


@st.cache_data(show_spinner=False)
def cached_enumerate(
    path_str: str,
    target: str,
    max_depth: int,
    single_domain: bool,
    max_traces: int,
) -> List[Dict[str, Any]]:
    g = HyperGraph(path_str)
    return g.enumerate_traces(
        target,
        max_depth=max_depth,
        single_domain=single_domain,
        max_traces=max_traces if max_traces > 0 else None,
    )


def trace_to_digraph(trace: Dict[str, Any]) -> nx.DiGraph:
    """Build a directed graph: inputs → formula node → output for each step."""
    G = nx.DiGraph()
    target = trace["target"]
    leafs: Set[str] = set(trace["leafs"])
    derived: Set[str] = set()

    for step, edge in enumerate(trace["path"], start=1):
        out = edge["output"]
        derived.add(out)
        fid = f"{edge['id']}@{step}"
        G.add_node(
            fid,
            kind="formula",
            label=edge["label"],
            formula_id=edge["id"],
            domain=edge.get("domain", ""),
            step=step,
        )
        for inp in edge["inputs"]:
            G.add_edge(inp, fid, relation="input")
        G.add_edge(fid, out, relation="produces")

    for node in G.nodes:
        if G.nodes[node].get("kind") == "formula":
            continue
        if node == target:
            G.nodes[node]["kind"] = "target"
        elif node in leafs:
            G.nodes[node]["kind"] = "leaf"
        elif node in derived:
            G.nodes[node]["kind"] = "derived"
        else:
            G.nodes[node]["kind"] = "leaf"

    return G


def merged_digraph(traces: List[Dict[str, Any]]) -> nx.DiGraph:
    """Union of per-trace graphs (formula nodes tagged by trace index)."""
    G = nx.DiGraph()
    for t_idx, trace in enumerate(traces):
        sub = trace_to_digraph(trace)
        for n, data in sub.nodes(data=True):
            key = n if data.get("kind") != "formula" else f"T{t_idx}:{n}"
            G.add_node(key, **{**data, "trace_index": t_idx})
        for u, v, data in sub.edges(data=True):
            uk = u if sub.nodes[u].get("kind") != "formula" else f"T{t_idx}:{u}"
            vk = v if sub.nodes[v].get("kind") != "formula" else f"T{t_idx}:{v}"
            G.add_edge(uk, vk, **data)
    return G


def draw_trace_graph(
    G: nx.DiGraph,
    *,
    title: str,
    figsize: Tuple[float, float] = (11, 7),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    if G.number_of_nodes() == 0:
        ax.set_title(title)
        ax.axis("off")
        return fig

    pos = nx.spring_layout(G, seed=0, k=1.8 / max(G.number_of_nodes() ** 0.5, 1))

    by_kind: Dict[str, List[str]] = {}
    for n, data in G.nodes(data=True):
        by_kind.setdefault(data.get("kind", "derived"), []).append(n)

    for kind, nodes in by_kind.items():
        color = NODE_COLORS.get(kind, "#95a5a6")
        nx.draw_networkx_nodes(
            G, pos, nodelist=nodes, node_color=color,
            node_size=900 if kind == "formula" else 1200,
            alpha=0.92, ax=ax,
        )

    nx.draw_networkx_edges(
        G, pos, arrows=True, arrowsize=14,
        edge_color="#7f8c8d", width=1.2, ax=ax,
        connectionstyle="arc3,rad=0.08",
    )

    labels: Dict[str, str] = {}
    for n, data in G.nodes(data=True):
        if data.get("kind") == "formula":
            labels[n] = data.get("formula_id", n)
        else:
            labels[n] = n.replace("_", "\n") if len(n) > 14 else n

    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def draw_overview_stats(traces: List[Dict[str, Any]]) -> None:
    lengths = [len(t["path"]) for t in traces]
    c = Counter(lengths)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    xs = sorted(c.keys())
    ax.bar(xs, [c[x] for x in xs], color="#4a90e2", edgecolor="white")
    ax.set_xlabel("Trace length (formula steps)")
    ax.set_ylabel("Count")
    ax.set_title(f"Trace length distribution ({len(traces)} traces)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def main() -> None:
    st.set_page_config(page_title="Solution trace explorer", layout="wide")
    st.title("Solution trace explorer")
    st.caption(
        "Enumerate all distinct backward-derivation traces for a producible target "
        "and visualize each as a NetworkX dependency graph."
    )

    with st.sidebar:
        st.header("Hypergraph")
        graph_name = st.selectbox(
            "Graph file",
            options=list(GRAPH_OPTIONS.keys()),
            index=0,
        )
        graph_path = GRAPH_OPTIONS[graph_name]
        g = load_graph(str(graph_path))

        target = st.selectbox(
            "Target variable",
            options=g.producible(),
            index=g.producible().index("kinetic_energy")
            if "kinetic_energy" in g.producible() else 0,
        )
        st.caption(f"Chapter: **{g.hypernyms.get(target, 'unknown')}**")

        max_depth = st.slider("Max backward depth", 1, 8, 5)
        single_domain = st.checkbox("Single domain", value=True)
        max_traces = st.number_input(
            "Max traces (0 = unlimited)",
            min_value=0,
            max_value=5000,
            value=300,
            step=50,
        )
        enumerate_btn = st.button("Enumerate traces", type="primary")

    if "traces" not in st.session_state:
        st.session_state.traces = []
        st.session_state.trace_key = None

    key = (str(graph_path), target, max_depth, single_domain, max_traces)
    if enumerate_btn:
        with st.spinner("Enumerating traces…"):
            traces = cached_enumerate(
                str(graph_path),
                target,
                max_depth,
                single_domain,
                int(max_traces),
            )
        st.session_state.traces = traces
        st.session_state.trace_key = key

    if st.session_state.trace_key != key:
        st.info("Settings changed — click **Enumerate traces** to refresh.")
        return

    traces: List[Dict[str, Any]] = st.session_state.traces
    if not traces:
        st.warning(f"No traces found for **{target}** with the current settings.")
        return

    st.subheader(f"{len(traces)} trace(s) for `{target}`")
    if max_traces and len(traces) >= max_traces:
        st.warning(f"Stopped at max_traces={max_traces}; there may be more.")

    draw_overview_stats(traces)

    tab_one, tab_all, tab_text = st.tabs(["Single trace", "Overview graph", "Text"])

    with tab_one:
        idx = st.slider("Trace index", 0, len(traces) - 1, 0)
        trace = traces[idx]
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.metric("Steps", len(trace["path"]))
            st.metric("Leaf givens", len(trace["leafs"]))
        with col_b:
            st.write("Domains:", ", ".join(trace.get("chapters", [])))
            st.write("Subdomains:", ", ".join(trace.get("subdomains", [])))

        G = trace_to_digraph(trace)
        title = (
            f"Trace {idx + 1}/{len(traces)} — {target} "
            f"({len(trace['path'])} steps)"
        )
        fig = draw_trace_graph(G, title=title)
        st.pyplot(fig)
        plt.close(fig)

        with st.expander("Legend"):
            st.markdown(
                "- **Red**: target  \n"
                "- **Green**: leaf givens  \n"
                "- **Blue**: derived intermediates  \n"
                "- **Orange**: formula nodes (edge id label)"
            )

    with tab_all:
        show_merged = st.checkbox("Show merged graph (all traces)", value=False)
        if show_merged and len(traces) <= 40:
            MG = merged_digraph(traces)
            fig = draw_trace_graph(
                MG,
                title=f"Merged graphs for {len(traces)} traces",
                figsize=(13, 8),
            )
            st.pyplot(fig)
            plt.close(fig)
        elif show_merged:
            st.warning("Merged view disabled for >40 traces (too cluttered).")
        else:
            st.caption(
                "Enable merged view to overlay all traces (formula nodes prefixed "
                "by trace index)."
            )

    with tab_text:
        idx_t = st.number_input(
            "Trace # (1-based)",
            min_value=1,
            max_value=len(traces),
            value=1,
        )
        st.code(format_trace(traces[idx_t - 1]), language=None)


if __name__ == "__main__":
    main()
