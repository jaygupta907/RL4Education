"""Backward traversal of the physics hypergraph.

Given a target variable, randomly chooses a producing formula, recurses
on the formula's inputs, and returns a single solution trace in topological
(execution) order. Cycles are broken by treating the offending variable as
a leaf for that branch.
"""
import json
import random
from collections import defaultdict
from typing import Dict, List, Optional


class HyperGraph:
    def __init__(self, path: str):
        with open(path) as f:
            g = json.load(f)
        self.nodes: List[str] = g["nodes"]
        self.edges: List[Dict] = g["hyperedges"]
        self.hypernyms: Dict[str, str] = g["hypernyms"]
        self.chapters: List[str] = g["chapters"]
        self.out_to_edges: Dict[str, List[Dict]] = defaultdict(list)
        for e in self.edges:
            self.out_to_edges[e["output"]].append(e)

    def producible(self) -> List[str]:
        return sorted(self.out_to_edges.keys())

    def traverse(self, target: str, max_depth: int = 5,
                 seed: Optional[int] = None,
                 single_domain: bool = False,
                 domain: Optional[str] = None,
                 subdomain: Optional[str] = None) -> Optional[Dict]:
        """Backward-DFS from `target` to a single solution trace.

        single_domain: if True, only formulas in the same chapter (and
            sub-domain, if the edge has one) as the first chosen formula
            are allowed; out-of-domain inputs are forced to be leaf givens.
            This guarantees the trace stays within one physics chapter
            AND one physical-model subdomain (e.g. kinematics:projectile,
            kinematics:uniform_accel), preventing scenarios that would
            mix incompatible motion models.
        domain: if given, restricts allowed formulas to this chapter
            (overrides the auto-detection from the target).
        subdomain: if given, restricts allowed formulas to this subdomain
            (overrides the auto-detection from the target).
        """
        rng = random.Random(seed)
        chosen: Dict[str, Dict] = {}
        leafs = set()
        order: List[Dict] = []
        allowed = {"d": domain, "sd": subdomain}  # mutable closure cell

        def dfs(var: str, depth: int, visiting: set) -> bool:
            """Returns True if `var` is resolvable (as derived or as leaf)."""
            if var in chosen or var in leafs:
                return True
            if var in visiting:
                return False  # cycle on caller's chain
            edges = [e for e in self.out_to_edges.get(var, [])
                     if target not in e["inputs"]]
            if single_domain and allowed["d"] is not None:
                edges = [e for e in edges if e["domain"] == allowed["d"]]
                if allowed["sd"] is not None:
                    edges = [e for e in edges
                             if e.get("subdomain", e["domain"]) == allowed["sd"]]
            if not edges or depth >= max_depth:
                leafs.add(var)
                return True
            shuffled = edges[:]
            rng.shuffle(shuffled)
            # Prefer formulas with more producible inputs (longer traces) when
            # several edges are valid; random tie-break keeps sampling stochastic.
            shuffled.sort(
                key=lambda e: (
                    -sum(1 for i in e["inputs"] if i in self.out_to_edges),
                    rng.random(),
                )
            )
            visiting.add(var)
            for edge in shuffled:
                snap = (dict(chosen), list(order), set(leafs),
                        allowed["d"], allowed["sd"])
                if single_domain and allowed["d"] is None:
                    allowed["d"] = edge["domain"]
                    allowed["sd"] = edge.get("subdomain", edge["domain"])
                if all(dfs(i, depth + 1, visiting) for i in edge["inputs"]):
                    chosen[var] = edge
                    order.append(edge)  # post-order => topological order
                    visiting.remove(var)
                    return True
                # rollback choices made by this failed branch
                chosen.clear(); chosen.update(snap[0])
                order.clear(); order.extend(snap[1])
                leafs.clear(); leafs.update(snap[2])
                allowed["d"], allowed["sd"] = snap[3], snap[4]
            visiting.remove(var)
            leafs.add(var)  # no formula works without cycle - treat as given
            return True

        dfs(target, 0, set())
        if target not in chosen:
            return None
        leafs.discard(target)
        return {
            "target": target,
            "path": order,
            "leafs": sorted(leafs),
            "chapters": sorted({e["domain"] for e in order}),
            "subdomains": sorted({e.get("subdomain", e["domain"])
                                  for e in order}),
            "hypernym": self.hypernyms.get(target, "unknown"),
        }

    def enumerate_traces(
        self,
        target: str,
        max_depth: int = 5,
        *,
        single_domain: bool = False,
        domain: Optional[str] = None,
        subdomain: Optional[str] = None,
        max_traces: Optional[int] = None,
    ) -> List[Dict]:
        """Enumerate all distinct solution traces for ``target`` (no randomness).

        Traces are deduplicated by ``(path edge ids, leaf set)``. Enumeration
        tries every valid producing formula at each step (sorted by edge id).
        """
        results: List[Dict] = []
        seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
        chosen: Dict[str, Dict] = {}
        leafs: set[str] = set()
        order: List[Dict] = []
        allowed = {"d": domain, "sd": subdomain}

        def record() -> None:
            key = (tuple(e["id"] for e in order), tuple(sorted(leafs)))
            if key in seen:
                return
            seen.add(key)
            results.append({
                "target": target,
                "path": list(order),
                "leafs": sorted(leafs),
                "chapters": sorted({e["domain"] for e in order}),
                "subdomains": sorted({
                    e.get("subdomain", e["domain"]) for e in order
                }),
                "hypernym": self.hypernyms.get(target, "unknown"),
            })

        def snap() -> tuple:
            return (dict(chosen), list(order), set(leafs),
                    allowed["d"], allowed["sd"])

        def restore(state: tuple) -> None:
            chosen.clear()
            chosen.update(state[0])
            order.clear()
            order.extend(state[1])
            leafs.clear()
            leafs.update(state[2])
            allowed["d"], allowed["sd"] = state[3], state[4]

        def resolve_var(var: str, depth: int, visiting: set[str]) -> None:
            """Backtracking generator: resolve ``var`` then yield."""
            if max_traces is not None and len(results) >= max_traces:
                return
            if var in chosen or var in leafs:
                yield
                return
            if var in visiting:
                return
            edges = [e for e in self.out_to_edges.get(var, [])
                     if target not in e["inputs"]]
            if single_domain and allowed["d"] is not None:
                edges = [e for e in edges if e["domain"] == allowed["d"]]
                if allowed["sd"] is not None:
                    edges = [e for e in edges
                             if e.get("subdomain", e["domain"]) == allowed["sd"]]
            if not edges or depth >= max_depth:
                leafs.add(var)
                yield
                leafs.discard(var)
                return
            edges = sorted(edges, key=lambda e: e["id"])
            visiting.add(var)
            for edge in edges:
                if max_traces is not None and len(results) >= max_traces:
                    break
                branch = snap()
                lock_d, lock_sd = allowed["d"], allowed["sd"]
                if single_domain and allowed["d"] is None:
                    allowed["d"] = edge["domain"]
                    allowed["sd"] = edge.get("subdomain", edge["domain"])
                sub_ok = False
                for _ in resolve_inputs(edge["inputs"], depth + 1, visiting):
                    sub_ok = True
                    chosen[var] = edge
                    order.append(edge)
                    yield
                    order.pop()
                    del chosen[var]
                if not sub_ok:
                    restore(branch)
                allowed["d"], allowed["sd"] = lock_d, lock_sd
            visiting.discard(var)
            leafs.add(var)
            yield
            leafs.discard(var)

        def resolve_inputs(inputs: List[str], depth: int, visiting: set[str]):
            if not inputs:
                yield
                return
            var, rest = inputs[0], inputs[1:]
            for _ in resolve_var(var, depth, visiting):
                yield from resolve_inputs(rest, depth, visiting)

        def resolve_target() -> None:
            edges = [e for e in self.out_to_edges.get(target, [])
                     if target not in e["inputs"]]
            if single_domain and allowed["d"] is not None:
                edges = [e for e in edges if e["domain"] == allowed["d"]]
                if allowed["sd"] is not None:
                    edges = [e for e in edges
                             if e.get("subdomain", e["domain"]) == allowed["sd"]]
            if not edges:
                return
            edges = sorted(edges, key=lambda e: e["id"])
            visiting: set[str] = {target}
            for edge in edges:
                if max_traces is not None and len(results) >= max_traces:
                    break
                branch = snap()
                lock_d, lock_sd = allowed["d"], allowed["sd"]
                if single_domain and allowed["d"] is None:
                    allowed["d"] = edge["domain"]
                    allowed["sd"] = edge.get("subdomain", edge["domain"])
                for _ in resolve_inputs(edge["inputs"], 1, visiting):
                    chosen[target] = edge
                    order.append(edge)
                    record()
                    order.pop()
                    del chosen[target]
                restore(branch)
                allowed["d"], allowed["sd"] = lock_d, lock_sd

        chosen.clear()
        leafs.clear()
        order.clear()
        allowed["d"] = domain
        allowed["sd"] = subdomain
        resolve_target()
        return results


def format_trace(trace: Dict) -> str:
    lines = [f"Target: {trace['target']}  (chapter: {trace['hypernym']})",
             f"Given variables: {', '.join(trace['leafs']) or '(none)'}",
             "Solution steps:"]
    for i, e in enumerate(trace["path"], 1):
        lines.append(
            f"  {i}. {e['output']} = {e['label']}  "
            f"[uses: {', '.join(e['inputs'])}]  ({e['domain']})"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "kinetic_energy"
    g = HyperGraph("data/physics_hypergraph.json")
    t = g.traverse(target, max_depth=5, seed=0)
    if t is None:
        print(f"No producing formula for '{target}'.")
    else:
        print(format_trace(t))
