"""Evaluation: SFT model generates difficulty-conditioned questions; three
independent LLM judges (Claude or OpenAI via ``--llm-provider``) score each
question for difficulty, faithfulness, and physical feasibility.

Supports both LoRA adapters (preferred) and full SFT checkpoints. The SFT
model is invoked through the Llama-3 chat template (system + user roles)
to match the training prompt format and to inherit Llama's instruction-
following behaviour from pretraining."""
import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cot_utils import strip_reasoning_prefix
from judges import judge_difficulty, judge_faithfulness, judge_feasibility
from llm_client import add_llm_cli, describe_llm_client, llm_client_from_args
from prompts import build_sft_chat_messages, format_trace
from traversal import HyperGraph

HERE = Path(__file__).parent
DATA = HERE / "data"


def load_model(base: str, sft_dir: str):
    """Load the SFT model. Three modes, in order of preference:
      1. LoRA adapter dir (sft_dir contains adapter_config.json) -> base + adapter merged.
      2. Full SFT checkpoint dir (sft_dir contains config.json + weights) -> direct load.
      3. sft_dir missing -> plain base model.
    """
    sft_path = Path(sft_dir)
    is_lora = sft_path.exists() and (sft_path / "adapter_config.json").exists()
    is_full_sft = sft_path.exists() and (sft_path / "config.json").exists() and not is_lora

    tok_src = sft_dir if (is_lora or is_full_sft) else base
    tok = AutoTokenizer.from_pretrained(tok_src)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    if is_lora:
        from peft import PeftModel
        print(f"Loading base model {base} + LoRA adapter from {sft_dir}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, sft_dir)
        model = model.merge_and_unload()
    else:
        src = sft_dir if is_full_sft else base
        print(f"Loading model from {src}")
        model = AutoModelForCausalLM.from_pretrained(
            src, torch_dtype=torch.bfloat16, device_map="auto"
        )
    model.eval()
    return model, tok


@torch.no_grad()
def generate_question(model, tok, messages, max_new_tokens: int = 768,
                      temperature: float = 0.8, top_p: float = 0.95) -> str:
    prompt_ids = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    out = model.generate(
        prompt_ids, max_new_tokens=max_new_tokens, do_sample=True,
        temperature=temperature, top_p=top_p,
        pad_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0][prompt_ids.shape[1]:], skip_special_tokens=True)
    return text.strip()


def tokens_for_difficulty(d: int, base: int, scale: int) -> int:
    """Easy questions are short; difficulty-10 narratives are long."""
    return base + scale * max(0, d - 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=str(DATA / "physics_hypergraph.json"))
    ap.add_argument("--sft_dir", default="/mnt/storage/ae21b026/sft_lora",
                    help="LoRA adapter dir (preferred) or full SFT checkpoint dir.")
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--output", default=str(DATA / "eval_results.json"))
    ap.add_argument("--num_targets", type=int, default=40)
    ap.add_argument(
        "--difficulties",
        type=int,
        nargs="+",
        default=list(range(1, 11)),
        help="Requested difficulty levels per target (default: 1..10).",
    )
    ap.add_argument("--max_depth", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude_targets_from", default=str(DATA / "dataset.json"),
                    help="JSON file whose `target` values are skipped.")
    ap.add_argument("--gen_tokens_base", type=int, default=350,
                    help="max_new_tokens at difficulty<=3")
    ap.add_argument("--gen_tokens_scale", type=int, default=80,
                    help="extra max_new_tokens per difficulty step above 3")
    ap.add_argument("--gen_temperature", type=float, default=0.5,
                    help="sampling temperature for the SFT model")
    ap.add_argument("--gen_top_p", type=float, default=0.95)
    ap.add_argument(
        "--with-cot",
        action="store_true",
        help="Use the same system/user prompts as CoT SFT (model emits <reasoning> "
        "then question). Strip reasoning before Claude judges. Match train_sft.py --with-cot.",
    )
    ap.add_argument("--single_domain", action="store_true", default=True,
                    help="restrict every eval trace to one physics chapter "
                         "and subdomain (must match dataset generation)")
    ap.add_argument("--no_single_domain", dest="single_domain",
                    action="store_false")
    add_llm_cli(ap)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    g = HyperGraph(args.graph)
    candidates = g.producible()
    excluded = set()
    if Path(args.exclude_targets_from).exists():
        with open(args.exclude_targets_from) as f:
            excluded = {it["target"] for it in json.load(f)}
    pool = [c for c in candidates if c not in excluded] or candidates

    model, tok = load_model(args.base_model, args.sft_dir)
    judge = llm_client_from_args(args)
    judge_desc = describe_llm_client(judge)
    print(f"Judge backend: {judge_desc}", flush=True)

    out = []
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_suffix(".meta.json")
    meta = {
        "llm_provider": args.llm_provider,
        "llm_model": judge.model,
        "judge_backend": judge_desc,
        "base_model": args.base_model,
        "sft_dir": args.sft_dir,
    }
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)
    i, attempts = 0, 0
    while i < args.num_targets and attempts < args.num_targets * 5:
        attempts += 1
        target = rng.choice(pool)
        trace = g.traverse(target, max_depth=args.max_depth,
                           seed=rng.randint(0, 10**9),
                           single_domain=args.single_domain)
        if trace is None or not trace["path"]:
            continue
        if args.single_domain and (len(trace["chapters"]) != 1
                                   or len(trace.get("subdomains", [])) > 1):
            continue
        i += 1
        trace_str = format_trace(trace)
        chapters = ",".join(trace["chapters"])
        subs = ",".join(trace.get("subdomains", trace["chapters"]))
        print(f"[{i}/{args.num_targets}] target={target} "
              f"chapter={chapters} sub={subs} steps={len(trace['path'])}",
              flush=True)
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
            mnt = tokens_for_difficulty(d, args.gen_tokens_base,
                                        args.gen_tokens_scale)
            if args.with_cot:
                mnt += 512
            raw_out = generate_question(
                model, tok, messages, max_new_tokens=mnt,
                temperature=args.gen_temperature, top_p=args.gen_top_p,
            )
            question = strip_reasoning_prefix(raw_out)
            claude_diff = judge_difficulty(
                judge, trace_str, trace["leafs"], target, question)
            claude_faith = judge_faithfulness(
                judge, trace_str, trace["leafs"], target, question)
            claude_feas = judge_feasibility(
                judge, trace_str, trace["leafs"], target, question)
            out.append({
                "target": target,
                "trace": {
                    "leafs": trace["leafs"],
                    "chapters": trace["chapters"],
                    "hypernym": trace["hypernym"],
                    "path": [{"id": e["id"], "output": e["output"],
                              "inputs": e["inputs"], "label": e["label"],
                              "domain": e["domain"]} for e in trace["path"]],
                },
                "trace_str": trace_str,
                "requested_difficulty": d,
                "claude_difficulty": claude_diff,
                "claude_faithfulness": claude_faith,
                "claude_feasibility": claude_feas,
                "question": question,
            })
            print(f"  d={d} requested={d} claude={claude_diff} "
                  f"faithful={claude_faith['faithful']} "
                  f"coverage={claude_faith['variable_coverage']:.2f} "
                  f"feasibility={claude_feas}",
                  flush=True)
            tmp = out_path.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(out, f, indent=2)
            tmp.replace(out_path)

    print(f"\nSaved {len(out)} eval records to {out_path}")
    print(f"Judge metadata: {meta_path}", flush=True)


if __name__ == "__main__":
    main()
