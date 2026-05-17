"""LoRA SFT of Llama-3-8B-Instruct on the difficulty-conditioned dataset.

The training prompt uses the model's native chat template (system + user
roles) and the assistant turn is the Claude-generated question. Loss is
computed only on the assistant tokens. LoRA preserves the base model's
broad instruction-following and physics-reasoning capability while adding
a small adapter that conditions generation on (trace, target, given,
domain, subdomain, difficulty).

Pass ``--with-cot`` to train on ``<reasoning>`` + question (uses the
``chain_of_thought`` field when present). Run ``eval_pipeline.py`` with the
same ``--with-cot`` so prompts match; judges still see question-only text
after stripping the reasoning block.
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

from prompts import build_sft_chat_messages

HERE = Path(__file__).parent


def _resolve_domain(item):
    chapters = item.get("trace", {}).get("chapters") or []
    subdomains = item.get("trace", {}).get("subdomains") or []
    domain = item.get("domain") or (chapters[0] if chapters else "")
    subdomain = item.get("subdomain") or (subdomains[0] if subdomains else domain)
    return domain, subdomain


def build_supervised_text(item, tok, *, with_cot: bool) -> dict:
    domain, subdomain = _resolve_domain(item)
    messages = build_sft_chat_messages(
        item["trace_str"],
        item["target"],
        item["trace"]["leafs"],
        item["requested_difficulty"],
        domain=domain,
        subdomain=subdomain,
        expect_chain_of_thought=with_cot,
    )
    prefix = tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    q = item["question"].strip()
    if with_cot:
        cot = str(item.get("chain_of_thought") or "").strip()
        if cot:
            completion = (
                f"<reasoning>\n{cot}\n</reasoning>\n\n{q}{tok.eos_token}"
            )
        else:
            completion = q + tok.eos_token
    else:
        completion = q + tok.eos_token
    return {"prefix": prefix, "completion": completion}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(HERE / "data" / "dataset.json"))
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--output_dir", default="/mnt/storage/ae21b026/sft_lora")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="LoRA needs a higher LR than full fine-tuning")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--with-cot",
        action="store_true",
        help="Train assistant to emit <reasoning>...</reasoning> then the question; "
        "requires chain_of_thought on rows (falls back to question-only if missing). "
        "Use eval_pipeline.py --with-cot at inference.",
    )
    args = ap.parse_args()

    with open(args.dataset) as f:
        raw = json.load(f)
    if not raw:
        raise SystemExit("Dataset is empty - run generate_dataset.py first.")

    if args.with_cot:
        n_cot = sum(1 for it in raw if str(it.get("chain_of_thought") or "").strip())
        print(
            f"--with-cot: {n_cot}/{len(raw)} rows have non-empty chain_of_thought",
            flush=True,
        )
        if n_cot < len(raw):
            print(
                "  (rows without CoT use question-only completion; consider a full CoT dataset)",
                flush=True,
            )

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    if args.with_cot and args.max_len < 3072:
        print(
            f"  Note: --with-cot sequences are longer; consider --max_len 3072 or 4096 "
            f"(current {args.max_len}).",
            flush=True,
        )

    rows = [build_supervised_text(it, tok, with_cot=args.with_cot) for it in raw]

    def encode(ex):
        prefix_ids = tok(ex["prefix"], add_special_tokens=False)["input_ids"]
        comp_ids = tok(ex["completion"], add_special_tokens=False)["input_ids"]
        ids = prefix_ids + comp_ids
        labels = [-100] * len(prefix_ids) + comp_ids
        ids = ids[: args.max_len]
        labels = labels[: args.max_len]
        return {"input_ids": ids, "labels": labels,
                "attention_mask": [1] * len(ids)}

    ds = Dataset.from_list(rows).map(encode, remove_columns=["prefix", "completion"])
    print(f"Training examples: {len(ds)}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
    )

    def collate(batch):
        m = max(len(b["input_ids"]) for b in batch)
        pad_id = tok.pad_token_id
        ids, lbl, am = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            ids.append(b["input_ids"] + [pad_id] * (m - n))
            lbl.append(b["labels"] + [-100] * (m - n))
            am.append(b["attention_mask"] + [0] * (m - n))
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(lbl),
            "attention_mask": torch.tensor(am),
        }

    trainer = Trainer(
        model=model, args=targs, train_dataset=ds,
        tokenizer=tok, data_collator=collate,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
