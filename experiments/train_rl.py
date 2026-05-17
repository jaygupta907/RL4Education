"""RL fine-tuning of the SFT LoRA adapter with a configurable Claude reward.

Loads the supervised LoRA checkpoint (same format as ``train_sft.py``), samples
multiple completions per prompt, scores with ``--reward faithfulness``,
``feasibility``, or ``combined`` (both judges; PG uses weighted sum in ``[0,2]``
then scaled to ``[-1,1]``), and applies a GRPO-style policy gradient.

Outputs a new LoRA directory (see ``--output_dir``) for ``eval_rl.py`` / ``eval_pipeline``.

Requires ``ANTHROPIC_API_KEY`` and a trained SFT adapter at ``--sft_dir``.

GPU: single device via ``--cuda-device`` (default 0).

**Logging:** default ``<rl-log-dir>/rl_training.jsonl`` and ``rl_reward_loss.png``;
override paths with ``--rl-log-jsonl`` and ``--rl-plot-png`` (relative paths use cwd).

**Checkpoints:** by default saves ``<output_dir>/checkpoint-<step>`` every
``max(1, total_steps // 3)`` steps. Use ``--no-save-every-third`` to disable; ``--save-every N``
adds another interval.

**Optional KL:** ``--kl-beta B`` adds ``B * mean_t (log π_θ - log π_ref)`` on generated
tokens vs a **frozen** copy of the initial SFT adapter (same ``--sft_dir`` by default).
Use ``--kl-ref-cpu`` to load the reference on CPU to save GPU memory.

**W&B:** ``--wandb`` / ``--wandb_project`` as before.

Prompts include **target difficulty** in the user turn, same as SFT.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import statistics
from pathlib import Path

import torch
from peft import PeftModel
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from claude_client import ClaudeClient
from cot_utils import strip_reasoning_prefix
from judges import judge_faithfulness, judge_feasibility
from prompts import build_sft_chat_messages

HERE = Path(__file__).parent


def _resolve_domain(item):
    chapters = item.get("trace", {}).get("chapters") or []
    subdomains = item.get("trace", {}).get("subdomains") or []
    domain = item.get("domain") or (chapters[0] if chapters else "")
    subdomain = item.get("subdomain") or (subdomains[0] if subdomains else domain)
    return domain, subdomain


def tokens_for_difficulty(d: int, base: int, scale: int) -> int:
    """Match eval_pipeline: longer generations for higher difficulty."""
    return base + scale * max(0, d - 3)


def faithfulness_raw_score(faith: dict) -> float:
    """Composite faithfulness scalar from Claude judge outputs (roughly in ``[0, 2]``).

    Uses ``variable_coverage``, ``target_present``, and ``faithful`` (same structure
    as ``eval_pipeline`` / ``judge_faithfulness``).
    """
    cov = float(faith.get("variable_coverage") or 0.0)
    tgt = 1.0 if faith.get("target_present") else 0.0
    ful = 1.0 if faith.get("faithful") else 0.0
    return cov + 0.5 * tgt + 0.5 * ful


def feasibility_raw_score(score: int) -> float:
    """Map Claude feasibility (integer ``1``..``10``) onto ``[0, 2]`` (same band as faithfulness raw)."""
    try:
        s = int(score)
    except (TypeError, ValueError):
        return 0.0
    if s < 1 or s > 10:
        return 0.0
    return (s - 1) * (2.0 / 9.0)


def policy_scaled_reward(raw: float) -> float:
    """Linear map of raw reward from ``[0, 2]`` to ``[-1, 1]`` (PG advantage signal)."""
    return float(max(-1.0, min(1.0, raw - 1.0)))


def combined_raw_score(faith_raw: float, feas_raw: float, w_faith: float, w_feas: float) -> float:
    """Weighted sum of faithfulness and feasibility raw scores (each in ``[0,2]``)."""
    w = max(1e-8, w_faith + w_feas)
    return (w_faith / w) * float(faith_raw) + (w_feas / w) * float(feas_raw)


def build_prompt_tensor(item, tok, *, with_cot: bool) -> torch.Tensor:
    """Chat prompt matching SFT; user turn includes ``Target difficulty: d/10``."""
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
    return tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def generate_group(
    model,
    tok,
    prompt_ids: torch.Tensor,
    *,
    num_generations: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    """Return ``[num_generations, seq_len]`` token ids (left-padded batch)."""
    dev = _model_device(model)
    p = prompt_ids.to(dev)
    out = model.generate(
        p,
        num_return_sequences=num_generations,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=max(temperature, 1e-5),
        top_p=top_p,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    return out


def completion_logprob_sum(
    model,
    full_ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """Sum of log p(token | context) over generated tokens only (scalar tensor)."""
    dev = _model_device(model)
    ids = full_ids.to(dev).unsqueeze(0)
    out = model(ids, use_cache=False)
    logits = out.logits[0, :-1].float()
    targets = ids[0, 1:]
    logp = torch.log_softmax(logits, dim=-1)
    token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    start = max(0, prompt_len - 1)
    gen_logp = token_logp[start:]
    return gen_logp.sum()


def load_frozen_reference_peft(
    base: str,
    sft_dir: str,
    *,
    cuda_device: int,
    on_cpu: bool,
):
    """Frozen base + initial LoRA for KL(π_θ || π_ref) along sampled completions."""
    tok = AutoTokenizer.from_pretrained(sft_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    if on_cpu:
        device_map: dict | str = {"": "cpu"}
    elif torch.cuda.is_available():
        if cuda_device < 0 or cuda_device >= torch.cuda.device_count():
            raise SystemExit(
                f"--cuda-device {cuda_device} invalid for KL ref: "
                f"{torch.cuda.device_count()} CUDA device(s) visible."
            )
        device_map = {"": cuda_device}
    else:
        device_map = {"": "cpu"}

    load_kw = {"device_map": device_map}
    try:
        load_kw["dtype"] = torch.bfloat16
        base_model = AutoModelForCausalLM.from_pretrained(base, **load_kw)
    except TypeError:
        load_kw.pop("dtype", None)
        load_kw["torch_dtype"] = torch.bfloat16
        base_model = AutoModelForCausalLM.from_pretrained(base, **load_kw)
    base_model.config.use_cache = False

    if not Path(sft_dir).exists():
        raise SystemExit(f"KL ref SFT path not found: {sft_dir}")
    peft_kw = {}
    if "is_trainable" in inspect.signature(PeftModel.from_pretrained).parameters:
        peft_kw["is_trainable"] = False
    model = PeftModel.from_pretrained(base_model, sft_dir, **peft_kw)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model, tok


def load_trainable_sft(base: str, sft_dir: str, *, cuda_device: int = 0):
    tok = AutoTokenizer.from_pretrained(sft_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    if torch.cuda.is_available():
        if cuda_device < 0 or cuda_device >= torch.cuda.device_count():
            raise SystemExit(
                f"--cuda-device {cuda_device} invalid: "
                f"{torch.cuda.device_count()} CUDA device(s) visible."
            )
        device_map = {"": cuda_device}
    else:
        device_map = {"": "cpu"}

    load_kw = {"device_map": device_map}
    try:
        load_kw["dtype"] = torch.bfloat16
        base_model = AutoModelForCausalLM.from_pretrained(base, **load_kw)
    except TypeError:
        load_kw.pop("dtype", None)
        load_kw["torch_dtype"] = torch.bfloat16
        base_model = AutoModelForCausalLM.from_pretrained(base, **load_kw)
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()
    base_model.enable_input_require_grads()

    if not Path(sft_dir).exists():
        raise SystemExit(f"SFT adapter path not found: {sft_dir}")
    peft_kw = {}
    if "is_trainable" in inspect.signature(PeftModel.from_pretrained).parameters:
        peft_kw["is_trainable"] = True
    model = PeftModel.from_pretrained(base_model, sft_dir, **peft_kw)
    # Some PEFT versions still leave LoRA frozen after load; force trainable.
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable == 0:
        sample = [n for n, _ in list(model.named_parameters())[:16]]
        raise SystemExit(
            "No trainable LoRA parameters after load. "
            "train_rl.py expects a PEFT LoRA folder from train_sft.py "
            "(adapter_config.json and adapter weights), not a merged full model. "
            f"Example parameter names: {sample}"
        )
    model.train()
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(HERE / "data" / "dataset.json"))
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument(
        "--sft_dir",
        default="/mnt/storage/ae21b026/sft_lora",
        help="LoRA adapter from train_sft.py (starting policy).",
    )
    ap.add_argument(
        "--output_dir",
        default="/mnt/storage/ae21b026/rl_faithfulness_lora",
        help="Where to save the RL-updated LoRA adapter.",
    )
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--batch_size", type=int, default=1,
                    help="Dataset prompts per optimizer step.")
    ap.add_argument("--num_generations", type=int, default=4,
                    help="Samples per prompt for GRPO-style advantage.")
    ap.add_argument("--max_rl_examples", type=int, default=0,
                    help="If >0, only use this many shuffled dataset rows (debug).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen_temperature", type=float, default=0.8)
    ap.add_argument("--gen_top_p", type=float, default=0.95)
    ap.add_argument("--gen_tokens_base", type=int, default=350)
    ap.add_argument("--gen_tokens_scale", type=int, default=80)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--save_every", type=int, default=0,
                    help="Save adapter every N optimizer steps (0 = off). "
                    "Runs in addition to third-interval saves unless disabled.")
    ap.add_argument(
        "--no-save-every-third",
        action="store_true",
        help="Disable automatic saves every max(1, total_steps//3) steps (default: those saves are on).",
    )
    ap.add_argument(
        "--with-cot",
        action="store_true",
        help="Match train_sft.py / eval_pipeline.py --with-cot prompts.",
    )
    ap.add_argument(
        "--reward",
        type=str,
        choices=("faithfulness", "feasibility", "combined"),
        default="faithfulness",
        help="Claude signal(s): faithfulness composite, feasibility 1–10→[0,2], or "
        "both (see --combined-faith-weight / --combined-feas-weight).",
    )
    ap.add_argument(
        "--combined-faith-weight",
        type=float,
        default=0.5,
        help="Weight on faithfulness raw [0,2] when --reward combined (default 0.5).",
    )
    ap.add_argument(
        "--combined-feas-weight",
        type=float,
        default=0.5,
        help="Weight on feasibility raw [0,2] when --reward combined (default 0.5).",
    )
    ap.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help="Place the full model on this single CUDA device index (default: 0).",
    )
    ap.add_argument(
        "--wandb",
        dest="wandb_on",
        action="store_true",
        help="Log to Weights & Biases. Uses --wandb_project if set, else "
        "RL4Education-<reward> (e.g. RL4Education-feasibility).",
    )
    ap.add_argument(
        "--wandb_project",
        default="",
        help="W&B project name. If set (with or without --wandb), online logging is enabled.",
    )
    ap.add_argument(
        "--wandb_run_name",
        default="",
        help="Optional W&B run name (default: auto).",
    )
    ap.add_argument(
        "--rl-log-dir",
        default=str(HERE / "data" / "rl"),
        help="Default directory for rl_training.jsonl when --rl-log-jsonl is omitted.",
    )
    ap.add_argument(
        "--rl-log-jsonl",
        default="",
        help="Path to per-step JSONL log (default: <rl-log-dir>/rl_training.jsonl). "
        "Relative paths are resolved from the current working directory.",
    )
    ap.add_argument(
        "--rl-plot-png",
        default="",
        help="Path to the training-curve PNG (default: <log-dir>/rl_reward_loss.png). "
        "Relative paths are resolved from the current working directory.",
    )
    ap.add_argument(
        "--kl-beta",
        type=float,
        default=0.0,
        help="If >0, add KL penalty β * (log π_θ - log π_ref) summed over generated "
        "tokens (ref = frozen initial SFT LoRA). 0 disables.",
    )
    ap.add_argument(
        "--kl-ref-sft-dir",
        default="",
        help="LoRA folder for π_ref (default: same as --sft_dir).",
    )
    ap.add_argument(
        "--kl-ref-cpu",
        action="store_true",
        help="Load π_ref on CPU (saves GPU RAM; KL forward is slower).",
    )
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    rl_log_dir = Path(args.rl_log_dir)

    log_arg = (args.rl_log_jsonl or "").strip()
    if log_arg:
        log_path = Path(log_arg).expanduser()
        if not log_path.is_absolute():
            log_path = (Path.cwd() / log_path).resolve()
    else:
        rl_log_dir.mkdir(parents=True, exist_ok=True)
        log_path = rl_log_dir / "rl_training.jsonl"

    plot_arg = (args.rl_plot_png or "").strip()
    if plot_arg:
        plot_path = Path(plot_arg).expanduser()
        if not plot_path.is_absolute():
            plot_path = (Path.cwd() / plot_path).resolve()
    else:
        plot_path = log_path.parent / "rl_reward_loss.png"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.dataset) as f:
        data = json.load(f)
    if not data:
        raise SystemExit("Dataset is empty.")

    rng = random.Random(args.seed)
    if args.max_rl_examples > 0:
        data = data[:]
        rng.shuffle(data)
        data = data[: args.max_rl_examples]

    model, tok = load_trainable_sft(
        args.base_model, args.sft_dir, cuda_device=args.cuda_device
    )
    judge = ClaudeClient()
    ref_model = None
    kl_beta = float(args.kl_beta)
    if kl_beta > 0.0:
        ref_dir = (args.kl_ref_sft_dir or "").strip() or args.sft_dir
        print(
            f"KL penalty: beta={kl_beta}, ref_dir={ref_dir}, ref_on_cpu={args.kl_ref_cpu}",
            flush=True,
        )
        ref_model, _ = load_frozen_reference_peft(
            args.base_model,
            ref_dir,
            cuda_device=args.cuda_device,
            on_cpu=bool(args.kl_ref_cpu),
        )
    if torch.cuda.is_available():
        print(
            f"Using CUDA device {args.cuda_device}: "
            f"{torch.cuda.get_device_name(args.cuda_device)}",
            flush=True,
        )

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise SystemExit("No trainable parameters (LoRA missing?).")
    opt = AdamW(params, lr=args.lr)

    steps_per_epoch = max(1, math.ceil(len(data) / args.batch_size))
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    print(
        f"RL: {len(data)} examples, batch_size={args.batch_size}, "
        f"num_generations={args.num_generations}, ~{total_steps} steps",
        flush=True,
    )
    print(f"Reward signal: {args.reward}", flush=True)

    third_save_period = 0
    if not args.no_save_every_third:
        third_save_period = max(1, total_steps // 3)
        print(
            f"Periodic checkpoints: every {third_save_period} step(s) "
            f"(~1/3 of {total_steps} total steps)",
            flush=True,
        )

    log_path.write_text("", encoding="utf-8")
    print(f"RL metrics log: {log_path}", flush=True)
    print(f"RL training plot: {plot_path}", flush=True)

    wb = None
    wb_project = args.wandb_project.strip()
    if getattr(args, "wandb_on", False) and not wb_project:
        wb_project = f"RL4Education-{args.reward}"
    if wb_project:
        try:
            import wandb

            wb = wandb.init(
                project=wb_project,
                name=(args.wandb_run_name.strip() or None),
                config={
                    k: v
                    for k, v in vars(args).items()
                    if isinstance(v, (int, float, str, bool))
                },
            )
            print(f"Weights & Biases: project={wb_project}", flush=True)
        except ImportError:
            print(
                "Warning: wandb not installed; `pip install wandb` to log online.",
                flush=True,
            )

    global_step = 0
    data_indices = list(range(len(data)))
    epoch = 0
    try:
        while global_step < total_steps:
            rng.shuffle(data_indices)
            for start in range(0, len(data), args.batch_size):
                if global_step >= total_steps:
                    break
                batch_idx = data_indices[start : start + args.batch_size]
                opt.zero_grad()
                per_prompt_losses = []
                kl_samples: list[float] = []
                all_rollout_raw: list[float] = []
                all_rollout_scaled: list[float] = []
                all_feas_claude: list[int] = []
                all_faith_raw_rollouts: list[float] = []
                all_feas_raw_rollouts: list[float] = []
                w_faith = float(args.combined_faith_weight)
                w_feas = float(args.combined_feas_weight)
                for bi in batch_idx:
                    item = data[bi]
                    d = int(item.get("requested_difficulty") or 5)
                    prompt_ids = build_prompt_tensor(item, tok, with_cot=args.with_cot)
                    prompt_len = int(prompt_ids.shape[1])
                    mnt = tokens_for_difficulty(
                        d, args.gen_tokens_base, args.gen_tokens_scale
                    )
                    if args.with_cot:
                        mnt += 512

                    was_training = model.training
                    model.eval()
                    with torch.no_grad():
                        gen_batch = generate_group(
                            model,
                            tok,
                            prompt_ids,
                            num_generations=args.num_generations,
                            max_new_tokens=mnt,
                            temperature=args.gen_temperature,
                            top_p=args.gen_top_p,
                        )
                    if was_training:
                        model.train()

                    rewards_raw: list[float] = []
                    rewards_scaled: list[float] = []
                    feas_batch: list[int] = []
                    faith_batch: list[float] = []
                    feas_raw_batch: list[float] = []
                    for gi in range(args.num_generations):
                        full = gen_batch[gi]
                        gen_tokens = full[prompt_len:]
                        if gen_tokens.numel() == 0:
                            rewards_raw.append(0.0)
                            rewards_scaled.append(policy_scaled_reward(0.0))
                            if args.reward in ("feasibility", "combined"):
                                feas_batch.append(-1)
                            if args.reward == "combined":
                                faith_batch.append(0.0)
                                feas_raw_batch.append(0.0)
                            continue
                        raw_text = tok.decode(
                            gen_tokens, skip_special_tokens=True
                        ).strip()
                        q = strip_reasoning_prefix(raw_text)
                        if args.reward == "faithfulness":
                            faith = judge_faithfulness(
                                judge,
                                item["trace_str"],
                                item["trace"]["leafs"],
                                item["target"],
                                q,
                            )
                            rscore = faithfulness_raw_score(faith)
                            rewards_raw.append(rscore)
                            rewards_scaled.append(policy_scaled_reward(rscore))
                        elif args.reward == "feasibility":
                            s = judge_feasibility(
                                judge,
                                item["trace_str"],
                                item["trace"]["leafs"],
                                item["target"],
                                q,
                            )
                            si = int(s) if isinstance(s, int) else -1
                            feas_batch.append(si)
                            rscore = feasibility_raw_score(si)
                            rewards_raw.append(rscore)
                            rewards_scaled.append(policy_scaled_reward(rscore))
                        else:
                            faith = judge_faithfulness(
                                judge,
                                item["trace_str"],
                                item["trace"]["leafs"],
                                item["target"],
                                q,
                            )
                            s = judge_feasibility(
                                judge,
                                item["trace_str"],
                                item["trace"]["leafs"],
                                item["target"],
                                q,
                            )
                            si = int(s) if isinstance(s, int) else -1
                            feas_batch.append(si)
                            fr = faithfulness_raw_score(faith)
                            fer = feasibility_raw_score(si)
                            faith_batch.append(fr)
                            feas_raw_batch.append(fer)
                            cr = combined_raw_score(fr, fer, w_faith, w_feas)
                            rewards_raw.append(cr)
                            rewards_scaled.append(policy_scaled_reward(cr))

                    if args.reward in ("feasibility", "combined"):
                        all_feas_claude.extend(feas_batch)
                    if args.reward == "combined":
                        all_faith_raw_rollouts.extend(faith_batch)
                        all_feas_raw_rollouts.extend(feas_raw_batch)

                    r = torch.tensor(
                        rewards_scaled,
                        device=_model_device(model),
                        dtype=torch.float32,
                    )
                    adv = r - r.mean()
                    std = float(r.std(unbiased=False).item())
                    if std < 1e-8:
                        adv = torch.zeros_like(adv)

                    pg_terms = []
                    for gi in range(args.num_generations):
                        a = adv[gi].detach()
                        lp = completion_logprob_sum(model, gen_batch[gi], prompt_len)
                        pg_part = -(a * lp.float())
                        if ref_model is not None and kl_beta > 0.0:
                            with torch.no_grad():
                                lp_ref = completion_logprob_sum(
                                    ref_model, gen_batch[gi], prompt_len
                                )
                            lp_rf = lp_ref.to(lp.device).float()
                            kl_row = lp.float() - lp_rf
                            kl_samples.append(float(kl_row.detach().cpu().item()))
                            pg_terms.append(pg_part + kl_beta * kl_row)
                        else:
                            if abs(float(a.item())) < 1e-12:
                                continue
                            pg_terms.append(pg_part)

                    all_rollout_raw.extend(float(x) for x in rewards_raw)
                    all_rollout_scaled.extend(float(x) for x in rewards_scaled)
                    if pg_terms:
                        per_prompt_losses.append(torch.stack(pg_terms).mean())

                if per_prompt_losses:
                    loss = torch.stack(per_prompt_losses).mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                    opt.step()
                    loss_val = float(loss.detach().item())
                else:
                    loss_val = None

                global_step += 1
                mean_raw = (
                    statistics.mean(all_rollout_raw) if all_rollout_raw else 0.0
                )
                mean_scaled = (
                    statistics.mean(all_rollout_scaled)
                    if all_rollout_scaled
                    else 0.0
                )
                reward_std_scaled = (
                    statistics.pstdev(all_rollout_scaled)
                    if len(all_rollout_scaled) > 1
                    else 0.0
                )
                reward_std_raw = (
                    statistics.pstdev(all_rollout_raw)
                    if len(all_rollout_raw) > 1
                    else 0.0
                )
                mean_kl = (
                    statistics.mean(kl_samples) if kl_samples else None
                )
                log_rec: dict = {
                    "step": global_step,
                    "reward_kind": args.reward,
                    "kl_beta": kl_beta,
                    "mean_kl_sample": mean_kl,
                    "mean_reward_raw": mean_raw,
                    "mean_reward_scaled": mean_scaled,
                    "mean_reward": mean_scaled,
                    "reward_std": reward_std_scaled,
                    "reward_std_raw": reward_std_raw,
                    "loss": loss_val,
                    "skipped_grad": loss_val is None,
                    "reward_raw": all_rollout_raw,
                    "rewards": all_rollout_scaled,
                    "batch_size": len(batch_idx),
                }
                if args.reward == "faithfulness":
                    log_rec["mean_faithfulness_raw"] = mean_raw
                    log_rec["faithfulness_raw"] = all_rollout_raw
                elif args.reward == "feasibility":
                    valid_fc = [x for x in all_feas_claude if isinstance(x, int) and x >= 1]
                    log_rec["feasibility_claude_scores"] = all_feas_claude
                    log_rec["mean_feasibility_claude"] = (
                        float(statistics.mean(valid_fc)) if valid_fc else None
                    )
                else:
                    log_rec["combined_faith_weight"] = w_faith
                    log_rec["combined_feas_weight"] = w_feas
                    log_rec["faithfulness_raw_rollout"] = all_faith_raw_rollouts
                    log_rec["feasibility_raw_rollout"] = all_feas_raw_rollouts
                    log_rec["combined_raw_rollout"] = all_rollout_raw
                    log_rec["mean_faithfulness_raw"] = (
                        statistics.mean(all_faith_raw_rollouts)
                        if all_faith_raw_rollouts
                        else 0.0
                    )
                    log_rec["mean_feasibility_raw"] = (
                        statistics.mean(all_feas_raw_rollouts)
                        if all_feas_raw_rollouts
                        else 0.0
                    )
                    log_rec["mean_combined_raw"] = mean_raw
                    valid_fc = [x for x in all_feas_claude if isinstance(x, int) and x >= 1]
                    log_rec["feasibility_claude_scores"] = all_feas_claude
                    log_rec["mean_feasibility_claude"] = (
                        float(statistics.mean(valid_fc)) if valid_fc else None
                    )
                with open(log_path, "a", encoding="utf-8") as logf:
                    logf.write(json.dumps(log_rec) + "\n")

                if wb is not None:
                    import wandb

                    payload = {
                        "ppo/reward_kind": args.reward,
                        "ppo/mean_reward_raw": mean_raw,
                        "ppo/mean_reward_scaled": mean_scaled,
                        "ppo/mean_reward": mean_scaled,
                        "ppo/reward_std_scaled": reward_std_scaled,
                        "ppo/reward_std_raw": reward_std_raw,
                        "ppo/skipped_grad": int(loss_val is None),
                    }
                    if args.reward == "faithfulness":
                        payload["ppo/mean_faithfulness_raw"] = mean_raw
                    elif args.reward == "feasibility":
                        valid_fc = [
                            x for x in all_feas_claude
                            if isinstance(x, int) and x >= 1
                        ]
                        if valid_fc:
                            payload["ppo/mean_feasibility_claude"] = float(
                                statistics.mean(valid_fc)
                            )
                    else:
                        payload["ppo/mean_faithfulness_raw"] = (
                            statistics.mean(all_faith_raw_rollouts)
                            if all_faith_raw_rollouts
                            else 0.0
                        )
                        payload["ppo/mean_feasibility_raw"] = (
                            statistics.mean(all_feas_raw_rollouts)
                            if all_feas_raw_rollouts
                            else 0.0
                        )
                        payload["ppo/mean_combined_raw"] = mean_raw
                        valid_fc = [
                            x for x in all_feas_claude
                            if isinstance(x, int) and x >= 1
                        ]
                        if valid_fc:
                            payload["ppo/mean_feasibility_claude"] = float(
                                statistics.mean(valid_fc)
                            )
                    if loss_val is not None:
                        payload["ppo/pg_loss"] = loss_val
                    if mean_kl is not None:
                        payload["ppo/mean_kl_sample"] = mean_kl
                    wandb.log(payload, step=global_step)

                kl_s = (
                    f" kl_sample_mean={mean_kl:.4f}"
                    if mean_kl is not None
                    else ""
                )

                if loss_val is not None:
                    if args.reward == "faithfulness":
                        print(
                            f"step {global_step}/{total_steps} "
                            f"[faithfulness] raw={mean_raw:.4f} (std={reward_std_raw:.4f}) "
                            f"scaled={mean_scaled:.4f} (std={reward_std_scaled:.4f}) "
                            f"ppo_loss={loss_val:.4f}{kl_s}",
                            flush=True,
                        )
                    elif args.reward == "feasibility":
                        valid_fc = [
                            x for x in all_feas_claude
                            if isinstance(x, int) and x >= 1
                        ]
                        mfc = (
                            statistics.mean(valid_fc) if valid_fc else float("nan")
                        )
                        print(
                            f"step {global_step}/{total_steps} "
                            f"[feasibility] claude_mean={mfc:.2f} "
                            f"raw={mean_raw:.4f} (std={reward_std_raw:.4f}) "
                            f"scaled={mean_scaled:.4f} (std={reward_std_scaled:.4f}) "
                            f"ppo_loss={loss_val:.4f}{kl_s}",
                            flush=True,
                        )
                    else:
                        mf = (
                            statistics.mean(all_faith_raw_rollouts)
                            if all_faith_raw_rollouts
                            else 0.0
                        )
                        mfe = (
                            statistics.mean(all_feas_raw_rollouts)
                            if all_feas_raw_rollouts
                            else 0.0
                        )
                        print(
                            f"step {global_step}/{total_steps} "
                            f"[combined w_f={w_faith:.2f} w_e={w_feas:.2f}] "
                            f"faith_raw={mf:.4f} feas_raw={mfe:.4f} "
                            f"comb_raw={mean_raw:.4f} scaled={mean_scaled:.4f} "
                            f"ppo_loss={loss_val:.4f}{kl_s}",
                            flush=True,
                        )
                else:
                    if args.reward == "faithfulness":
                        print(
                            f"step {global_step}/{total_steps} "
                            f"[faithfulness] raw={mean_raw:.4f} (std={reward_std_raw:.4f}) "
                            f"scaled={mean_scaled:.4f} (std={reward_std_scaled:.4f}) "
                            f"ppo_loss=skipped (no grad){kl_s}",
                            flush=True,
                        )
                    elif args.reward == "feasibility":
                        valid_fc = [
                            x for x in all_feas_claude
                            if isinstance(x, int) and x >= 1
                        ]
                        mfc = (
                            statistics.mean(valid_fc) if valid_fc else float("nan")
                        )
                        print(
                            f"step {global_step}/{total_steps} "
                            f"[feasibility] claude_mean={mfc:.2f} "
                            f"raw={mean_raw:.4f} (std={reward_std_raw:.4f}) "
                            f"scaled={mean_scaled:.4f} (std={reward_std_scaled:.4f}) "
                            f"ppo_loss=skipped (no grad){kl_s}",
                            flush=True,
                        )
                    else:
                        mf = (
                            statistics.mean(all_faith_raw_rollouts)
                            if all_faith_raw_rollouts
                            else 0.0
                        )
                        mfe = (
                            statistics.mean(all_feas_raw_rollouts)
                            if all_feas_raw_rollouts
                            else 0.0
                        )
                        print(
                            f"step {global_step}/{total_steps} "
                            f"[combined w_f={w_faith:.2f} w_e={w_feas:.2f}] "
                            f"faith_raw={mf:.4f} feas_raw={mfe:.4f} "
                            f"comb_raw={mean_raw:.4f} scaled={mean_scaled:.4f} "
                            f"ppo_loss=skipped (no grad){kl_s}",
                            flush=True,
                        )

                do_save = False
                if args.save_every > 0 and global_step % args.save_every == 0:
                    do_save = True
                if third_save_period > 0 and global_step % third_save_period == 0:
                    do_save = True
                if do_save:
                    ck = Path(args.output_dir) / f"checkpoint-{global_step}"
                    ck.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(ck)
                    tok.save_pretrained(ck)
                    print(f"  saved {ck}", flush=True)

            epoch += 1

        model.save_pretrained(args.output_dir)
        tok.save_pretrained(args.output_dir)
        print(f"Saved RL LoRA adapter to {args.output_dir}", flush=True)

        from metrics import plot_rl_combined_training_metrics, plot_rl_training_metrics

        rows_log: list[dict] = []
        with open(log_path, encoding="utf-8") as logf:
            for line in logf:
                line = line.strip()
                if line:
                    rows_log.append(json.loads(line))
        plot_ok = False
        if args.reward == "combined":
            plot_ok = plot_rl_combined_training_metrics(rows_log, str(plot_path))
        else:
            plot_ok = plot_rl_training_metrics(rows_log, str(plot_path))
        if plot_ok:
            print(f"Saved RL training plot to {plot_path}", flush=True)
        else:
            print("Skipping RL training plot (empty or invalid log).", flush=True)

        if wb is not None:
            import wandb

            if plot_path.exists():
                try:
                    wandb.log({"ppo/rl_curves": wandb.Image(str(plot_path))})
                except Exception as e:
                    print(f"Warning: could not log plot to wandb: {e}", flush=True)
    finally:
        if wb is not None:
            import wandb

            wandb.finish()


if __name__ == "__main__":
    main()
