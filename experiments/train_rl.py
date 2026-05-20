"""RL fine-tuning of the SFT LoRA adapter with a configurable Claude reward.

Loads the supervised LoRA checkpoint (same format as ``train_sft.py``), samples
multiple completions per prompt, scores with ``--reward faithfulness``,
``feasibility``, ``combined``, ``alternating``, or ``adaptive`` (faithfulness +
feasibility + difficulty alignment with EMA-adaptive weights).

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
from judges import judge_difficulty, judge_faithfulness, judge_feasibility
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


def difficulty_alignment_raw_score(requested_d: int, claude_d: int) -> float:
    """Map requested vs Claude-judged difficulty onto ``[0, 2]`` (perfect match → 2)."""
    try:
        req = int(requested_d)
        got = int(claude_d)
    except (TypeError, ValueError):
        return 0.0
    if req < 1 or req > 10 or got < 1 or got > 10:
        return 0.0
    err = abs(req - got)
    return max(0.0, 2.0 * (1.0 - err / 9.0))


def adaptive_component_weights(
    ema_faith: float,
    ema_feas: float,
    ema_diff: float,
    *,
    eps: float,
) -> tuple[float, float, float]:
    """Up-weight components with lower running EMA (weaker objectives get more gradient signal)."""
    inv = [
        1.0 / (float(ema_faith) + eps),
        1.0 / (float(ema_feas) + eps),
        1.0 / (float(ema_diff) + eps),
    ]
    s = sum(inv)
    return inv[0] / s, inv[1] / s, inv[2] / s


def triple_adaptive_raw(
    faith_raw: float,
    feas_raw: float,
    diff_raw: float,
    w_faith: float,
    w_feas: float,
    w_diff: float,
) -> float:
    return (
        w_faith * float(faith_raw)
        + w_feas * float(feas_raw)
        + w_diff * float(diff_raw)
    )


def resolve_active_reward(reward_mode: str, step_num: int, *, switch_steps: int) -> str:
    """Which Claude signal to use on optimizer step ``step_num`` (1-based).

    For ``alternating``, blocks of ``switch_steps`` use faithfulness, then feasibility,
    then faithfulness, etc.
    """
    if reward_mode != "alternating":
        return reward_mode
    period = max(1, int(switch_steps))
    phase = (max(1, step_num) - 1) // period
    return "faithfulness" if phase % 2 == 0 else "feasibility"


def score_completion_reward(
    judge,
    item: dict,
    question: str,
    active_reward: str,
    *,
    w_faith: float,
    w_feas: float,
) -> tuple[float, int | None, float | None, float | None]:
    """Return (raw_score, feas_claude_1_10_or_None, faith_raw_or_None, feas_raw_or_None)."""
    trace_str = item["trace_str"]
    leafs = item["trace"]["leafs"]
    target = item["target"]
    if active_reward == "faithfulness":
        faith = judge_faithfulness(judge, trace_str, leafs, target, question)
        r = faithfulness_raw_score(faith)
        return r, None, r, None
    if active_reward == "feasibility":
        s = judge_feasibility(judge, trace_str, leafs, target, question)
        si = int(s) if isinstance(s, int) else -1
        fer = feasibility_raw_score(si)
        return fer, si, None, fer
    faith = judge_faithfulness(judge, trace_str, leafs, target, question)
    s = judge_feasibility(judge, trace_str, leafs, target, question)
    si = int(s) if isinstance(s, int) else -1
    fr = faithfulness_raw_score(faith)
    fer = feasibility_raw_score(si)
    cr = combined_raw_score(fr, fer, w_faith, w_feas)
    return cr, si, fr, fer


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
        choices=("faithfulness", "feasibility", "combined", "alternating", "adaptive"),
        default="faithfulness",
        help="Claude signal(s): faithfulness, feasibility, combined (fixed 2-way), "
        "alternating, or adaptive (3 judges + EMA-adaptive weights).",
    )
    ap.add_argument(
        "--reward-switch-steps",
        type=int,
        default=0,
        help="When --reward alternating: optimizer steps per phase before switching "
        "(faithfulness then feasibility). 0 = max(1, total_steps // 6).",
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
        "--adaptive-ema-decay",
        type=float,
        default=0.9,
        help="EMA decay for component means when --reward adaptive (higher = slower weight changes).",
    )
    ap.add_argument(
        "--adaptive-weight-eps",
        type=float,
        default=0.2,
        help="Stabilizer in adaptive inverse weights: w_i ∝ 1/(EMA_i + eps).",
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

    reward_switch_period = 0
    if args.reward == "alternating":
        reward_switch_period = (
            max(1, int(args.reward_switch_steps))
            if int(args.reward_switch_steps) > 0
            else max(1, total_steps // 6)
        )
        print(
            f"Alternating reward: faithfulness for steps 1–{reward_switch_period}, "
            f"then feasibility for {reward_switch_period} steps, then repeat "
            f"(period={reward_switch_period})",
            flush=True,
        )

    adaptive_ema_decay = float(args.adaptive_ema_decay)
    adaptive_weight_eps = float(args.adaptive_weight_eps)
    ema_faith_adapt = ema_feas_adapt = ema_diff_adapt = 1.0
    w_f_adapt, w_e_adapt, w_d_adapt = 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    if args.reward == "adaptive":
        print(
            f"Adaptive reward: faithfulness + feasibility + difficulty alignment; "
            f"weights from EMA (decay={adaptive_ema_decay}, eps={adaptive_weight_eps}); "
            f"initial w=({w_f_adapt:.3f},{w_e_adapt:.3f},{w_d_adapt:.3f})",
            flush=True,
        )

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
                all_diff_raw_rollouts: list[float] = []
                w_faith = float(args.combined_faith_weight)
                w_feas = float(args.combined_feas_weight)
                step_num = global_step + 1
                switch_steps = (
                    reward_switch_period
                    if args.reward == "alternating"
                    else 1
                )
                active_reward = resolve_active_reward(
                    args.reward, step_num, switch_steps=switch_steps
                )
                if (
                    args.reward == "alternating"
                    and reward_switch_period > 0
                    and (step_num - 1) % reward_switch_period == 0
                ):
                    print(
                        f"  >>> reward phase: {active_reward} "
                        f"(optimizer steps {step_num}–"
                        f"{min(step_num + reward_switch_period - 1, total_steps)})",
                        flush=True,
                    )
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
                    diff_batch: list[int] = []
                    diff_raw_batch: list[float] = []
                    for gi in range(args.num_generations):
                        full = gen_batch[gi]
                        gen_tokens = full[prompt_len:]
                        if gen_tokens.numel() == 0:
                            rewards_raw.append(0.0)
                            rewards_scaled.append(policy_scaled_reward(0.0))
                            if active_reward in ("feasibility",) or args.reward == "adaptive":
                                feas_batch.append(-1)
                            if active_reward == "combined":
                                faith_batch.append(0.0)
                                feas_raw_batch.append(0.0)
                            if args.reward == "adaptive":
                                faith_batch.append(0.0)
                                feas_raw_batch.append(0.0)
                                diff_batch.append(-1)
                                diff_raw_batch.append(0.0)
                            continue
                        raw_text = tok.decode(
                            gen_tokens, skip_special_tokens=True
                        ).strip()
                        q = strip_reasoning_prefix(raw_text)
                        if args.reward == "adaptive":
                            req_d = int(item.get("requested_difficulty") or 5)
                            faith = judge_faithfulness(
                                judge,
                                item["trace_str"],
                                item["trace"]["leafs"],
                                item["target"],
                                q,
                            )
                            s_feas = judge_feasibility(
                                judge,
                                item["trace_str"],
                                item["trace"]["leafs"],
                                item["target"],
                                q,
                            )
                            cd = judge_difficulty(
                                judge,
                                item["trace_str"],
                                item["trace"]["leafs"],
                                item["target"],
                                q,
                            )
                            fr = faithfulness_raw_score(faith)
                            si = int(s_feas) if isinstance(s_feas, int) else -1
                            fer = feasibility_raw_score(si)
                            dr = difficulty_alignment_raw_score(req_d, cd)
                            cr = triple_adaptive_raw(
                                fr, fer, dr, w_f_adapt, w_e_adapt, w_d_adapt
                            )
                            rewards_raw.append(cr)
                            rewards_scaled.append(policy_scaled_reward(cr))
                            faith_batch.append(fr)
                            feas_raw_batch.append(fer)
                            diff_raw_batch.append(dr)
                            diff_batch.append(
                                int(cd) if isinstance(cd, int) else -1
                            )
                            feas_batch.append(si)
                        else:
                            rscore, si, fr, fer = score_completion_reward(
                                judge,
                                item,
                                q,
                                active_reward,
                                w_faith=w_faith,
                                w_feas=w_feas,
                            )
                            rewards_raw.append(rscore)
                            rewards_scaled.append(policy_scaled_reward(rscore))
                            if si is not None:
                                feas_batch.append(si)
                            if fr is not None:
                                faith_batch.append(fr)
                            if fer is not None and active_reward == "combined":
                                feas_raw_batch.append(fer)

                    if args.reward == "adaptive":
                        all_feas_claude.extend(feas_batch)
                        all_faith_raw_rollouts.extend(faith_batch)
                        all_feas_raw_rollouts.extend(feas_raw_batch)
                        all_diff_raw_rollouts.extend(diff_raw_batch)
                    elif active_reward in ("feasibility",):
                        all_feas_claude.extend(feas_batch)
                    if active_reward == "combined":
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
                    "active_reward": active_reward,
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
                if args.reward == "alternating":
                    log_rec["reward_switch_steps"] = reward_switch_period
                if args.reward == "adaptive":
                    mf_ad = (
                        statistics.mean(all_faith_raw_rollouts)
                        if all_faith_raw_rollouts
                        else 0.0
                    )
                    me_ad = (
                        statistics.mean(all_feas_raw_rollouts)
                        if all_feas_raw_rollouts
                        else 0.0
                    )
                    md_ad = (
                        statistics.mean(all_diff_raw_rollouts)
                        if all_diff_raw_rollouts
                        else 0.0
                    )
                    log_rec["weight_faithfulness"] = w_f_adapt
                    log_rec["weight_feasibility"] = w_e_adapt
                    log_rec["weight_difficulty"] = w_d_adapt
                    log_rec["ema_faithfulness"] = ema_faith_adapt
                    log_rec["ema_feasibility"] = ema_feas_adapt
                    log_rec["ema_difficulty"] = ema_diff_adapt
                    log_rec["faithfulness_raw_rollout"] = all_faith_raw_rollouts
                    log_rec["feasibility_raw_rollout"] = all_feas_raw_rollouts
                    log_rec["difficulty_raw_rollout"] = all_diff_raw_rollouts
                    log_rec["adaptive_combined_raw_rollout"] = all_rollout_raw
                    log_rec["mean_faithfulness_raw"] = mf_ad
                    log_rec["mean_feasibility_raw"] = me_ad
                    log_rec["mean_difficulty_raw"] = md_ad
                    log_rec["mean_adaptive_combined_raw"] = mean_raw
                    valid_fc = [
                        x for x in all_feas_claude if isinstance(x, int) and x >= 1
                    ]
                    log_rec["feasibility_claude_scores"] = all_feas_claude
                    log_rec["mean_feasibility_claude"] = (
                        float(statistics.mean(valid_fc)) if valid_fc else None
                    )
                    ema_faith_adapt = (
                        adaptive_ema_decay * ema_faith_adapt
                        + (1.0 - adaptive_ema_decay) * mf_ad
                    )
                    ema_feas_adapt = (
                        adaptive_ema_decay * ema_feas_adapt
                        + (1.0 - adaptive_ema_decay) * me_ad
                    )
                    ema_diff_adapt = (
                        adaptive_ema_decay * ema_diff_adapt
                        + (1.0 - adaptive_ema_decay) * md_ad
                    )
                    w_f_adapt, w_e_adapt, w_d_adapt = adaptive_component_weights(
                        ema_faith_adapt,
                        ema_feas_adapt,
                        ema_diff_adapt,
                        eps=adaptive_weight_eps,
                    )
                log_kind = active_reward
                if args.reward == "adaptive":
                    pass
                elif log_kind == "faithfulness":
                    log_rec["mean_faithfulness_raw"] = mean_raw
                    log_rec["faithfulness_raw"] = all_rollout_raw
                elif log_kind == "feasibility":
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
                        "ppo/active_reward": active_reward,
                        "ppo/mean_reward_raw": mean_raw,
                        "ppo/mean_reward_scaled": mean_scaled,
                        "ppo/mean_reward": mean_scaled,
                        "ppo/reward_std_scaled": reward_std_scaled,
                        "ppo/reward_std_raw": reward_std_raw,
                        "ppo/skipped_grad": int(loss_val is None),
                    }
                    if args.reward == "adaptive":
                        payload["ppo/mean_faithfulness_raw"] = log_rec.get(
                            "mean_faithfulness_raw", 0.0
                        )
                        payload["ppo/mean_feasibility_raw"] = log_rec.get(
                            "mean_feasibility_raw", 0.0
                        )
                        payload["ppo/mean_difficulty_raw"] = log_rec.get(
                            "mean_difficulty_raw", 0.0
                        )
                        payload["ppo/mean_adaptive_combined_raw"] = mean_raw
                        payload["ppo/weight_faithfulness"] = log_rec.get(
                            "weight_faithfulness", 0.0
                        )
                        payload["ppo/weight_feasibility"] = log_rec.get(
                            "weight_feasibility", 0.0
                        )
                        payload["ppo/weight_difficulty"] = log_rec.get(
                            "weight_difficulty", 0.0
                        )
                        valid_fc = [
                            x for x in all_feas_claude
                            if isinstance(x, int) and x >= 1
                        ]
                        if valid_fc:
                            payload["ppo/mean_feasibility_claude"] = float(
                                statistics.mean(valid_fc)
                            )
                    elif log_kind == "faithfulness":
                        payload["ppo/mean_faithfulness_raw"] = mean_raw
                    elif log_kind == "feasibility":
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
                if args.reward == "adaptive":
                    wf = float(log_rec.get("weight_faithfulness", w_f_adapt))
                    we = float(log_rec.get("weight_feasibility", w_e_adapt))
                    wd = float(log_rec.get("weight_difficulty", w_d_adapt))
                    tag = f"adaptive w=({wf:.2f},{we:.2f},{wd:.2f})"
                elif args.reward == "alternating":
                    tag = f"alternating→{active_reward}"
                else:
                    tag = active_reward

                if loss_val is not None:
                    if args.reward == "adaptive":
                        mf = log_rec.get("mean_faithfulness_raw", 0.0)
                        mfe = log_rec.get("mean_feasibility_raw", 0.0)
                        md = log_rec.get("mean_difficulty_raw", 0.0)
                        print(
                            f"step {global_step}/{total_steps} "
                            f"[{tag}] "
                            f"faith_raw={mf:.4f} feas_raw={mfe:.4f} diff_raw={md:.4f} "
                            f"comb_raw={mean_raw:.4f} scaled={mean_scaled:.4f} "
                            f"ppo_loss={loss_val:.4f}{kl_s}",
                            flush=True,
                        )
                    elif log_kind == "faithfulness":
                        print(
                            f"step {global_step}/{total_steps} "
                            f"[{tag}] raw={mean_raw:.4f} (std={reward_std_raw:.4f}) "
                            f"scaled={mean_scaled:.4f} (std={reward_std_scaled:.4f}) "
                            f"ppo_loss={loss_val:.4f}{kl_s}",
                            flush=True,
                        )
                    elif log_kind == "feasibility":
                        valid_fc = [
                            x for x in all_feas_claude
                            if isinstance(x, int) and x >= 1
                        ]
                        mfc = (
                            statistics.mean(valid_fc) if valid_fc else float("nan")
                        )
                        print(
                            f"step {global_step}/{total_steps} "
                            f"[{tag}] claude_mean={mfc:.2f} "
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
                            f"[{tag} w_f={w_faith:.2f} w_e={w_feas:.2f}] "
                            f"faith_raw={mf:.4f} feas_raw={mfe:.4f} "
                            f"comb_raw={mean_raw:.4f} scaled={mean_scaled:.4f} "
                            f"ppo_loss={loss_val:.4f}{kl_s}",
                            flush=True,
                        )
                else:
                    if args.reward == "adaptive":
                        mf = log_rec.get("mean_faithfulness_raw", 0.0)
                        mfe = log_rec.get("mean_feasibility_raw", 0.0)
                        md = log_rec.get("mean_difficulty_raw", 0.0)
                        print(
                            f"step {global_step}/{total_steps} "
                            f"[{tag}] "
                            f"faith_raw={mf:.4f} feas_raw={mfe:.4f} diff_raw={md:.4f} "
                            f"comb_raw={mean_raw:.4f} scaled={mean_scaled:.4f} "
                            f"ppo_loss=skipped (no grad){kl_s}",
                            flush=True,
                        )
                    elif log_kind == "faithfulness":
                        print(
                            f"step {global_step}/{total_steps} "
                            f"[{tag}] raw={mean_raw:.4f} (std={reward_std_raw:.4f}) "
                            f"scaled={mean_scaled:.4f} (std={reward_std_scaled:.4f}) "
                            f"ppo_loss=skipped (no grad){kl_s}",
                            flush=True,
                        )
                    elif log_kind == "feasibility":
                        valid_fc = [
                            x for x in all_feas_claude
                            if isinstance(x, int) and x >= 1
                        ]
                        mfc = (
                            statistics.mean(valid_fc) if valid_fc else float("nan")
                        )
                        print(
                            f"step {global_step}/{total_steps} "
                            f"[{tag}] claude_mean={mfc:.2f} "
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
                            f"[{tag} w_f={w_faith:.2f} w_e={w_feas:.2f}] "
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

        from metrics import (
            plot_rl_adaptive_combined_metrics,
            plot_rl_combined_training_metrics,
            plot_rl_training_metrics,
        )

        rows_log: list[dict] = []
        with open(log_path, encoding="utf-8") as logf:
            for line in logf:
                line = line.strip()
                if line:
                    rows_log.append(json.loads(line))
        plot_ok = False
        if args.reward == "adaptive":
            plot_ok = plot_rl_adaptive_combined_metrics(rows_log, str(plot_path))
        elif args.reward == "combined":
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
