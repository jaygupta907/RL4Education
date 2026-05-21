"""Evaluate the RL faithfulness-tuned LoRA adapter.

Same pipeline, metrics, and CLI as ``eval_pipeline.py``, but the default
``--sft_dir`` points at the RL output from ``train_rl.py`` under
``/mnt/storage/ae21b026/rl_faithfulness_lora``. Pass ``--sft_dir`` explicitly
to compare another checkpoint.

Judges use ``--llm-provider`` (``claude`` or ``openai``); keys load from
``experiments/.env`` (``OPENAI_API_KEY``, ``OPENAI_MODEL``) unless overridden.

Examples::

    python eval_rl.py --output data/rl/eval_rl.json

    python eval_rl.py --llm-provider openai --llm-model gpt-5.5 \\
        --output data/rl/eval_rl_openai.json

    python eval_rl.py --sft_dir /mnt/storage/ae21b026/rl_adaptive_openai/checkpoint-100 \\
        --llm-provider openai --output data/rl/eval_ckpt.json
"""
import sys

DEFAULT_RL_ADAPTER = "/mnt/storage/ae21b026/rl_faithfulness_lora"


def main():
    if "--sft_dir" not in sys.argv:
        sys.argv[1:1] = ["--sft_dir", DEFAULT_RL_ADAPTER]
    from eval_pipeline import main as eval_main

    eval_main()


if __name__ == "__main__":
    main()
