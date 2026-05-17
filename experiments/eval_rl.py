"""Evaluate the RL faithfulness-tuned LoRA adapter.

Same pipeline, metrics, and CLI as ``eval_pipeline.py``, but the default
``--sft_dir`` points at the RL output from ``train_rl.py`` under
``/mnt/storage/ae21b026/rl_faithfulness_lora``. Pass ``--sft_dir`` explicitly
to compare another checkpoint.

Example::

    python eval_rl.py --output data/cot/eval_rl.json
    python eval_rl.py --sft_dir /mnt/storage/ae21b026/rl_faithfulness_lora/checkpoint-100 \\
        --output data/cot/eval_rl_ckpt.json
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
