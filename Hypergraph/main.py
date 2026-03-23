"""
Main entry point for PPO training with hypergraph traces.

This script fine-tunes the question generation LLM using Proximal Policy Optimization (PPO)
with rewards from the judge LLM. The judge evaluates whether generated questions correctly
ask for the solution trace based on the hypergraph traces.

The script first performs instruction fine-tuning on the dataset, then proceeds with RL training.
Requirements:
    - transformers library: pip install transformers torch
    - trl library: pip install trl
    - accelerate: pip install accelerate
    - datasets: pip install datasets
"""
# Reduce CUDA fragmentation (set before any torch/cuda import)
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import logging
from config import get_default_rl_training_config
from trainer import QuestionGeneratorPPOTrainer

import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
)


# Try to import required libraries
try:
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    TRL_AVAILABLE = True
except ImportError as e:
    TRL_AVAILABLE = False
    print(f"Warning: Required libraries not available: {e}")
    print("Install with: pip install transformers torch trl accelerate")

# Configure logging (no file handler, only console)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Only console, no file logging
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run optimized training."""
    if not TRL_AVAILABLE:
        logger.error("Required libraries not available. Please install: pip install transformers torch trl accelerate")
        return

    config = get_default_rl_training_config()

    if config.instruction_tuned_model_path is None:
        logger.warning(f"Instruction-tuned model not found a {config.instruction_tuned_model_path}")
        logger.warning("RL training will use the pretrained model instead.")
        logger.warning("To use instruction-tuned model, run instruction_tune.py first.")
    else:
        logger.info(f"Found instruction-tuned model at: {config.instruction_tuned_model_path}")
        logger.info("RL training will use the instruction-tuned model.")

    logger.info("=" * 80)
    logger.info("Starting RL (PPO) training phase with hypergraph traces...")
    logger.info("=" * 80)

    trainer = QuestionGeneratorPPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
