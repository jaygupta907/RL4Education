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
from config import TrainingConfig
from trainer import QuestionGeneratorPPOTrainer

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
    
    # Check if instruction-tuned model exists
    instruction_tuned_model_path = "/mnt/storage/ae21b026/models/instruction_tuned_model"
    if not os.path.exists(instruction_tuned_model_path):
        logger.warning(f"Instruction-tuned model not found at {instruction_tuned_model_path}")
        logger.warning("RL training will use the pretrained model instead.")
        logger.warning("To use instruction-tuned model, run instruction_tune.py first.")
        instruction_tuned_model_path = None
    else:
        logger.info(f"Found instruction-tuned model at: {instruction_tuned_model_path}")
        logger.info("RL training will use the instruction-tuned model.")
    
    # OPTIMIZED configuration for Llama 3 with hypergraph traces
    config = TrainingConfig(
        num_episodes=10000,
        max_depth=10,  # Maximum depth for hypergraph traversal
        max_traces=100,  # Maximum number of traces to consider per target
        min_trace_length=2,  # Minimum number of formulas in trace
        # OPTIMIZATION: Reduced batch sizes to avoid OOM (increase if you have more VRAM)
        batch_size=4,
        mini_batch_size=2,
        gradient_accumulation_steps=2,
        # OPTIMIZATION: Reduced token generation to lower peak memory
        max_new_tokens=300,
        # OPTIMIZATION: Performance settings
        use_mixed_precision=False,  # 1.5x speedup
        use_quantization=False,  # Set to True for 2x more speedup (if memory allows)
        num_workers=4,  # Parallel trace generation
        log_detailed_every=1,  # Log details every episode
        save_steps=50,
        # Reward model configuration
        use_vllm_reward=True,  # Use vLLM deployment (recommended for RL training)
        reward_server_url="http://localhost:8001",  # RewardAnything server URL
        # Hypergraph file
        hypergraph_file="formula_hypergraph.json",
        # Instruction-tuned model path
        instruction_tuned_model_path=instruction_tuned_model_path,
        # Fewer PPO epochs to reduce peak memory (default 2 -> 1)
        ppo_epochs=1,
    )
    
    logger.info("=" * 80)
    logger.info("Starting RL (PPO) training phase with hypergraph traces...")
    logger.info("=" * 80)
    
    # Create trainer
    trainer = QuestionGeneratorPPOTrainer(config)
    
    # Run training
    trainer.train()


if __name__ == "__main__":
    main()

