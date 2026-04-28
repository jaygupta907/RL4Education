"""
Configuration module for PPO training with hypergraph traces.
"""
import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning with hypergraph traces."""
    # Model configuration
    policy_model_name: str = "meta-llama/Meta-Llama-3-8B"  # Pretrained model (not Instruct)
    instruction_tuned_model_path: Optional[str] = None  # Path to instruction-tuned model (if available)
    judge_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"  # Judge can still use Instruct
    hypergraph_file: str = "formula_hypergraph.json"

    # Training configuration - OPTIMIZED
    max_depth: int = 10  # Maximum depth for hypergraph traversal
    max_traces: int = 100  # Maximum number of traces to consider per target
    min_trace_length: int = 2  # Minimum number of formulas in trace
    num_episodes: int = 300
    batch_size: int = 4  # UPDATED: Reduced from 8 to 4
    mini_batch_size: int = 2  # UPDATED: Reduced from 4 to 2
    gradient_accumulation_steps: int = 2  # OPTIMIZED: Adjusted for larger batch

    # PPO hyperparameters
    learning_rate: float = 5e-7
    ppo_epochs: int = 2
    cliprange: float = 0.1
    cliprange_value: float = 0.1
    gamma: float = 1.0
    lam: float = 0.95
    init_kl_coef: float = 3.0
    target_kl: float = 0.1

    # Generation configuration - OPTIMIZED
    max_new_tokens: int = 300
    temperature: float = 0.7
    top_p: float = 0.9

    # Output configuration - OPTIMIZED
    output_dir: str = "/mnt/storage/ae21b026/models/ppo_checkpoints_hypergraph_rl_no_quant"
    logs_dir: str = "./checkpoints/logs"  # Logs stay in current directory
    wandb_project: str = "verifiable-question-generation"
    experiment_name: str = "ppo-hypergraph-question-generation"
    save_steps: int = 20  # Save evaluator-compatible checkpoints every 10 PPO steps
    logging_steps: int = 10  # OPTIMIZED: Log less frequently

    # Performance configuration
    use_mixed_precision: bool = False  # NEW: Enable mixed precision training
    use_quantization: bool = False  # NEW: Enable 8-bit quantization (set to True for more speed)
    num_workers: int = 4  # NEW: Number of parallel workers for trace generation
    log_detailed_every: int = 1  # NEW: Log detailed results every N episodes (1 = every episode)

    # Device configuration
    max_memory: Optional[Dict] = None
    max_memory_utilization: float = 0.85
    cpu_max_memory: str = "64GiB"

    # Reward model configuration
    use_vllm_reward: bool = True  # Use vLLM deployment instead of local inference
    reward_server_url: str = "http://localhost:8001"  # RewardAnything server URL

    # Composite reward configuration
    # PPO reward = (difficulty_weight * difficulty_reward) + (faithfulness_weight * faithfulness_reward)
    # difficulty_reward is in [-1, 1], faithfulness score is mapped from [1, 10] -> [-1, 1].
    difficulty_weight: float = 0.0
    faithfulness_weight: float = 1.0


def get_default_config() -> TrainingConfig:
    """Return the RL training configuration used by the main training entrypoint."""
    instruction_tuned_model_path = "/mnt/storage/ae21b026/models/instruction_tuned_model"
    if not (
        os.path.exists(instruction_tuned_model_path)
        and os.path.exists(os.path.join(instruction_tuned_model_path, "config.json"))
    ):
        instruction_tuned_model_path = None

    return TrainingConfig(
        instruction_tuned_model_path=instruction_tuned_model_path,
    )


def get_default_rl_training_config() -> TrainingConfig:
    """Backward-compatible alias used by the RL training entrypoint."""
    return get_default_config()
