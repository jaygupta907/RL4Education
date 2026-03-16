"""
Configuration module for PPO training.
"""
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning."""
    # Model configuration
    policy_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    instruction_tuned_model_path: Optional[str] = None  # Path to instruction-tuned model (if available)
    judge_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    graph_file: str = "variable_concept_graph.json"
    
    # Training configuration - OPTIMIZED
    max_length: int = 8
    min_tree_walk_length: int = 2  # UPDATED: Reduced from 4 to 1
    num_episodes: int = 1000
    batch_size: int = 4  # UPDATED: Reduced from 8 to 4
    mini_batch_size: int = 2  # UPDATED: Reduced from 4 to 2
    gradient_accumulation_steps: int = 2  # OPTIMIZED: Adjusted for larger batch
    
    # PPO hyperparameters
    learning_rate: float = 1.41e-6
    ppo_epochs: int = 2
    cliprange: float = 0.05
    cliprange_value: float = 0.05
    gamma: float = 1.0
    lam: float = 0.95
    init_kl_coef: float = 2.0
    target_kl: float = 0.001
    
    # Generation configuration - OPTIMIZED
    max_new_tokens: int = 512  
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 2
    
    # Output configuration - OPTIMIZED
    output_dir: str = "/mnt/storage/ae21b026/models/ppo_checkpoints"
    logs_dir: str = "./checkpoints/logs"  # Logs stay in current directory
    save_steps: int = 50  # OPTIMIZED: Save less frequently
    logging_steps: int = 10  # OPTIMIZED: Log less frequently
    
    # Performance configuration
    use_mixed_precision: bool = True  # NEW: Enable mixed precision training
    use_quantization: bool = False  # NEW: Enable 8-bit quantization (set to True for more speed)
    num_workers: int = 4  # NEW: Number of parallel workers for tree walk generation
    log_detailed_every: int = 1  # NEW: Log detailed results every N episodes (1 = every episode)
    
    # Device configuration
    max_memory: Optional[Dict] = None
    
    # Reward model configuration
    use_vllm_reward: bool = True  # Use vLLM deployment instead of local inference
    reward_server_url: str = "http://localhost:8001"  # RewardAnything server URL
    

