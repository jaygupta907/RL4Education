"""
Configuration module for PPO training with hypergraph traces.
"""
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
    num_episodes: int = 1000
    batch_size: int = 4  # UPDATED: Reduced from 8 to 4
    mini_batch_size: int = 2  # UPDATED: Reduced from 4 to 2
    gradient_accumulation_steps: int = 2  # OPTIMIZED: Adjusted for larger batch
    
    # PPO hyperparameters
    learning_rate: float = 1.41e-6
    ppo_epochs: int = 2
    cliprange: float = 0.05
    cliprange_value: float = 0.2
    gamma: float = 1.0
    lam: float = 0.95
    init_kl_coef: float = 1.0
    target_kl: float = 2.0
    
    # Generation configuration - OPTIMIZED
    max_new_tokens: int = 300  
    temperature: float = 0.5
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 2
    
    # Output configuration - OPTIMIZED
    output_dir: str = "/mnt/storage/ae21b026/models/ppo_checkpoints_hypergraph"
    logs_dir: str = "./checkpoints/logs"  # Logs stay in current directory
    save_steps: int = 50  # OPTIMIZED: Save less frequently
    logging_steps: int = 10  # OPTIMIZED: Log less frequently
    
    # Performance configuration
    use_mixed_precision: bool = False  # NEW: Enable mixed precision training
    use_quantization: bool = False  # NEW: Enable 8-bit quantization (set to True for more speed)
    num_workers: int = 4  # NEW: Number of parallel workers for trace generation
    log_detailed_every: int = 1  # NEW: Log detailed results every N episodes (1 = every episode)
    
    # Device configuration
    max_memory: Optional[Dict] = None
    
    # Reward model configuration
    use_vllm_reward: bool = True  # Use vLLM deployment instead of local inference
    reward_server_url: str = "http://localhost:8001"  # RewardAnything server URL

