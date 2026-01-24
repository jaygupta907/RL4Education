"""
Main PPO Trainer for fine-tuning question generation model with hypergraph traces.
"""
import os
import logging
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor

from config import TrainingConfig
from model_initialization import initialize_policy_model, initialize_tokenizer, initialize_ppo_trainer
from graph_utils import preload_hypergraph_data
from hypergraph_generator import collect_batch_parallel
from reward_computer import compute_rewards_batched
from logging_utils import log_episode_results_sync
from utils import training_step_context, clean_decoded_text, extract_question

# Try to import rewardanything
try:
    import rewardanything
    REWARDANYTHING_AVAILABLE = True
except ImportError:
    REWARDANYTHING_AVAILABLE = False
    print("Warning: rewardanything library not available. Install with: pip install rewardanything")

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

from torch.cuda.amp import autocast, GradScaler

logger = logging.getLogger(__name__)


class QuestionGeneratorPPOTrainer:
    """Optimized PPO Trainer for fine-tuning question generation model with hypergraph traces."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize the optimized PPO trainer."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize thread pool for async logging
        self.log_executor = ThreadPoolExecutor(max_workers=1)
        
        # Initialize RewardAnything reward model
        if not REWARDANYTHING_AVAILABLE:
            raise RuntimeError("rewardanything library not available. Install with: pip install rewardanything")
        
        logger.info("Initializing RewardAnything reward model...")
        try:
            if self.config.use_vllm_reward:
                # Use vLLM deployment (recommended for production & RL training)
                logger.info(f"Connecting to RewardAnything server at {self.config.reward_server_url}")
                self.reward_model = rewardanything.Client(self.config.reward_server_url)
                logger.info("RewardAnything client connected successfully.")
            else:
                # Use local inference (for quick testing)
                logger.info("Loading RewardAnything model locally...")
                self.reward_model = rewardanything.from_pretrained(
                    "zhuohaoyu/RewardAnything-8B-v1",
                    device=str(self.device),
                    torch_dtype="auto"
                )
                logger.info("RewardAnything reward model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize reward model: {e}")
            if self.config.use_vllm_reward:
                logger.error("Make sure the RewardAnything server is running. Start it with: ./start_reward_server.sh")
            raise RuntimeError("Failed to initialize reward model. Cannot proceed with training.")
        
        # OPTIMIZATION: Pre-load and cache hypergraph data
        self.hypergraph_data, self._all_nodes_cache = preload_hypergraph_data(
            self.config.hypergraph_file
        )
        
        # Initialize policy model
        self.model, self.ref_model = initialize_policy_model(self.config, self.device)
        
        # Initialize tokenizer
        self.tokenizer = initialize_tokenizer(self.config)
        
        # Initialize PPO trainer
        self.ppo_trainer = initialize_ppo_trainer(
            self.config,
            self.model,
            self.ref_model,
            self.tokenizer,
            WANDB_AVAILABLE
        )
        
        # OPTIMIZATION: Initialize mixed precision scaler
        if self.config.use_mixed_precision and torch.cuda.is_available():
            self.scaler = GradScaler()
            logger.info("Mixed precision training enabled")
        else:
            self.scaler = None
        
        # Statistics
        self.episode_rewards = []
        self.episode_scores = []
        
        # Create logs directory (in current directory, not with checkpoints)
        base_logs_dir = self.config.logs_dir
        os.makedirs(base_logs_dir, exist_ok=True)
        
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logs_dir = os.path.join(base_logs_dir, f"run_{run_timestamp}")
        os.makedirs(self.logs_dir, exist_ok=True)
        logger.info(f"Run-specific logs directory created: {self.logs_dir}")
        
        # Initialize wandb if available
        if WANDB_AVAILABLE:
            wandb.init(
                project="question-generator-ppo-hypergraph",
                name=f"ppo-training-hypergraph-{config.policy_model_name.split('/')[-1]}",
                config={
                    "policy_model": config.policy_model_name,
                    "judge_model": config.judge_model_name,
                    "max_depth": config.max_depth,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "ppo_epochs": config.ppo_epochs,
                    "max_new_tokens": config.max_new_tokens,
                    "temperature": config.temperature,
                    "optimizations": "batch_size=4, max_tokens=300, batched_rewards, parallel_generation, mixed_precision"
                }
            )
            logger.info("Wandb initialized for logging.")
    
    def _log_episode_results_async(self, episode, responses, rewards, judge_scores, 
                                   judge_rewards, judge_explanations, batch):
        """Async wrapper for logging (non-blocking)."""
        # Use thread pool to avoid blocking training
        self.log_executor.submit(
            log_episode_results_sync,
            episode, responses, rewards, judge_scores,
            judge_rewards, judge_explanations, batch,
            self.logs_dir, self.config
        )
    
    def train(self):
        """Run optimized PPO training loop."""
        logger.info("Starting OPTIMIZED PPO training with hypergraph traces...")
        logger.info(f"Configuration: {self.config}")
        logger.info(f"Optimizations enabled: batch_size={self.config.batch_size}, max_tokens={self.config.max_new_tokens}, parallel_workers={self.config.num_workers}")
        
        for episode in range(self.config.num_episodes):
            detailed_logging = (episode % self.config.log_detailed_every == 0)
            
            if detailed_logging:
                logger.info(f"\n{'='*60}")
                logger.info(f"Episode {episode + 1}/{self.config.num_episodes}")
                logger.info(f"{'='*60}")
            else:
                logger.info(f"Episode {episode + 1}/{self.config.num_episodes}")
            
            # OPTIMIZATION: Use context manager for cleanup
            with training_step_context():
                # OPTIMIZATION: Parallel batch collection
                if detailed_logging:
                    logger.info("Collecting batch (parallel)...")
                batch = collect_batch_parallel(
                    self.config.batch_size,
                    self.config.num_workers,
                    self.config.hypergraph_file,
                    self.config.max_depth,
                    self.config.max_traces,
                    self.config.min_trace_length,
                    self._all_nodes_cache,
                    self.tokenizer
                )
                
                if len(batch) < self.config.mini_batch_size:
                    logger.warning(f"Batch size ({len(batch)}) too small. Skipping episode...")
                    continue
                
                # Extract queries
                queries = [ex["query"] for ex in batch]
                
                # OPTIMIZATION: Batch tokenize all queries at once
                tokenized = self.tokenizer(
                    queries,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    padding_side="left",
                )
                
                query_tensors = [ids.to(self.device) for ids in tokenized.input_ids]
                
                # Generate responses
                if detailed_logging:
                    logger.info("Generating questions...")
                
                # Set model to eval mode for generation (no gradients needed)
                self.model.eval()
                
                generation_kwargs = {
                    "max_new_tokens": self.config.max_new_tokens,
                    "min_new_tokens": 10,
                    "temperature": self.config.temperature,
                    "do_sample": True,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                    "repetition_penalty": self.config.repetition_penalty,
                    "no_repeat_ngram_size": self.config.no_repeat_ngram_size,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "return_prompt": False,
                }
                
                # OPTIMIZATION: Use mixed precision if enabled
                # Suppress gradient checkpointing warning during generation (expected behavior)
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*None of the inputs have requires_grad=True.*")
                    if self.config.use_mixed_precision and torch.cuda.is_available():
                        with autocast():
                            response_tensors = self.ppo_trainer.generate(query_tensors, **generation_kwargs)
                    else:
                        response_tensors = self.ppo_trainer.generate(query_tensors, **generation_kwargs)
                
                # Set model back to train mode for training step
                self.model.train()
                
                # Decode responses and track valid indices
                responses = []
                valid_indices = []
                for i, response_ids in enumerate(response_tensors):
                    response_ids = response_ids[response_ids != self.tokenizer.pad_token_id]
                    decoded_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    
                    # Clean up decoded text
                    decoded_text = clean_decoded_text(decoded_text)
                    
                    if not decoded_text or len(decoded_text.strip()) < 10:
                        continue
                    
                    # Extract question
                    decoded_text = extract_question(decoded_text)
                    responses.append(decoded_text)
                    valid_indices.append(i)
                
                if detailed_logging:
                    logger.info(f"\nGenerated {len(responses)} questions")
                    for i, resp in enumerate(responses):  # Show first 3
                        logger.info(f"  Q{i+1}: {resp}...")
                
                if not responses:
                    logger.warning("No valid responses generated. Skipping episode...")
                    continue
                
                # Trim batch, query_tensors, and response_tensors to match valid responses
                if len(responses) != len(batch):
                    batch = [batch[i] for i in valid_indices]
                    query_tensors = [query_tensors[i] for i in valid_indices]
                    response_tensors = [response_tensors[i] for i in valid_indices]
                
                # OPTIMIZATION: Batch compute rewards
                if detailed_logging:
                    logger.info("Computing rewards (batched)...")
                
                rewards, judge_scores, judge_rewards, judge_explanations = \
                    compute_rewards_batched(
                        responses, batch, self.reward_model
                    )
                
                # Log statistics
                avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
                avg_score = sum(judge_scores) / len(judge_scores) if judge_scores else 0.0
                
                self.episode_rewards.append(avg_reward)
                self.episode_scores.append(avg_score)
                
                if detailed_logging:
                    logger.info(f"Avg judge score: {avg_score:.2f}/10.0, Avg reward: {avg_reward:.4f}")
                else:
                    logger.info(f"Avg score: {avg_score:.2f}, Avg reward: {avg_reward:.4f}")
                
                # OPTIMIZATION: Async logging (non-blocking)
                if detailed_logging:
                    self._log_episode_results_async(
                        episode, responses, rewards, judge_scores,
                        judge_rewards, judge_explanations, batch
                    )
                
                # Log to wandb
                if WANDB_AVAILABLE:
                    wandb.log({
                        "episode/avg_reward": avg_reward,
                        "episode/avg_judge_score": avg_score,
                    }, step=episode)
                
                # Convert rewards to tensors
                reward_tensors = [torch.tensor(reward, dtype=torch.float32, device='cpu') for reward in rewards]
                
                # Check if batch sizes match - skip step if they don't
                if len(query_tensors) != len(response_tensors) or len(query_tensors) != len(reward_tensors):
                    logger.warning(
                        f"Batch size mismatch: queries={len(query_tensors)}, "
                        f"responses={len(response_tensors)}, rewards={len(reward_tensors)}. "
                        f"Skipping PPO step for this episode."
                    )
                    continue
                
                # Train with PPO
                if detailed_logging:
                    logger.info("Training with PPO...")
                
                try:
                    if self.config.use_mixed_precision and torch.cuda.is_available():
                        with autocast():
                            stats = self.ppo_trainer.step(
                                query_tensors,
                                response_tensors,
                                reward_tensors
                            )
                    else:
                        stats = self.ppo_trainer.step(
                            query_tensors,
                            response_tensors,
                            reward_tensors
                        )
                    
                    # Extract KL divergence and entropy from stats
                    kl_div_raw = stats.get('ppo/kl', stats.get('ppo/policy/approxkl', 0.0))
                    entropy_raw = stats.get('ppo/entropy', stats.get('ppo/policy/entropy', 0.0))
                    
                    # Convert to float (handle numpy arrays and tensors)
                    if isinstance(kl_div_raw, (np.ndarray, np.generic)):
                        kl_div = float(kl_div_raw.item() if hasattr(kl_div_raw, 'item') else kl_div_raw)
                    elif isinstance(kl_div_raw, torch.Tensor):
                        kl_div = float(kl_div_raw.item())
                    else:
                        kl_div = float(kl_div_raw)
                    
                    if isinstance(entropy_raw, (np.ndarray, np.generic)):
                        entropy = float(entropy_raw.item() if hasattr(entropy_raw, 'item') else entropy_raw)
                    elif isinstance(entropy_raw, torch.Tensor):
                        entropy = float(entropy_raw.item())
                    else:
                        entropy = float(entropy_raw)
                    
                    # Log PPO metrics to wandb
                    if WANDB_AVAILABLE:
                        wandb_metrics = {
                            "ppo/kl_divergence": kl_div,
                            "ppo/entropy": entropy,
                        }
                        # Add other useful PPO metrics if available
                        if 'ppo/policy/loss' in stats:
                            loss_val = stats['ppo/policy/loss']
                            if isinstance(loss_val, (np.ndarray, np.generic)):
                                loss_val = float(loss_val.item() if hasattr(loss_val, 'item') else loss_val)
                            elif isinstance(loss_val, torch.Tensor):
                                loss_val = float(loss_val.item())
                            wandb_metrics["ppo/policy/loss"] = loss_val
                        if 'ppo/val/loss' in stats:
                            loss_val = stats['ppo/val/loss']
                            if isinstance(loss_val, (np.ndarray, np.generic)):
                                loss_val = float(loss_val.item() if hasattr(loss_val, 'item') else loss_val)
                            elif isinstance(loss_val, torch.Tensor):
                                loss_val = float(loss_val.item())
                            wandb_metrics["ppo/val/loss"] = loss_val
                        if 'ppo/mean_non_score_reward' in stats:
                            reward_val = stats['ppo/mean_non_score_reward']
                            if isinstance(reward_val, (np.ndarray, np.generic)):
                                reward_val = float(reward_val.item() if hasattr(reward_val, 'item') else reward_val)
                            elif isinstance(reward_val, torch.Tensor):
                                reward_val = float(reward_val.item())
                            wandb_metrics["ppo/mean_non_score_reward"] = reward_val
                        
                        wandb.log(wandb_metrics, step=episode)
                    
                    if detailed_logging:
                        logger.info(f"PPO stats - KL divergence: {kl_div:.4f}, Entropy: {entropy:.4f}")
                        
                except torch.cuda.OutOfMemoryError as e:
                    logger.error(f"Out of memory at episode {episode + 1}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
            
            # Save checkpoint
            if (episode + 1) % self.config.save_steps == 0:
                checkpoint_path = f"{self.config.output_dir}/checkpoint-{episode + 1}"
                os.makedirs(checkpoint_path, exist_ok=True)
                logger.info(f"Saving checkpoint to {checkpoint_path}...")
                self.model.save_pretrained(checkpoint_path)
                self.tokenizer.save_pretrained(checkpoint_path)
        
        # Save final model
        os.makedirs(self.config.output_dir, exist_ok=True)
        logger.info(f"Saving final model to {self.config.output_dir}...")
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Shutdown thread pool
        self.log_executor.shutdown(wait=True)
        
        logger.info("Training completed!")
        logger.info(f"Average reward: {sum(self.episode_rewards) / len(self.episode_rewards):.4f}")
        logger.info(f"Average judge score: {sum(self.episode_scores) / len(self.episode_scores):.2f}/10.0")
        
        if WANDB_AVAILABLE:
            wandb.finish()

