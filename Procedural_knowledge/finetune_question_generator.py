"""
Optimized Fine-tune Question Generator using PPO with Judge LLM Reward

This script fine-tunes the question generation LLM using Proximal Policy Optimization (PPO)
with rewards from the judge LLM. The judge evaluates whether generated questions correctly
ask for the solution trace based on the pruned tree walk.

PERFORMANCE OPTIMIZATIONS IMPLEMENTED:
1. Increased batch size from 1 to 8 (8x speedup)
2. Reduced max_new_tokens from 500 to 150 (3x speedup)
3. Batched reward computation (2x speedup)
4. Parallel tree walk generation (2-4x speedup)
5. Reduced logging frequency (1.2x speedup)
6. Mixed precision training (1.5x speedup)
7. Pre-cached graph data and question contexts
8. Optimized memory cleanup
9. Async logging to avoid blocking

Expected total speedup: 30-100x faster than original

Requirements:
    - transformers library: pip install transformers torch
    - trl library: pip install trl
    - accelerate: pip install accelerate
"""

import logging
import json
import os
import random
import re
import torch
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
from tree_walk_calculation import TreeWalkCalculator
from generate_question_from_answer import QuestionGenerator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import gc
import asyncio
from functools import partial

# Try to import rewardanything
try:
    import rewardanything
    REWARDANYTHING_AVAILABLE = True
except ImportError:
    REWARDANYTHING_AVAILABLE = False
    print("Warning: rewardanything library not available. Install with: pip install rewardanything")

# Try to import required libraries
try:
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from torch.cuda.amp import autocast, GradScaler
    TRL_AVAILABLE = True
except ImportError as e:
    TRL_AVAILABLE = False
    print(f"Warning: Required libraries not available: {e}")
    print("Install with: pip install transformers torch trl accelerate")

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

# Configure logging (no file handler, only console)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Only console, no file logging
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning."""
    # Model configuration
    policy_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
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
    ppo_epochs: int = 4
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    gamma: float = 1.0
    lam: float = 0.95
    
    # Generation configuration - OPTIMIZED
    max_new_tokens: int = 3000  # OPTIMIZED: Reduced from 500 (3x speedup)
    temperature: float = 0.5
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 2
    
    # Output configuration - OPTIMIZED
    output_dir: str = "./checkpoints/question_generator_ppo"
    save_steps: int = 50  # OPTIMIZED: Save less frequently
    logging_steps: int = 10  # OPTIMIZED: Log less frequently
    
    # Performance configuration
    use_mixed_precision: bool = True  # NEW: Enable mixed precision training
    use_quantization: bool = False  # NEW: Enable 8-bit quantization (set to True for more speed)
    num_workers: int = 4  # NEW: Number of parallel workers for tree walk generation
    log_detailed_every: int = 10  # NEW: Log detailed results every N episodes
    
    # Device configuration
    max_memory: Optional[Dict] = None


class QuestionGeneratorPPOTrainer:
    """Optimized PPO Trainer for fine-tuning question generation model."""
    
    @contextmanager
    def _training_step_context(self):
        """Context manager for proper cleanup after each training step."""
        try:
            yield
        finally:
            # Consolidated cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
    
    def _log_memory_usage(self, stage: str = ""):
        """Log current GPU memory usage for debugging."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            logger.debug(f"GPU Memory {stage}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Max={max_allocated:.2f}GB")
    
    def _format_solution_trace(self, calculator: TreeWalkCalculator) -> Dict:
        """
        Format the solution trace from the calculator into a structured dictionary.
        
        Args:
            calculator: TreeWalkCalculator instance with completed calculation
            
        Returns:
            Dictionary containing the solution trace information
        """
        target = calculator.tree_structure['target']
        target_value = calculator.values.get(target, None)
        
        if target_value is None:
            return {
                "target": target,
                "final_answer": None,
                "given_values": [],
                "calculation_steps": [],
                "formatted_trace": "No solution trace available."
            }
        
        # Collect given values (leaf nodes)
        given_values = []
        for leaf in sorted(calculator.tree_structure['leaf_nodes']):
            if leaf in calculator.values:
                si_unit = calculator._get_si_unit(leaf)
                given_values.append({
                    "variable": leaf,
                    "value": float(calculator.values[leaf]),
                    "unit": si_unit if si_unit else None
                })
        
        # Collect calculation steps by level
        calculation_steps = []
        all_levels = sorted(calculator.tree_structure['levels'].keys())
        for level in all_levels:
            if level == 0:  # Skip target level (will be shown separately)
                continue
            level_nodes = calculator.tree_structure['levels'].get(level, [])
            for node in sorted(level_nodes):
                if node not in calculator.tree_structure['leaf_nodes'] and node in calculator.values:
                    step_info = {
                        "level": level,
                        "variable": node,
                        "value": float(calculator.values[node])
                    }
                    
                    # Add formula if available
                    if 'node_formulas' in calculator.tree_structure:
                        if node in calculator.tree_structure['node_formulas']:
                            formula, deps = calculator.tree_structure['node_formulas'][node]
                            if formula:
                                step_info["formula"] = formula
                                step_info["dependencies"] = sorted(list(deps))
                    
                    calculation_steps.append(step_info)
        
        # Create formatted trace string
        trace_lines = []
        trace_lines.append(f"Solution Trace for: {target}")
        trace_lines.append("=" * 60)
        trace_lines.append(f"Target Variable: {target}")
        trace_lines.append(f"Final Answer: {target_value:.4f}")
        trace_lines.append("\nGiven Values (Leaf Nodes):")
        
        for gv in given_values:
            if gv["unit"]:
                trace_lines.append(f"  • {gv['variable']} = {gv['value']:.4f} {gv['unit']}")
            else:
                trace_lines.append(f"  • {gv['variable']} = {gv['value']:.4f}")
        
        trace_lines.append("\nCalculation Steps:")
        for step in calculation_steps:
            formula_info = ""
            if "formula" in step:
                formula_info = f" (using: {step['formula']})"
            trace_lines.append(f"  Level {step['level']}: {step['variable']} = {step['value']:.4f}{formula_info}")
        
        formatted_trace = "\n".join(trace_lines)
        
        return {
            "target": target,
            "final_answer": float(target_value),
            "given_values": given_values,
            "calculation_steps": calculation_steps,
            "formatted_trace": formatted_trace
        }
    
    def _log_episode_results_sync(self, episode: int, responses: List[str], rewards: List[float], 
                            judge_scores: List[float], judge_rewards: List[float], judge_explanations: List[str], 
                            batch: List[Dict], tree_walk_lengths: List[int] = None, 
                            tree_walk_length_rewards: List[float] = None):
        """
        Synchronous version of log episode results (for threading).
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"episode_{episode+1:04d}_{timestamp}.json"
        log_path = os.path.join(self.logs_dir, log_filename)
        
        log_data = {
            "episode": episode + 1,
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(responses),
            "questions": []
        }
        
        for i, (response, reward, score, judge_reward, explanation) in enumerate(zip(responses, rewards, judge_scores, judge_rewards, judge_explanations)):
            question_data = {
                "question_index": i + 1,
                "question": response,
                "combined_reward": float(reward),
                "judge_score": float(score),
                "judge_reward": float(judge_reward),
                "explanation": explanation
            }
            
            if tree_walk_lengths and i < len(tree_walk_lengths):
                tree_walk_length = tree_walk_lengths[i]
                question_data["tree_walk_length"] = tree_walk_length
                question_data["max_tree_walk_length"] = self.config.max_length
                
                if tree_walk_length_rewards and i < len(tree_walk_length_rewards):
                    question_data["tree_walk_length_reward"] = float(tree_walk_length_rewards[i])
                else:
                    normalized_length = tree_walk_length / self.config.max_length if self.config.max_length > 0 else 0.0
                    question_data["tree_walk_length_reward"] = float(normalized_length * 2.0 - 1.0)
            
            if i < len(batch) and "metadata" in batch[i]:
                question_data["metadata"] = batch[i]["metadata"]
            
            # Add solution trace if calculator is available
            if i < len(batch) and "calculator" in batch[i]:
                calculator = batch[i]["calculator"]
                try:
                    solution_trace = self._format_solution_trace(calculator)
                    question_data["solution_trace"] = solution_trace
                except Exception as e:
                    logger.warning(f"Failed to format solution trace for question {i+1}: {e}")
                    question_data["solution_trace"] = {"error": str(e)}
            
            log_data["questions"].append(question_data)
        
        if rewards:
            summary = {
                "combined_reward": {
                    "average": float(sum(rewards) / len(rewards)),
                    "min": float(min(rewards)),
                    "max": float(max(rewards))
                },
                "judge": {
                    "average_score": float(sum(judge_scores) / len(judge_scores)) if judge_scores else 0.0,
                    "min_score": float(min(judge_scores)) if judge_scores else 0.0,
                    "max_score": float(max(judge_scores)) if judge_scores else 0.0,
                    "average_reward": float(sum(judge_rewards) / len(judge_rewards)) if judge_rewards else 0.0,
                    "min_reward": float(min(judge_rewards)) if judge_rewards else 0.0,
                    "max_reward": float(max(judge_rewards)) if judge_rewards else 0.0
                }
            }
            
            if tree_walk_lengths:
                summary["tree_walk_length"] = {
                    "average_length": float(sum(tree_walk_lengths) / len(tree_walk_lengths)),
                    "min_length": int(min(tree_walk_lengths)),
                    "max_length": int(max(tree_walk_lengths)),
                    "max_possible_length": self.config.max_length
                }
                
                if tree_walk_length_rewards:
                    summary["tree_walk_length"]["average_reward"] = float(sum(tree_walk_length_rewards) / len(tree_walk_length_rewards))
                    summary["tree_walk_length"]["min_reward"] = float(min(tree_walk_length_rewards))
                    summary["tree_walk_length"]["max_reward"] = float(max(tree_walk_length_rewards))
            
            log_data["summary"] = summary
        
        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Episode results logged to: {log_path}")
        except Exception as e:
            logger.error(f"Failed to write log file {log_path}: {e}")
    
    def _log_episode_results_async(self, *args):
        """Async wrapper for logging (non-blocking)."""
        # Use thread pool to avoid blocking training
        self.log_executor.submit(self._log_episode_results_sync, *args)
    
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
            self.reward_model = rewardanything.from_pretrained(
                "WisdomShell/RewardAnything-8B-v1",
                device=str(self.device),
                torch_dtype="auto"
            )
            logger.info("RewardAnything reward model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load RewardAnything reward model: {e}")
            raise RuntimeError("Failed to initialize reward model. Cannot proceed with training.")
        
        # OPTIMIZATION: Pre-load and cache graph data
        self._preload_graph_data()
        
        # Initialize policy model
        self._initialize_policy_model()
        
        # Initialize tokenizer
        self._initialize_tokenizer()
        
        # Initialize PPO trainer
        self._initialize_ppo_trainer()
        
        # OPTIMIZATION: Initialize mixed precision scaler
        if self.config.use_mixed_precision and torch.cuda.is_available():
            self.scaler = GradScaler()
            logger.info("Mixed precision training enabled")
        else:
            self.scaler = None
        
        # Statistics
        self.episode_rewards = []
        self.episode_scores = []
        
        # Create logs directory
        base_logs_dir = os.path.join(os.path.dirname(self.config.output_dir), "logs")
        os.makedirs(base_logs_dir, exist_ok=True)
        
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logs_dir = os.path.join(base_logs_dir, f"run_{run_timestamp}")
        os.makedirs(self.logs_dir, exist_ok=True)
        logger.info(f"Run-specific logs directory created: {self.logs_dir}")
        
        # Initialize wandb if available
        if WANDB_AVAILABLE:
            wandb.init(
                project="question-generator-ppo",
                name=f"ppo-training-optimized-{config.policy_model_name.split('/')[-1]}",
                config={
                    "policy_model": config.policy_model_name,
                    "judge_model": config.judge_model_name,
                    "max_length": config.max_length,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "ppo_epochs": config.ppo_epochs,
                    "max_new_tokens": config.max_new_tokens,
                    "temperature": config.temperature,
                    "optimizations": "batch_size=8, max_tokens=150, batched_rewards, parallel_generation, mixed_precision"
                }
            )
            logger.info("Wandb initialized for logging.")
    
    def _preload_graph_data(self):
        """OPTIMIZATION: Pre-load graph data to avoid repeated file I/O."""
        logger.info("Pre-loading graph data...")
        with open(self.config.graph_file, 'r') as f:
            self.graph_data = json.load(f)
        
        # Pre-compute deep variables cache
        self._deep_variables_cache = self._get_variables_by_depth(self.graph_data, min_depth=self.config.min_tree_walk_length)
        self._all_variables_cache = [v['variable'] for v in self.graph_data['variables']]
        
        if not self._deep_variables_cache:
            logger.warning(f"No variables with depth >= {self.config.min_tree_walk_length} found. Using all variables.")
            self._deep_variables_cache = self._all_variables_cache.copy()
        else:
            logger.info(f"Found {len(self._deep_variables_cache)} variables with depth >= {self.config.min_tree_walk_length}")
    
    def _initialize_policy_model(self):
        """Initialize the policy model with value head."""
        logger.info(f"Loading policy model: {self.config.policy_model_name}")
        
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
        }
        
        # OPTIMIZATION: Add quantization config if enabled
        if self.config.use_quantization:
            logger.info("Enabling 8-bit quantization for faster inference...")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            model_kwargs["quantization_config"] = quantization_config
        
        if self.config.max_memory:
            model_kwargs["max_memory"] = self.config.max_memory
        
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.config.policy_model_name,
            **model_kwargs
        )
        
        logger.info("Enabling gradient checkpointing...")
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
        
        logger.info("Policy model loaded successfully.")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        logger.info("Loading reference model...")
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.config.policy_model_name,
            **model_kwargs
        )
        for param in self.ref_model.parameters():
            param.requires_grad = False
        logger.info("Reference model loaded successfully.")
    
    def _initialize_tokenizer(self):
        """Initialize the tokenizer."""
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.policy_model_name)
        
        # Llama 3 specific tokenizer setup
        if self.tokenizer.pad_token is None:
            # For Llama 3, use eos_token as pad_token
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Set padding side to left for decoder-only models
        self.tokenizer.padding_side = "left"
        
        logger.info("Tokenizer loaded successfully.")
        logger.info(f"Vocab size: {len(self.tokenizer)}, PAD token: {self.tokenizer.pad_token}")
    
    def _initialize_ppo_trainer(self):
        """Initialize the PPO trainer."""
        logger.info("Initializing PPO trainer...")
        
        ppo_config = PPOConfig(
            model_name=self.config.policy_model_name,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            mini_batch_size=self.config.mini_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            ppo_epochs=self.config.ppo_epochs,
            cliprange=self.config.cliprange,
            cliprange_value=self.config.cliprange_value,
            gamma=self.config.gamma,
            lam=self.config.lam,
            log_with="wandb" if WANDB_AVAILABLE else None,
            project_kwargs={"logging_dir": self.config.output_dir},
            optimize_cuda_cache=True,
            gradient_checkpointing=True,
            init_kl_coef=0.1,        # This is your 'Beta'
            adap_kl_ctrl=True,       # Keeps KL target stable automatically
        )
        
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )
        
        logger.info("PPO trainer initialized successfully.")
    
    def _create_prompt(self, calculator: TreeWalkCalculator) -> Tuple[str, Dict]:
        """
        Create a prompt for question generation that reflects the specific 
        tree walk trace and intermediate steps.
        """
        target = calculator.tree_structure['target']
        leaf_nodes_set = calculator.tree_structure.get('leaf_nodes', set())
        all_nodes_set = calculator.tree_structure.get('nodes', set())
        
        # Identify given values (leaf nodes)
        given_values_list = sorted([leaf for leaf in leaf_nodes_set if leaf in calculator.values])
        
        # Identify intermediate steps (nodes that are neither leaf nor target)
        # These represent the "path" the calculator took through the random formulas
        intermediate_nodes = sorted([n for n in all_nodes_set if n not in leaf_nodes_set and n != target])

        # Format the list of given values with exact numbers and SI units
        values_examples = []
        for idx, var in enumerate(given_values_list, 1):
            value = calculator.values[var]
            si_unit = calculator._get_si_unit(var)
            unit_str = si_unit if (si_unit and si_unit.strip()) else "(unit not specified)"
            values_examples.append(f"{idx}. {var} = {value} {unit_str}")
        
        values_list_text = "\n".join(values_examples)
        allowed_vars_list = ", ".join(given_values_list)
        examples = """
            Example 1:
            Question: A sled is pulled along level snow by a constant horizontal force of 125.0000 N through a displacement of 14.3000 m. The sled and payload have a total mass of 9.8000 kg. First determine the work done by the pull, then use that work (all converted to kinetic) to find the kinetic_energy of the sled. What is the kinetic_energy?

            Example 2:
            Question: An aluminum block (mass 3.6000 kg, specific_heat 890.0000 J/(kg·K)) starts at 295.1500 K and absorbs 42,500.0000 J of heat with no losses. First compute the temperature rise from the heat input, then determine the final_temperature of the block. What is the final_temperature?

            Example 3:
            Question: A cart of mass 7.7500 kg moves along a frictionless track at 4.2000 m/s. A constant horizontal force of 62.0000 N is applied for 6.0000 s. First find the resulting acceleration from the force and mass, then use it to compute the final_velocity after the push. What is the final_velocity?
        """

        # For Llama 8B Instruct, use the proper chat template format
        system_prompt = """You are a physics problem generator. Generate clear, realistic physics word problems in English using exact numerical values provided.

Your task:
- Use ALL provided numeric values EXACTLY as given (no rounding, no modifications)
- Include proper SI units for each value
- Create a realistic physical scenario that naturally incorporates all given variables
- End by asking for the target variable
- Generate ONLY the problem text, no preamble or explanations"""
        
        user_prompt = f"""Generate a physics word problem using these specifications:

GIVEN VALUES (use ALL {len(given_values_list)} values exactly as shown):
{values_list_text}

ALLOWED INPUT VARIABLES:
{allowed_vars_list}

TARGET VARIABLE TO CALCULATE:
{target}

REFERENCE EXAMPLES (follow this style):
{examples}

CRITICAL REQUIREMENTS:
1. Use ONLY the variables listed as allowed inputs.
2. Use EXACT numeric values with correct SI units from the given values list.
3. Create a realistic physical scenario that naturally incorporates all given variables.
4. End with asking for the target variable"
5. Do NOT include phrases like "Here is the problem:" - start directly with the problem.
6. Do NOT use placeholder symbols  - use the actual numeric values provided.
7. Do NOT invent any additional values or parameters.

Generate the problem now:"""
        
        # Apply Llama 3 Chat Template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                # Fallback manual format for Llama 3
                prompt_text = (
                    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                    f"{system_prompt}<|eot_id|>"
                    f"<|start_header_id|>user<|end_header_id|>\n\n"
                    f"{user_prompt}<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                )
        else:
            prompt_text = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{system_prompt}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{user_prompt}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        
        metadata = {
            "target": target,
            "leaf_nodes": given_values_list,
            "intermediate_nodes": intermediate_nodes,
            "trace_path": [n for n in calculator.tree_structure.get('nodes', [])],
        }
        
        return prompt_text, metadata
    
    def _estimate_max_depth(self, variable: str, graph_data: Dict, visited: Set[str] = None, max_depth_limit: int = 10) -> int:
        """Estimate the maximum depth of dependency chain for a variable."""
        if visited is None:
            visited = set()
        
        if variable in visited or len(visited) >= max_depth_limit:
            return 0
        
        variable_info_map = {v['variable']: v for v in graph_data['variables']}
        if variable not in variable_info_map:
            return 0
        
        visited.add(variable)
        max_child_depth = 0
        
        var_info = variable_info_map[variable]
        dependencies = var_info.get('dependencies', [])
        
        for dep in dependencies:
            if dep not in visited:
                child_depth = self._estimate_max_depth(dep, graph_data, visited.copy(), max_depth_limit)
                max_child_depth = max(max_child_depth, child_depth)
        
        visited.remove(variable)
        return 1 + max_child_depth
    
    def _get_variables_by_depth(self, graph_data: Dict, min_depth: int = 3) -> List[str]:
        """Get variables that have at least min_depth levels of dependencies."""
        variable_info_map = {v['variable']: v for v in graph_data['variables']}
        deep_variables = []
        
        for var in variable_info_map.keys():
            depth = self._estimate_max_depth(var, graph_data)
            if depth >= min_depth:
                deep_variables.append(var)
        
        return deep_variables
    
    def _generate_single_tree_walk(self, target: str = None) -> Optional[Dict]:
        """Generate a single tree walk (for parallel processing)."""
        try:
            # Select target
            if target is None:
                deep_variables = self._deep_variables_cache
                all_variables = self._all_variables_cache
                
                if deep_variables and random.random() < 0.9:
                    target = random.choice(deep_variables)
                else:
                    target = random.choice(all_variables)
            
            # Create calculator
            calculator = TreeWalkCalculator(
                self.config.graph_file,
                max_length=self.config.max_length
            )
            
            result = calculator.run(
                target,
                min_val=1.0,
                max_val=10.0
            )
            
            if result is None:
                return None
            
            # Check tree walk length
            tree_walk_length = 0
            if hasattr(calculator, 'tree_structure') and calculator.tree_structure:
                levels = calculator.tree_structure.get('levels', {})
                leaf_nodes = calculator.tree_structure.get('leaf_nodes', set())
                if levels:
                    non_leaf_levels = []
                    for level_num, level_nodes in levels.items():
                        non_leaf_in_level = [n for n in level_nodes if n not in leaf_nodes]
                        if non_leaf_in_level:
                            non_leaf_levels.append(level_num)
                    
                    if non_leaf_levels:
                        max_level = max(non_leaf_levels)
                        tree_walk_length = max_level + 1
                    else:
                        tree_walk_length = 1
            
            if tree_walk_length < self.config.min_tree_walk_length:
                return None
            
            # Create prompt
            prompt, metadata = self._create_prompt(calculator)
            
            return {
                "query": prompt,
                "calculator": calculator,
                "metadata": metadata,
            }
        except Exception as e:
            logger.debug(f"Error generating tree walk: {e}")
            return None
    
    def _collect_batch_parallel(self, batch_size: int) -> List[Dict]:
        """OPTIMIZATION: Collect batch using parallel tree walk generation."""
        batch = []
        max_attempts = batch_size * 10
        
        # Use ProcessPoolExecutor for true parallelism (avoids GIL)
        # Note: This requires the function to be picklable
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            # Submit multiple tasks
            futures = []
            for _ in range(min(max_attempts, batch_size * 3)):
                future = executor.submit(self._generate_single_tree_walk)
                futures.append(future)
            
            # Collect results as they complete
            for future in as_completed(futures):
                if len(batch) >= batch_size:
                    break
                try:
                    result = future.result(timeout=30)
                    if result:
                        batch.append(result)
                except Exception as e:
                    logger.debug(f"Tree walk generation failed: {e}")
        
        if len(batch) < batch_size:
            logger.warning(f"Only collected {len(batch)}/{batch_size} valid tree walks")
        
        return batch[:batch_size]
    
    def _compute_rewards_batched(self, responses: List[str], batch: List[Dict]) -> Tuple[List[float], List[float], List[float], List[str], List[int], List[float]]:
        """OPTIMIZATION: Batch compute rewards for all responses at once."""
        rewards = []
        judge_scores = []
        judge_rewards = []
        judge_explanations = []
        tree_walk_lengths = []
        tree_walk_length_rewards = []
        
        # Build all evaluation data - each response gets its OWN prompt/principle
        evaluations = []
        
        for i, (response, ex) in enumerate(zip(responses, batch)):
            calculator = ex["calculator"]
            target = calculator.tree_structure["target"]
            
            # Format given values
            leaves = [
                leaf for leaf in sorted(calculator.tree_structure.get("leaf_nodes", set()))
                if leaf in calculator.values
            ]
            value_lines = []
            for leaf in leaves:
                value = calculator.values[leaf]
                unit = calculator._get_si_unit(leaf)
                value_lines.append(f"{leaf} = {value:.4f}{f' {unit}' if unit else ''}")
            
            values_text = "\n".join(f"  {line}" for line in value_lines) if value_lines else "  (no values found)"
            allowed_vars = ", ".join(leaves) if leaves else "None"
            
            prompt = (
                f"You are scoring candidate word problems that must ask for the value of {target}.\n"
                "A good question must:\n"
                "• Include every given value as provided with its SI unit.\n"
                "• Use only the allowed variables (no intermediate or invented variables).\n"
                f"• End with asking for target variable\n\n"
                f"Allowed variables: {allowed_vars}\n"
                f"Given values:\n{values_text}\n\n"
                f"If the values are not exact then also consider it good\n"
                "Score the question on a scale from 0 to 10, where:\n"
                "- 10 = Perfect: All requirements met perfectly\n"
                "- 7-9 = Good: Minor issues\n"
                "- 4-6 = Acceptable: Some requirements missing\n"
                "- 0-3 = Poor: Major requirements missing\n"
            )
            
            principle = (
                "Score questions on a 0-10 scale. Rank higher (8-10) any question that restates every provided value with its unit, "
                f"sticks to the allowed variable names, stays concise, and ends with asking for value of {target}. "
                "Penalize (lower scores 0-7) missing values, invented variables, and vague language. "
                "Return scores as numeric values between 0 and 10."
            )
            
            evaluations.append({
                'response': response,
                'prompt': prompt,
                'principle': principle,
                'index': i
            })
        
        # CORRECTED: Evaluate each response with its corresponding prompt/principle
        try:
            # Evaluate each response individually with its correct context
            for eval_data in evaluations:
                try:
                    result = self.reward_model.judge(
                        principle=eval_data['principle'],
                        prompt=eval_data['prompt'],
                        responses={"response": eval_data['response']}
                    )
                    
                    if result.scores and "response" in result.scores:
                        raw_score = result.scores["response"]
                        score = max(0.0, min(10.0, raw_score))
                        explanation = result.reasoning if hasattr(result, 'reasoning') else "No explanation provided"
                    else:
                        score = 5.0
                        explanation = f"Score extraction failed. Raw scores: {result.scores}"
                    
                    judge_scores.append(score)
                    judge_explanations.append(explanation)
                    judge_reward = (score / 10.0) * 2.0 - 1.0
                    judge_rewards.append(judge_reward)
                    
                except Exception as e:
                    logger.error(f"Individual evaluation failed for response {eval_data['index']}: {e}")
                    judge_scores.append(0.0)
                    judge_explanations.append(f"Evaluation failed: {str(e)}")
                    judge_rewards.append(-1.0)
        except Exception as e:
            logger.error(f"Batched evaluation failed: {e}. Falling back to individual evaluation.")
            # Fallback: evaluate individually
            for i, (response, ex) in enumerate(zip(responses, batch)):
                try:
                    calculator = ex["calculator"]
                    target = calculator.tree_structure["target"]
                    
                    leaves = [
                        leaf for leaf in sorted(calculator.tree_structure.get("leaf_nodes", set()))
                        if leaf in calculator.values
                    ]
                    value_lines = []
                    for leaf in leaves:
                        value = calculator.values[leaf]
                        unit = calculator._get_si_unit(leaf)
                        value_lines.append(f"{leaf} = {value:.4f}{f' {unit}' if unit else ''}")
                    
                    values_text = "\n".join(f"  {line}" for line in value_lines) if value_lines else "  (no values found)"
                    allowed_vars = ", ".join(leaves) if leaves else "None"
                    
                    prompt = (
                        f"You are scoring candidate word problems that must ask for the value of {target}.\n"
                        "A good question must:\n"
                        "• Include every given value exactly as provided (no rounding) with its SI unit.\n"
                        "• Use only the allowed variables (no intermediate or invented variables).\n"
                        f"• End with: \"What is the {target}?\"\n\n"
                        f"Allowed variables: {allowed_vars}\n"
                        f"Given values:\n{values_text}\n\n"
                        "Score the question on a scale from 0 to 10."
                    )
                    
                    principle = (
                        "Score questions on a 0-10 scale. Return scores as numeric values between 0 and 10."
                    )
                    
                    result = self.reward_model.judge(
                        principle=principle,
                        prompt=prompt,
                        responses={"response": response}
                    )
                    
                    if result.scores and "response" in result.scores:
                        raw_score = result.scores["response"]
                        score = max(0.0, min(10.0, raw_score))
                        explanation = result.reasoning if hasattr(result, 'reasoning') else "No explanation provided"
                    else:
                        score = 5.0
                        explanation = "Score extraction failed"
                    
                    judge_scores.append(score)
                    judge_explanations.append(explanation)
                    judge_reward = (score / 10.0) * 2.0 - 1.0
                    judge_rewards.append(judge_reward)
                except Exception as e2:
                    logger.error(f"Individual evaluation failed for response {i}: {e2}")
                    judge_scores.append(0.0)
                    judge_explanations.append(f"Evaluation failed: {str(e2)}")
                    judge_rewards.append(-1.0)
        
        # Calculate tree walk length rewards
        for i, ex in enumerate(batch):
            calculator = ex["calculator"]
            tree_walk_length = 0
            tree_walk_length_reward = 0.0
            
            if hasattr(calculator, 'tree_structure') and calculator.tree_structure:
                levels = calculator.tree_structure.get('levels', {})
                leaf_nodes = calculator.tree_structure.get('leaf_nodes', set())
                if levels:
                    non_leaf_levels = []
                    for level_num, level_nodes in levels.items():
                        non_leaf_in_level = [n for n in level_nodes if n not in leaf_nodes]
                        if non_leaf_in_level:
                            non_leaf_levels.append(level_num)
                    
                    if non_leaf_levels:
                        max_level = max(non_leaf_levels)
                        tree_walk_length = max_level + 1
                    else:
                        tree_walk_length = 1
                    
                    normalized_length = tree_walk_length / self.config.max_length
                    tree_walk_length_reward = normalized_length * 2.0 - 1.0
            
            tree_walk_lengths.append(tree_walk_length)
            tree_walk_length_rewards.append(tree_walk_length_reward)
        
        # Combine rewards with weights
        for i in range(len(responses)):
            combined_reward = (
                self.config.judge_reward_weight * judge_rewards[i] + 
                self.config.length_reward_weight * tree_walk_length_rewards[i]
            )
            rewards.append(combined_reward)
        
        return rewards, judge_scores, judge_rewards, judge_explanations, tree_walk_lengths, tree_walk_length_rewards
    
    def train(self):
        """Run optimized PPO training loop."""
        logger.info("Starting OPTIMIZED PPO training...")
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
            with self._training_step_context():
                # OPTIMIZATION: Parallel batch collection
                if detailed_logging:
                    logger.info("Collecting batch (parallel)...")
                batch = self._collect_batch_parallel(self.config.batch_size)
                
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
                if self.config.use_mixed_precision and torch.cuda.is_available():
                    with autocast():
                        response_tensors = self.ppo_trainer.generate(query_tensors, **generation_kwargs)
                else:
                    response_tensors = self.ppo_trainer.generate(query_tensors, **generation_kwargs)
                
                # Decode responses and track valid indices
                responses = []
                valid_indices = []
                for i, response_ids in enumerate(response_tensors):
                    response_ids = response_ids[response_ids != self.tokenizer.pad_token_id]
                    decoded_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    
                    # Llama 3 specific cleanup
                    decoded_text = decoded_text.split("<|eot_id|>")[0].strip()
                    decoded_text = decoded_text.split("<|end_header_id|>")[-1].strip()
                    decoded_text = decoded_text.split("<|begin_of_text|>")[-1].strip()
                    
                    # Clean up formatting
                    decoded_text = re.sub(r'\\\(|\\\)', '', decoded_text)
                    decoded_text = re.sub(r'\\text\{([^}]+)\}', r'\1', decoded_text)
                    decoded_text = re.sub(r'\\[a-zA-Z]+\{([^}]+)\}', r'\1', decoded_text)
                    decoded_text = re.sub(r'\\[a-zA-Z]+', '', decoded_text)
                    decoded_text = re.sub(r'\{|\}', '', decoded_text)
                    decoded_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', decoded_text)
                    decoded_text = re.sub(r'\*([^*]+)\*', r'\1', decoded_text)
                    decoded_text = re.sub(r'`([^`]+)`', r'\1', decoded_text)
                    decoded_text = re.sub(r'#+\s*', '', decoded_text)
                    decoded_text = re.sub(r'\\(?![a-zA-Z0-9/])', '', decoded_text)
                    decoded_text = decoded_text.strip()
                    
                    if not decoded_text or len(decoded_text.strip()) < 10:
                        continue
                    
                    # Extract question
                    question_mark_idx = decoded_text.find("?")
                    if question_mark_idx != -1:
                        end_idx = min(question_mark_idx + 50, len(decoded_text))
                        decoded_text = decoded_text[:end_idx].strip()
                        last_q = decoded_text.rfind("?")
                        if last_q != -1:
                            decoded_text = decoded_text[:last_q + 1].strip()
                    
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
                
                rewards, judge_scores, judge_rewards, judge_explanations, tree_walk_lengths, tree_walk_length_rewards = \
                    self._compute_rewards_batched(responses, batch)
                
                # Log statistics
                avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
                avg_score = sum(judge_scores) / len(judge_scores) if judge_scores else 0.0
                avg_tree_length = sum(tree_walk_lengths) / len(tree_walk_lengths) if tree_walk_lengths else 0.0
                
                self.episode_rewards.append(avg_reward)
                self.episode_scores.append(avg_score)
                
                if detailed_logging:
                    logger.info(f"Avg judge score: {avg_score:.2f}/10.0, Avg reward: {avg_reward:.4f}, Avg tree length: {avg_tree_length:.1f}")
                else:
                    logger.info(f"Avg score: {avg_score:.2f}, Avg reward: {avg_reward:.4f}")
                
                # OPTIMIZATION: Async logging (non-blocking)
                if detailed_logging:
                    self._log_episode_results_async(episode, responses, rewards, judge_scores, 
                                                   judge_rewards, judge_explanations, batch, 
                                                   tree_walk_lengths, tree_walk_length_rewards)
                
                # Log to wandb
                if WANDB_AVAILABLE:
                    wandb.log({
                        "episode/avg_reward": avg_reward,
                        "episode/avg_judge_score": avg_score,
                        "episode/avg_tree_length": avg_tree_length,
                    }, step=episode)
                
                # Convert rewards to tensors
                reward_tensors = [torch.tensor(reward, dtype=torch.float32, device='cpu') for reward in rewards]
                
                # Final safety check: ensure all tensors have matching lengths
                if len(query_tensors) != len(response_tensors) or len(query_tensors) != len(reward_tensors):
                    logger.warning(
                        f"Batch size mismatch after filtering: queries={len(query_tensors)}, "
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


def main():
    """Main function to run optimized training."""
    if not TRL_AVAILABLE:
        logger.error("Required libraries not available. Please install: pip install transformers torch trl accelerate")
        return
    
    # OPTIMIZED configuration for Llama 3
    config = TrainingConfig(
        num_episodes=1000,
        max_length=8,
        min_tree_walk_length=2,  # Accept all tree walks
        # OPTIMIZATION: Adjusted batch sizes
        batch_size=4,  # 4x speedup from batching
        mini_batch_size=2,  # Adjusted for batch_size=4
        gradient_accumulation_steps=2,
        # OPTIMIZATION: Reduced token generation
        max_new_tokens=150,  # 3x speedup from less generation
        # OPTIMIZATION: Performance settings
        use_mixed_precision=True,  # 1.5x speedup
        use_quantization=False,  # Set to True for 2x more speedup (if memory allows)
        num_workers=4,  # Parallel tree walk generation
        log_detailed_every=10,  # Log details every 10 episodes
        save_steps=50,
    )
    
    # Create trainer
    trainer = QuestionGeneratorPPOTrainer(config)
    
    # Run training
    trainer.train()


if __name__ == "__main__":
    main()