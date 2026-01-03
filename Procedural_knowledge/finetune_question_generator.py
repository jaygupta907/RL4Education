"""
Fine-tune Question Generator using PPO with Judge LLM Reward

This script fine-tunes the question generation LLM using Proximal Policy Optimization (PPO)
with rewards from the judge LLM. The judge evaluates whether generated questions correctly
ask for the solution trace based on the pruned tree walk.

Training Process:
1. Generate random tree walks with TreeWalkCalculator
2. Generate questions using the policy model (QuestionGenerator)
3. Evaluate questions using the judge LLM (QuestionJudge)
4. Update policy model using PPO based on judge rewards

Requirements:
    - transformers library: pip install transformers torch
    - trl library: pip install trl
    - peft library: pip install peft
    - bitsandbytes: pip install bitsandbytes (for quantization)
    - accelerate: pip install accelerate
"""

import logging
import json
import os
import random
import re
import torch
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
from tree_walk_calculation import TreeWalkCalculator
from generate_question_from_answer import QuestionGenerator
from question_judge import QuestionJudge

# Try to import required libraries
try:
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    TRL_AVAILABLE = True
except ImportError as e:
    TRL_AVAILABLE = False
    print(f"Warning: Required libraries not available: {e}")
    print("Install with: pip install transformers torch trl peft bitsandbytes accelerate")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('finetune_question_generator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning."""
    # Model configuration
    policy_model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    judge_model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    graph_file: str = "variable_concept_graph.json"
    
    # Training configuration
    max_length: int = 8 # Max tree walk length (increased to allow deeper dependency chains)
    num_episodes: int = 1000  # Number of training episodes
    batch_size: int = 1  # Batch size for PPO (1 question per step)
    mini_batch_size: int = 1  # Mini batch size (must satisfy: batch_size is multiple of mini_batch_size * gradient_accumulation_steps)
    gradient_accumulation_steps: int = 1  # Gradient accumulation steps
    
    # PPO hyperparameters
    learning_rate: float = 1.41e-5
    ppo_epochs: int = 4
    cliprange: float = 0.2  # Policy clipping range - limits how much the policy can change per update
    cliprange_value: float = 0.2  # Value function clipping range - limits value estimate updates for stability
    gamma: float = 1.0  # Discount factor (1.0 for immediate rewards)
    lam: float = 0.95  # GAE lambda
    
    # LoRA configuration (reduced for memory efficiency)
    lora_r: int = 4  # Reduced from 8 to save memory
    lora_alpha: int = 16  # Reduced proportionally
    lora_dropout: float = 0.1
    
    # Generation configuration
    max_new_tokens: int =500
    temperature: float = 0.5
    top_p: float = 0.9  # Nucleus sampling
    top_k: int = 50  # Top-k sampling
    repetition_penalty: float = 1.2  # Penalty for repetition (1.0 = no penalty, >1.0 = penalize)
    no_repeat_ngram_size: int = 2  # Prevent repeating n-grams
    
    # Output configuration
    output_dir: str = "./checkpoints/question_generator_ppo"
    save_steps: int = 10
    logging_steps: int = 1
    
    # Device configuration
    use_quantization: bool = True  # Use 4-bit quantization to save memory
    use_8bit: bool = False  # Use 8-bit quantization instead of 4-bit (less memory efficient but faster)
    max_memory: Optional[Dict] = None  # Max memory per device, e.g., {0: "20GiB", "cpu": "30GiB"}


class QuestionGeneratorPPOTrainer:
    """PPO Trainer for fine-tuning question generation model."""
    
    def _log_memory_usage(self, stage: str = ""):
        """Log current GPU memory usage for debugging."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
            logger.info(f"GPU Memory {stage}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Max={max_allocated:.2f}GB")
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the PPO trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize judge (reward model)
        logger.info("Initializing judge LLM...")
        self.judge = QuestionJudge(model_name=config.judge_model_name)
        self.judge._initialize_judge()
        
        if not self.judge.judge_pipeline:
            raise RuntimeError("Failed to initialize judge LLM. Cannot proceed with training.")
        
        # Initialize policy model
        self._initialize_policy_model()
        
        # Initialize tokenizer
        self._initialize_tokenizer()
        
        # Initialize PPO trainer
        self._initialize_ppo_trainer()
        
        # Statistics
        self.episode_rewards = []
        self.episode_scores = []
        
        # Cache for deep variables (variables with deeper dependency chains)
        self._deep_variables_cache = None
        self._all_variables_cache = None
    
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration."""
        if not self.config.use_quantization:
            return None
        
        if self.config.use_8bit:
            # 8-bit quantization (less memory efficient but faster)
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        else:
            # 4-bit quantization (more memory efficient)
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
    
    def _initialize_policy_model(self):
        """Initialize the policy model with value head and LoRA."""
        logger.info(f"Loading policy model: {self.config.policy_model_name}")
        logger.info(f"Using quantization: {self.config.use_quantization} ({'8-bit' if self.config.use_8bit else '4-bit' if self.config.use_quantization else 'None'})")
        
        quantization_config = self._get_quantization_config()
        
        # Prepare model kwargs
        model_kwargs = {
            "device_map": "auto",
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["torch_dtype"] = None  # Quantization handles dtype
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
        
        if self.config.max_memory:
            model_kwargs["max_memory"] = self.config.max_memory
        
        # Load base model with value head
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.config.policy_model_name,
            **model_kwargs
        )
        
        # Prepare for LoRA fine-tuning if using quantization
        if quantization_config:
            logger.info("Preparing model for k-bit training...")
            self.model.pretrained_model = prepare_model_for_kbit_training(
                self.model.pretrained_model
            )
        
        # Configure LoRA (using fewer target modules for memory efficiency)
        # Only target q_proj and v_proj to reduce memory usage
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=["q_proj", "v_proj"],  # Reduced from 4 to 2 modules
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA
        logger.info("Applying LoRA adapters...")
        self.model.pretrained_model = get_peft_model(
            self.model.pretrained_model, lora_config
        )
        
        # Enable gradient checkpointing to save memory
        logger.info("Enabling gradient checkpointing...")
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
        
        logger.info("Policy model loaded successfully.")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        # Load reference model (frozen copy for PPO)
        logger.info("Loading reference model...")
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.config.policy_model_name,
            **model_kwargs
        )
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        logger.info("Reference model loaded successfully.")
    
    def _initialize_tokenizer(self):
        """Initialize the tokenizer."""
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.policy_model_name)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("Tokenizer loaded successfully.")
    
    def _initialize_ppo_trainer(self):
        """Initialize the PPO trainer."""
        logger.info("Initializing PPO trainer...")
        
        # PPO configuration with memory-efficient settings
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
            log_with=None,  # Can set to "wandb" or "tensorboard" if desired
            project_kwargs={"logging_dir": self.config.output_dir},
            optimize_cuda_cache=True,  # Optimize CUDA cache to reduce memory usage
            gradient_checkpointing=True,  # Use gradient checkpointing to save memory
        )
        
        # Create PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )
        
        logger.info("PPO trainer initialized successfully.")
    
    def _create_prompt(self, calculator: TreeWalkCalculator) -> Tuple[str, Dict]:
        """
        Create prompt for question generation.
        
        Args:
            calculator: TreeWalkCalculator instance with completed calculation
            
        Returns:
            Tuple of (prompt_text, metadata_dict)
        """
        target = calculator.tree_structure['target']
        
        # Get questions context (similar to QuestionGenerator)
        generator = QuestionGenerator()
        questions_dict = generator._get_questions_for_nodes(calculator)
        questions_context = generator._format_questions_context(questions_dict, calculator)
        
        # Extract given values (leaf nodes)
        given_values_list = []
        for leaf in sorted(calculator.tree_structure['leaf_nodes']):
            if leaf in calculator.values:
                given_values_list.append(leaf)
        
        # Format given values - show EXACT values with full precision
        values_examples = []
        for idx, var in enumerate(sorted(given_values_list), 1):
            value = calculator.values[var]
            # Show full precision - DO NOT round or simplify
            values_examples.append(f"{idx}. {var} = {value:.10f}")
        values_list_text = "\n".join(values_examples)
        
        allowed_vars_list = ", ".join(sorted(given_values_list))
        
        # Identify intermediate nodes
        leaf_nodes_set = calculator.tree_structure.get('leaf_nodes', set())
        all_nodes_set = calculator.tree_structure.get('nodes', set())
        intermediate_nodes = sorted([n for n in all_nodes_set if n not in leaf_nodes_set and n != target])
        
        # Get formulas
        formulas_text = ""
        if 'node_formulas' in calculator.tree_structure:
            formulas_text = "\nFormulas that can be used (for context only, DO NOT include calculated results):\n"
            for node, (formula, deps) in calculator.tree_structure['node_formulas'].items():
                if formula:
                    formulas_text += f"  {node} = {formula}\n"
        
        # System prompt - concise and direct
        system_prompt = """Generate physics/mathematics word problems in English only.

Rules:
• Use EXACT numeric values from the trace - no rounding or approximations
• Write in plain English (ASCII only) - no LaTeX, markdown, or Unicode symbols
• Include all given values with their exact numbers
• End with "What is the [target]?"
• English only - no other languages"""
        
        # User prompt
        user_prompt = f"""Generate a physics/mathematics word problem question.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TASK: Find the {target}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

YOU MUST USE THESE {len(given_values_list)} VALUES (include ALL with exact numbers):

{values_list_text}

ALLOWED VARIABLES ONLY: {allowed_vars_list}

Do NOT use: {', '.join(intermediate_nodes[:5]) if intermediate_nodes else 'any calculated/intermediate variables'}

{formulas_text if formulas_text else ""}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXAMPLE QUESTIONS (for phrasing style only):

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{questions_context if questions_context else "No examples available."}

⚠️ Note: Examples show phrasing style. Use ONLY variables from the allowed list above.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REQUIREMENTS:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Include ALL {len(given_values_list)} values above with their EXACT numbers

2. Use ONLY these variables: {allowed_vars_list}

3. Write in natural language with appropriate units

4. End with: "What is the {target}?"

5. Do NOT use vague terms like "certain", "some", "a distance" - use actual numbers

6. Do NOT mention intermediate/calculated variables

Generate the question now:"""
        
        # Format using chat template
        # For Qwen models, add explicit language instruction
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # Add explicit English-only instruction for Qwen models
            if "qwen" in self.config.policy_model_name.lower():
                # Prepend English instruction to reinforce language requirement
                prompt_text = prompt_text + "\n[ENGLISH ONLY - NO CHINESE/JAPANESE/KOREAN]\n"
        else:
            # Fallback format
            prompt_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        metadata = {
            "target": target,
            "leaf_nodes": given_values_list,
            "intermediate_nodes": intermediate_nodes,
        }
        
        return prompt_text, metadata
    
    
    
    def _estimate_max_depth(self, variable: str, graph_data: Dict, visited: Set[str] = None, max_depth_limit: int = 10) -> int:
        """
        Estimate the maximum depth of dependency chain for a variable.
        Uses DFS to find the longest path from this variable to leaf nodes.
        
        Args:
            variable: Variable name
            graph_data: Graph data dictionary
            visited: Set of visited variables (to prevent cycles)
            max_depth_limit: Maximum depth to search (to prevent infinite loops)
            
        Returns:
            Maximum depth estimate (0 for leaf nodes, higher for variables with deeper dependencies)
        """
        if visited is None:
            visited = set()
        
        if variable in visited or len(visited) >= max_depth_limit:
            return 0
        
        # If variable is not in defined variables, it's a leaf (base input)
        variable_info_map = {v['variable']: v for v in graph_data['variables']}
        if variable not in variable_info_map:
            return 0
        
        visited.add(variable)
        max_child_depth = 0
        
        # Get dependencies for this variable
        var_info = variable_info_map[variable]
        dependencies = var_info.get('dependencies', [])
        
        # Find the deepest dependency
        for dep in dependencies:
            if dep not in visited:
                child_depth = self._estimate_max_depth(dep, graph_data, visited.copy(), max_depth_limit)
                max_child_depth = max(max_child_depth, child_depth)
        
        visited.remove(variable)
        return 1 + max_child_depth
    
    def _get_variables_by_depth(self, graph_data: Dict, min_depth: int = 3) -> List[str]:
        """
        Get variables that have at least min_depth levels of dependencies.
        
        Args:
            graph_data: Graph data dictionary
            min_depth: Minimum depth threshold
            
        Returns:
            List of variable names with depth >= min_depth
        """
        variable_info_map = {v['variable']: v for v in graph_data['variables']}
        deep_variables = []
        
        for var in variable_info_map.keys():
            depth = self._estimate_max_depth(var, graph_data)
            if depth >= min_depth:
                deep_variables.append(var)
        
        return deep_variables
    
    def _collect_batch(self, batch_size: int) -> List[Dict]:
        """
        Collect a batch of training examples.
        
        Args:
            batch_size: Number of examples to collect
            
        Returns:
            List of training examples, each containing:
            - query: Input prompt
            - response: Generated question
            - reward: Reward score
            - calculator: TreeWalkCalculator instance
        """
        batch = []
        
        # Load graph data and cache variables if not already cached
        if self._deep_variables_cache is None or self._all_variables_cache is None:
            with open(self.config.graph_file, 'r') as f:
                graph_data = json.load(f)
            
            # Get variables with deeper dependency chains (at least 3 levels)
            # This ensures we get more interesting tree walks
            self._deep_variables_cache = self._get_variables_by_depth(graph_data, min_depth=3)
            self._all_variables_cache = [v['variable'] for v in graph_data['variables']]
            
            # Fallback to all variables if no deep variables found
            if not self._deep_variables_cache:
                logger.warning("No variables with depth >= 3 found. Using all variables.")
                self._deep_variables_cache = self._all_variables_cache.copy()
            else:
                logger.info(f"Found {len(self._deep_variables_cache)} variables with depth >= 3 (out of {len(self._all_variables_cache)} total)")
        
        deep_variables = self._deep_variables_cache
        all_variables = self._all_variables_cache
        
        for _ in range(batch_size):
            # Prefer selecting from deep variables, but allow some randomness
            # 90% chance to select from deep variables, 10% from all variables
            if deep_variables and random.random() < 0.9:
                target = random.choice(deep_variables)
            else:
                target = random.choice(all_variables)
            
            # Create calculator and run tree walk
            calculator = TreeWalkCalculator(
                self.config.graph_file,
                max_length=self.config.max_length
            )
            
            result = calculator.run(
                target,
                min_val=1.0,
                max_val=100.0
            )
            
            if result is None:
                logger.warning(f"Failed to calculate {target}. Skipping...")
                continue
            
            # Create prompt
            prompt, metadata = self._create_prompt(calculator)
            
            batch.append({
                "query": prompt,
                "calculator": calculator,
                "metadata": metadata,
            })
        
        return batch
    
    def train(self):
        """Run PPO training loop."""
        logger.info("Starting PPO training...")
        logger.info(f"Configuration: {self.config}")
        
        for episode in range(self.config.num_episodes):
            logger.info(f"\n{'='*60}")
            logger.info(f"Episode {episode + 1}/{self.config.num_episodes}")
            logger.info(f"{'='*60}")
            
            # Log memory at start of episode and reset max memory stats periodically
            if episode % 5 == 0:  # Log every 5 episodes to avoid spam
                self._log_memory_usage(f"Episode {episode + 1} start")
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()  # Reset peak stats for accurate monitoring
            
            # Collect batch
            logger.info("Collecting batch...")
            batch = self._collect_batch(self.config.batch_size)
            
            if len(batch) < self.config.mini_batch_size:
                logger.warning(f"Batch size ({len(batch)}) too small. Skipping episode...")
                continue
            
            # Extract queries
            queries = [ex["query"] for ex in batch]
            
            # Tokenize queries (PPO trainer expects list of 1D tensors)
            # Don't move to device if using device_map="auto" - let the model handle it
            tokenized = self.tokenizer(
                queries,
                return_tensors="pt",
                padding=True,
                truncation=True,
                padding_side="left",
            )
            
            # Extract query tensors as list of 1D tensors
            # Only move to device if not using quantization (quantized models handle device placement)
            if not self.config.use_quantization:
                query_tensors = [ids.to(self.device) for ids in tokenized.input_ids]
            else:
                query_tensors = [ids for ids in tokenized.input_ids]
            
            # Generate responses using PPO trainer
            logger.info("Generating questions...")
            
            # Create stop sequences to prevent off-topic generation
            stop_sequences = [
                "\n\nGiven:",
                "\n\nAnswer this",
                "\n\nSee options",
                "\n\nStep-by-step",
                "\n\nExplanation:",
                "\n\nExample:",
                "\n\n**Note:**",
                "\n\nTAOPEXPLAIN",
                "\n\nProblem Description",
                "\n\nConsider point",
                "\n\nLet the function",
                "\n\nAssistant:",
            ]
            
            generation_kwargs = {
                "max_new_tokens": self.config.max_new_tokens,
                "min_new_tokens": 10,  # Minimum tokens to ensure complete questions
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
            
            
            # Add stop sequences if supported by the model
            # Note: Some models may not support stop_sequences directly in generate()
            # We'll handle filtering in post-processing instead
            
            # Check memory before generation and reduce max_new_tokens if needed
            effective_generation_kwargs = generation_kwargs.copy()
            if torch.cuda.is_available() and episode > 0:
                allocated_gb = torch.cuda.memory_allocated() / 1024**3
                reserved_gb = torch.cuda.memory_reserved() / 1024**3
                # If we're using more than 85% of reserved memory, reduce max_new_tokens
                if reserved_gb > 0 and allocated_gb / reserved_gb > 0.85:
                    original_max = effective_generation_kwargs["max_new_tokens"]
                    effective_generation_kwargs["max_new_tokens"] = min(original_max, 300)
                    if effective_generation_kwargs["max_new_tokens"] < original_max:
                        logger.warning(f"Reduced max_new_tokens from {original_max} to {effective_generation_kwargs['max_new_tokens']} due to high memory usage ({allocated_gb:.2f}GB/{reserved_gb:.2f}GB)")
            
            response_tensors = self.ppo_trainer.generate(query_tensors, **effective_generation_kwargs)
            del effective_generation_kwargs  # Clean up
            
            # Decode responses
            responses = []
            logger.info("\n" + "="*60)
            logger.info("Generated Questions:")
            logger.info("="*60)
            for i, response_ids in enumerate(response_tensors):
                # Remove padding tokens
                response_ids = response_ids[response_ids != self.tokenizer.pad_token_id]
                decoded_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                # Clean up special tokens
                decoded_text = decoded_text.split("<|im_end|>")[0].strip()
                decoded_text = decoded_text.split("<|end_of_text|>")[0].strip()
                
                # Remove LaTeX formatting (e.g., \(, \), \text{}, etc.)
                decoded_text = re.sub(r'\\\(|\\\)', '', decoded_text)  # Remove \( and \)
                decoded_text = re.sub(r'\\text\{([^}]+)\}', r'\1', decoded_text)  # Replace \text{...} with content
                decoded_text = re.sub(r'\\[a-zA-Z]+\{([^}]+)\}', r'\1', decoded_text)  # Remove other LaTeX commands like \frac, \sqrt, etc.
                decoded_text = re.sub(r'\\[a-zA-Z]+', '', decoded_text)  # Remove standalone LaTeX commands without braces
                decoded_text = re.sub(r'\{|\}', '', decoded_text)  # Remove remaining curly braces
                
                # Remove markdown formatting
                decoded_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', decoded_text)  # Remove **bold**
                decoded_text = re.sub(r'\*([^*]+)\*', r'\1', decoded_text)  # Remove *italic* (but be careful not to remove multiplication)
                decoded_text = re.sub(r'`([^`]+)`', r'\1', decoded_text)  # Remove `code`
                decoded_text = re.sub(r'#+\s*', '', decoded_text)  # Remove markdown headers
                
                # Remove any remaining standalone backslashes (LaTeX artifacts)
                # Keep backslashes that are part of units (like "m/s") by checking context
                decoded_text = re.sub(r'\\(?![a-zA-Z0-9/])', '', decoded_text)  # Remove backslashes not followed by alphanumeric or /
                
                # No language filtering - keep all characters as generated
                decoded_text = decoded_text.strip()
                
                # Final validation: ensure we have at least some English text
                if decoded_text and not re.search(r'[A-Za-z]', decoded_text):
                    logger.warning(f"Response {i+1} contains no English letters after filtering. Skipping.")
                    continue
                
                # Final cleanup: remove any remaining formatting artifacts
                decoded_text = re.sub(r'\s+', ' ', decoded_text)  # Normalize whitespace
                decoded_text = decoded_text.strip()
                
                # Skip if text is empty after filtering
                if not decoded_text or len(decoded_text.strip()) < 10:
                    logger.warning(f"Response {i+1} is too short or empty after filtering. Skipping...")
                    continue
                
                # Extract only the question part - stop at common question-ending patterns
                # Look for the question mark and truncate after it (with some buffer for units)
                question_mark_idx = decoded_text.find("?")
                if question_mark_idx != -1:
                    # Take up to 50 characters after the question mark to allow for units/formatting
                    end_idx = min(question_mark_idx + 50, len(decoded_text))
                    decoded_text = decoded_text[:end_idx].strip()
                    # Find the last complete sentence/question
                    last_q = decoded_text.rfind("?")
                    if last_q != -1:
                        decoded_text = decoded_text[:last_q + 1].strip()
                
                # Remove common unwanted patterns that indicate off-topic content
                unwanted_patterns = [
                    "Given:", "Answer this question", "See options", "Let's break down",
                    "Step-by-step", "Explanation:", "Example:", "Note:", "**Note:**",
                    "TAOPEXPLAIN", "Problem Description", "Consider point", "Let the function"
                ]
                for pattern in unwanted_patterns:
                    if pattern.lower() in decoded_text.lower():
                        # If we find these patterns, try to extract just the question part before them
                        pattern_idx = decoded_text.lower().find(pattern.lower())
                        if pattern_idx > 0:
                            # Take only the part before the unwanted pattern
                            decoded_text = decoded_text[:pattern_idx].strip()
                            # Find the last question mark in this truncated version
                            last_q = decoded_text.rfind("?")
                            if last_q != -1:
                                decoded_text = decoded_text[:last_q + 1].strip()
                
                responses.append(decoded_text)
                logger.info(f"\nQuestion {i+1}:")
                logger.info(f"{decoded_text}\n")
            logger.info("="*60 + "\n")
            
            # Check if we have valid responses
            if not responses or len(responses) == 0:
                logger.warning("No valid responses generated. Skipping episode...")
                continue
            
            # Ensure responses match batch size
            if len(responses) != len(batch):
                logger.warning(f"Mismatch: {len(responses)} responses for {len(batch)} batch items. Truncating...")
                responses = responses[:len(batch)]
                batch = batch[:len(responses)]
            
            # Compute rewards using judge
            logger.info("Computing rewards...")
            logger.info("="*60)
            logger.info("Question Evaluation:")
            logger.info("="*60)
            rewards = []
            judge_scores = []
            judge_explanations = []
            for i, (response, ex) in enumerate(zip(responses, batch)):
                # Evaluate using judge (get both score and explanation)
                result = self.judge.evaluate(ex["calculator"], response)
                
                if result:
                    score = result.get("score", 0.0)
                    explanation = result.get("explanation", "No explanation provided")
                    judge_scores.append(score)
                    judge_explanations.append(explanation)
                    # Normalize score from 0-10 to -1 to 1 for PPO
                    reward = (score / 10.0) * 2.0 - 1.0
                else:
                    score = 0.0
                    explanation = "Evaluation failed"
                    judge_scores.append(0.0)
                    judge_explanations.append(explanation)
                    reward = -1.0  # Penalty for failed evaluation
                
                rewards.append(reward)
                logger.info(f"\nQuestion {i+1}:")
                logger.info(f"  Score: {score:.2f}/10.0")
                logger.info(f"  Reward: {reward:.4f}")
                logger.info(f"  Explanation: {explanation}")
                logger.info("")
                
                # Clear result to free memory
                del result
            
            logger.info("="*60)
            
            # Clear judge pipeline cache if possible
            if hasattr(self.judge, 'judge_pipeline') and self.judge.judge_pipeline is not None:
                if hasattr(self.judge.judge_pipeline, 'model'):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Log statistics
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            self.episode_rewards.append(avg_reward)
            
            if judge_scores:
                avg_score = sum(judge_scores) / len(judge_scores)
                self.episode_scores.append(avg_score)
                logger.info(f"Average judge score: {avg_score:.2f}/10.0")
                logger.info(f"Average reward: {avg_reward:.4f}")
            
            # Convert rewards to tensors (PPO trainer requires tensors, not floats)
            # Create on CPU first to avoid GPU memory fragmentation
            reward_tensors = [torch.tensor(reward, dtype=torch.float32, device='cpu') for reward in rewards]
            
            # Train with PPO
            logger.info("Training with PPO...")
            try:
                stats = self.ppo_trainer.step(
                    query_tensors,
                    response_tensors,
                    reward_tensors
                )
                logger.info(f"PPO Stats: {stats}")
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"Out of memory during PPO step at episode {episode + 1}")
                logger.error(f"Memory stats: Allocated={torch.cuda.memory_allocated()/1024**3:.2f}GB, Reserved={torch.cuda.memory_reserved()/1024**3:.2f}GB")
                # Clear everything and skip this episode
                del reward_tensors, query_tensors, response_tensors, tokenized, batch, responses
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
                logger.warning("Skipping episode due to OOM. Consider reducing max_new_tokens or batch_size.")
                continue
            
            # Explicitly delete tensors immediately after PPO step to free memory
            del reward_tensors
            del query_tensors
            del response_tensors
            del tokenized
            
            # Clear Python variables that might hold references
            queries = None
            responses = None
            rewards = None
            judge_scores = None
            
            # Clear batch data (calculator objects might hold large state)
            batch = None
            
            # Clear gradients
            if hasattr(self.model, 'zero_grad'):
                self.model.zero_grad()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear cache to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all operations are complete before clearing
            
            # Save checkpoint
            if (episode + 1) % self.config.save_steps == 0:
                checkpoint_path = f"{self.config.output_dir}/checkpoint-{episode + 1}"
                os.makedirs(checkpoint_path, exist_ok=True)
                logger.info(f"Saving checkpoint to {checkpoint_path}...")
                # Save LoRA adapters (PEFT models have save_pretrained method)
                self.model.pretrained_model.save_pretrained(checkpoint_path)
                self.tokenizer.save_pretrained(checkpoint_path)
        
        # Save final model
        os.makedirs(self.config.output_dir, exist_ok=True)
        logger.info(f"Saving final model to {self.config.output_dir}...")
        # Save LoRA adapters (PEFT models have save_pretrained method)
        self.model.pretrained_model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info("Training completed!")
        logger.info(f"Average reward over all episodes: {sum(self.episode_rewards) / len(self.episode_rewards):.4f}")
        if self.episode_scores:
            logger.info(f"Average judge score over all episodes: {sum(self.episode_scores) / len(self.episode_scores):.2f}/10.0")


def main():
    """Main function to run training."""
    if not TRL_AVAILABLE:
        logger.error("Required libraries not available. Please install: pip install transformers torch trl peft bitsandbytes accelerate")
        return
    
    # Create configuration with memory-efficient settings
    config = TrainingConfig(
        num_episodes=100,
        max_length=8,  # Increased from 6 to allow deeper tree walks
        batch_size=1,  # Generate 1 question per step
        mini_batch_size=1,  # Mini batch size
        gradient_accumulation_steps=1,  # Gradient accumulation steps
        save_steps=10,
        use_quantization=True,  # Enable 4-bit quantization
        use_8bit=False,  # Use 4-bit (more memory efficient)
        # Optional: Set max memory per device (uncomment if needed)
        # max_memory={0: "20GiB", "cpu": "30GiB"},
    )
    
    # Create trainer
    trainer = QuestionGeneratorPPOTrainer(config)
    
    # Run training
    trainer.train()


if __name__ == "__main__":
    main()

