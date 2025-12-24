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
import random
import torch
from typing import Dict, List, Optional, Tuple
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
    max_length: int = 4  # Max tree walk length
    num_episodes: int = 100  # Number of training episodes
    batch_size: int = 4  # Batch size for PPO
    mini_batch_size: int = 2  # Mini batch size
    gradient_accumulation_steps: int = 1
    
    # PPO hyperparameters
    learning_rate: float = 1.41e-5
    ppo_epochs: int = 4
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    gamma: float = 1.0  # Discount factor (1.0 for immediate rewards)
    lam: float = 0.95  # GAE lambda
    
    # LoRA configuration
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Generation configuration
    max_new_tokens: int = 400
    temperature: float = 0.5
    
    # Output configuration
    output_dir: str = "./checkpoints/question_generator_ppo"
    save_steps: int = 10
    logging_steps: int = 1
    
    # Device configuration
    use_quantization: bool = True  # Use 4-bit quantization to save memory


class QuestionGeneratorPPOTrainer:
    """PPO Trainer for fine-tuning question generation model."""
    
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
    
    def _initialize_policy_model(self):
        """Initialize the policy model with value head and LoRA."""
        logger.info(f"Loading policy model: {self.config.policy_model_name}")
        
        if self.config.use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            quantization_config = None
        
        # Load base model with value head
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.config.policy_model_name,
            torch_dtype=torch.bfloat16 if not self.config.use_quantization else None,
            device_map="auto",
            quantization_config=quantization_config
        )
        
        # Prepare for LoRA fine-tuning
        if self.config.use_quantization:
            self.model.pretrained_model = prepare_model_for_kbit_training(
                self.model.pretrained_model
            )
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Qwen attention modules
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA
        self.model.pretrained_model = get_peft_model(
            self.model.pretrained_model, lora_config
        )
        
        # Enable gradient checkpointing
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
        
        logger.info("Policy model loaded successfully.")
        
        # Load reference model (frozen copy for PPO)
        logger.info("Loading reference model...")
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.config.policy_model_name,
            torch_dtype=torch.bfloat16 if not self.config.use_quantization else None,
            device_map="auto",
            quantization_config=quantization_config
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
        
        # PPO configuration
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
        
        # Format given values
        values_examples = []
        for idx, var in enumerate(sorted(given_values_list), 1):
            value = calculator.values[var]
            values_examples.append(f"{idx}. {var} = {value:.4f}")
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
        
        # System prompt
        system_prompt = """You are an expert physics/mathematics educator. Generate clear, natural word problem questions.

Key principles:
• Use ONLY the variables and values explicitly provided
• Include ALL given values with their EXACT numeric values
• Write naturally with appropriate physical units
• Example questions demonstrate phrasing style - do NOT copy their variable names"""
        
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
        else:
            # Fallback format
            prompt_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        metadata = {
            "target": target,
            "leaf_nodes": given_values_list,
            "intermediate_nodes": intermediate_nodes,
        }
        
        return prompt_text, metadata
    
    
    
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
        
        # Load graph data
        with open(self.config.graph_file, 'r') as f:
            graph_data = json.load(f)
        
        variables = [v['variable'] for v in graph_data['variables']]
        
        for _ in range(batch_size):
            # Randomly select target variable
            target = random.choice(variables)
            
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
            
            # Collect batch
            logger.info("Collecting batch...")
            batch = self._collect_batch(self.config.batch_size)
            
            if len(batch) < self.config.mini_batch_size:
                logger.warning(f"Batch size ({len(batch)}) too small. Skipping episode...")
                continue
            
            # Extract queries
            queries = [ex["query"] for ex in batch]
            
            # Tokenize queries (PPO trainer expects list of 1D tensors)
            tokenized = self.tokenizer(
                queries,
                return_tensors="pt",
                padding=True,
                truncation=True,
                padding_side="left",
            ).to(self.device)
            
            # Extract query tensors as list of 1D tensors
            query_tensors = [ids for ids in tokenized.input_ids]
            
            # Generate responses using PPO trainer
            logger.info("Generating questions...")
            generation_kwargs = {
                "max_new_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "return_prompt": False,
            }
            
            response_tensors = self.ppo_trainer.generate(query_tensors, **generation_kwargs)
            
            # Decode responses
            responses = []
            for response_ids in response_tensors:
                # Remove padding tokens
                response_ids = response_ids[response_ids != self.tokenizer.pad_token_id]
                decoded_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                # Clean up special tokens
                decoded_text = decoded_text.split("<|im_end|>")[0].strip()
                decoded_text = decoded_text.split("<|end_of_text|>")[0].strip()
                responses.append(decoded_text)
            
            # Compute rewards using judge
            logger.info("Computing rewards...")
            rewards = []
            judge_scores = []
            for i, (response, ex) in enumerate(zip(responses, batch)):
                # Evaluate using judge (get both score and explanation)
                result = self.judge.evaluate(ex["calculator"], response)
                
                if result:
                    score = result.get("score", 0.0)
                    judge_scores.append(score)
                    # Normalize score from 0-10 to -1 to 1 for PPO
                    reward = (score / 10.0) * 2.0 - 1.0
                else:
                    score = 0.0
                    judge_scores.append(0.0)
                    reward = -1.0  # Penalty for failed evaluation
                
                rewards.append(reward)
                logger.info(f"  Question {i+1}: Score={score:.2f}/10.0, Reward={reward:.4f}")
            
            # Log statistics
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            self.episode_rewards.append(avg_reward)
            
            if judge_scores:
                avg_score = sum(judge_scores) / len(judge_scores)
                self.episode_scores.append(avg_score)
                logger.info(f"Average judge score: {avg_score:.2f}/10.0")
                logger.info(f"Average reward: {avg_reward:.4f}")
            
            # Convert rewards to tensors (PPO trainer requires tensors, not floats)
            reward_tensors = [torch.tensor(reward, dtype=torch.float32) for reward in rewards]
            
            # Train with PPO
            logger.info("Training with PPO...")
            stats = self.ppo_trainer.step(
                query_tensors,
                response_tensors,
                reward_tensors
            )
            
            logger.info(f"PPO Stats: {stats}")
            
            # Save checkpoint
            if (episode + 1) % self.config.save_steps == 0:
                checkpoint_path = f"{self.config.output_dir}/checkpoint-{episode + 1}"
                logger.info(f"Saving checkpoint to {checkpoint_path}...")
                self.ppo_trainer.save_model(checkpoint_path)
                self.tokenizer.save_pretrained(checkpoint_path)
        
        # Save final model
        logger.info(f"Saving final model to {self.config.output_dir}...")
        self.ppo_trainer.save_model(self.config.output_dir)
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
    
    # Create configuration
    config = TrainingConfig(
        num_episodes=100,
        batch_size=4,
        mini_batch_size=2,
        save_steps=10,
    )
    
    # Create trainer
    trainer = QuestionGeneratorPPOTrainer(config)
    
    # Run training
    trainer.train()


if __name__ == "__main__":
    main()

