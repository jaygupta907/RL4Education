"""
Instruction fine-tuning script for Llama model.

This script performs supervised fine-tuning (SFT) on the dataset before RL training.
Uses TRL's SFTTrainer for efficient training.
"""
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from trl import SFTTrainer
from datasets import Dataset

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class InstructionTuningConfig:
    """Configuration for instruction fine-tuning."""
    # Model configuration
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    dataset_path: str = "dataset_new_format.json"
    output_dir: str = "/mnt/storage/ae21b026/models/instruction_tuned_model_pretrained"
    
    # Training configuration
    num_train_epochs: int = 6
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    max_seq_length: int = 2048
    
    # Optimization
    use_mixed_precision: bool = True
    use_quantization: bool = False
    gradient_checkpointing: bool = True
    
    # Saving
    save_steps: int = 100
    logging_steps: int = 10
    eval_steps: Optional[int] = None
    
    # Other
    seed: int = 42
    fp16: bool = False
    bf16: bool = True


def load_and_prepare_dataset(dataset_path: str, tokenizer):
    """
    Load dataset and convert to chat format for instruction tuning.
    
    The dataset should have 'prompt' and 'response' fields.
    We'll convert them to Llama 3 chat format.
    """
    logger.info(f"Loading dataset from {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} examples")
    
    def format_chat_messages(example):
        """Format prompt-response pairs into Llama 3 chat format."""
        # Use the same system prompt as in prompt_generator.py for consistency
        system_message = """You are a physics problem generator. Generate clear, realistic physics word problems """
        
        user_message = example['prompt']
        assistant_message = example['response']
        
        # Format according to Llama 3 chat template
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
        
        # Apply chat template
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        return {"text": formatted_text}
    
    # Convert to dataset format
    dataset = Dataset.from_list(data)
    
    # Apply formatting
    dataset = dataset.map(
        format_chat_messages,
        remove_columns=dataset.column_names,
        desc="Formatting dataset"
    )
    
    logger.info(f"Dataset formatted. Example: {dataset[0]['text'][:200]}...")
    
    return dataset


def train_instruction_model(config: InstructionTuningConfig):
    """Main training function for instruction fine-tuning."""
    logger.info("Starting instruction fine-tuning...")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Output directory: {config.output_dir}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    tokenizer.padding_side = "right"  # Right padding for training
    
    # Load and prepare dataset
    dataset = load_and_prepare_dataset(config.dataset_path, tokenizer)
    
    # Use full dataset for training (no validation split to maximize training data)
    train_dataset = dataset
    logger.info(f"Using full dataset for training: {len(train_dataset)} examples")
    
    # Prepare model
    logger.info("Loading model...")
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16 if config.bf16 else torch.float16,
    }
    
    if config.use_quantization:
        logger.info("Enabling 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = quantization_config
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs
    )
    
    if config.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    
    # Initialize wandb if available
    if WANDB_AVAILABLE:
        wandb.init(
            project="question-generator-instruction-tuning",
            name=f"instruction-tuning-{config.model_name.split('/')[-1]}",
            config={
                "model_name": config.model_name,
                "dataset_path": config.dataset_path,
                "num_train_epochs": config.num_train_epochs,
                "per_device_train_batch_size": config.per_device_train_batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "learning_rate": config.learning_rate,
                "warmup_steps": config.warmup_steps,
                "max_seq_length": config.max_seq_length,
                "use_quantization": config.use_quantization,
                "gradient_checkpointing": config.gradient_checkpointing,
            }
        )
        logger.info("Wandb initialized for logging.")
    
    # Training arguments - optimized for disk space
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_strategy="no",
        save_total_limit=1,  # Keep only 1 checkpoint to save disk space
        save_only_model=True,  # Don't save optimizer state to save disk space
        eval_strategy="no",  # Disable evaluation to simplify
        load_best_model_at_end=False,
        fp16=config.fp16,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        report_to="wandb" if WANDB_AVAILABLE else None,  # Enable wandb logging if available
        seed=config.seed,
        remove_unused_columns=True,  # Let trainer handle column removal
    )
    
    # Create trainer (no eval dataset to save memory/disk)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving final model to {config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    # Finish wandb run
    if WANDB_AVAILABLE:
        wandb.finish()
    
    logger.info("Instruction fine-tuning completed!")
    logger.info(f"Model saved to: {config.output_dir}")
    
    # Test the model with a sample prompt
    test_model_generation(model, tokenizer)
    
    return config.output_dir


def test_model_generation(model, tokenizer):
    """Test the fine-tuned model with multiple realistic sample prompts."""
    logger.info("=" * 80)
    logger.info("Testing fine-tuned model with multiple realistic examples...")
    logger.info("=" * 80)
    
    # Multiple realistic test prompts from the dataset
    test_examples = [
        {
            "name": "Cart on frictionless track",
            "prompt": """ Given the following values: 
1. mass = 6.50 kg
2. initial_velocity = 4.20 m/s
3. force = 5.50 N
4. time = 6.00 s
 and target_variable: final_velocity  

Calculation steps:
Step 1: Calculate acceleration (m/s²) using force / mass with inputs: force, mass
Step 2: Calculate final_velocity (m/s) using initial_velocity + acceleration * time with inputs: initial_velocity, acceleration, time

Generate a deep-reasoning physics question that tests a student's understanding of the relationship between final_velocity and the variables mass, initial_velocity, force, time"""
        },
        {
            "name": "Worker pushing crate",
            "prompt": """ Given the following values: 
1. force = 5.20 N
2. displacement = 3.50 m
 and target_variable: work

Calculation steps:
Step 1: Calculate work (J) using force * displacement with inputs: force, displacement

Generate a deep-reasoning physics question that tests a student's understanding of the relationship between work and the variables force, displacement"""
        },
        {
            "name": "Construction worker lifting toolbox",
            "prompt": """ Given the following values: 
1. mass = 5.25 kg
2. height = 8.40 m
3. gravity = 9.81 m/s²
 and target_variable: potential_energy

Calculation steps:
Step 1: Calculate potential_energy (J) using mass * gravity * height with inputs: mass, gravity, height

Generate a deep-reasoning physics question that tests a student's understanding of the relationship between potential_energy and the variables mass, height, gravity"""
        },
        {
            "name": "Electrical circuit",
            "prompt": """ Given the following values: 
1. voltage = 9.20 V
2. resistance = 8.30 Ω
3. time = 4.50 s
 and target_variable: energy

Calculation steps:
Step 1: Calculate current (A) using voltage / resistance with inputs: voltage, resistance
Step 2: Calculate power (W) using voltage * current with inputs: voltage, current
Step 3: Calculate energy (J) using power * time with inputs: power, time

Generate a deep-reasoning physics question that tests a student's understanding of the relationship between energy and the variables voltage, resistance, time"""
        },
        {
            "name": "Hockey puck deceleration",
            "prompt": """ Given the following values: 
1. mass = 1.85 kg
2. initial_velocity = 7.20 m/s
3. final_velocity = 3.40 m/s
4. time = 2.80 s
 and target_variable: friction_force

Calculation steps:
Step 1: Calculate acceleration (m/s²) using (final_velocity - initial_velocity) / time with inputs: final_velocity, initial_velocity, time
Step 2: Calculate friction_force (N) using mass * acceleration with inputs: mass, acceleration

Generate a deep-reasoning physics question that tests a student's understanding of the relationship between friction_force and the variables mass, initial_velocity, final_velocity, time"""
        },
        {
            "name": "Copper block heating",
            "prompt": """ Given the following values: 
1. mass = 8.50 kg
2. specific_heat = 2.40 J/(kg K)
3. temperature_change = 6.75 K
 and target_variable: heat

Calculation steps:
Step 1: Calculate heat (J) using mass * specific_heat * temperature_change with inputs: mass, specific_heat, temperature_change

Generate a deep-reasoning physics question that tests a student's understanding of the relationship between heat and the variables mass, specific_heat, temperature_change"""
        },
        {
            "name": "Rocket launch",
            "prompt": """ Given the following values: 
1. initial_velocity = 0.00 m/s
2. acceleration = 2.85 m/s²
3. time = 5.60 s
 and target_variable: displacement

Calculation steps:
Step 1: Calculate displacement (m) using initial_velocity * time + 0.5 * acceleration * time**2 with inputs: initial_velocity, acceleration, time

Generate a deep-reasoning physics question that tests a student's understanding of the relationship between displacement and the variables initial_velocity, acceleration, time"""
        }
    ]

    system_message = """You are a physics problem generator. Generate clear, realistic physics word problems """

    # Create log file for test results
    log_dir = "./checkpoints/logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"test_generation_{timestamp}.json")
    
    test_results = {
        "timestamp": timestamp,
        "test_examples": []
    }
    
    # Test each example
    for idx, example in enumerate(test_examples, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Test Example {idx}/{len(test_examples)}: {example['name']}")
        logger.info(f"{'='*80}")
        
        test_prompt = example["prompt"]
        
        # Format as chat
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": test_prompt}
        ]
        
        # Apply chat template with generation prompt
        formatted_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        logger.info(f"\n{'='*40} INPUT PROMPT {'='*40}")
        logger.info(f"\n{test_prompt}\n")
        
        # Tokenize
        inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)
        
        # Generate
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode response (only the new tokens)
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        logger.info(f"\n{'='*40} GENERATED RESPONSE {'='*40}")
        logger.info(f"\n{generated_text}\n")
        
        # Save to results
        test_results["test_examples"].append({
            "example_name": example["name"],
            "prompt": test_prompt,
            "generated_response": generated_text
        })
    
    # Save all results to log file
    with open(log_file_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"All test results saved to: {log_file_path}")
    logger.info(f"{'='*80}")


def main():
    """Main entry point."""
    config = InstructionTuningConfig(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        dataset_path="dataset_new_format.json",
        output_dir="/mnt/storage/ae21b026/models/instruction_tuned_model",
        num_train_epochs=10,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        max_seq_length=2048,
    )
    
    train_instruction_model(config)


if __name__ == "__main__":
    main()

