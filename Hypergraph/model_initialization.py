"""
Model initialization utilities.
"""
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

logger = logging.getLogger(__name__)


def initialize_policy_model(config, device):
    """Initialize the policy model with value head."""
    # Use instruction-tuned model if available, otherwise use base model
    model_path = config.instruction_tuned_model_path if config.instruction_tuned_model_path else config.policy_model_name
    
    if config.instruction_tuned_model_path:
        logger.info(f"Loading instruction-tuned model: {config.instruction_tuned_model_path}")
    else:
        logger.info(f"Loading base policy model: {config.policy_model_name}")
    
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }
    
    # OPTIMIZATION: Add quantization config if enabled
    if config.use_quantization:
        logger.info("Enabling 8-bit quantization for faster inference...")
        # ValueHead models do not support CPU/disk offload; keep full model on GPU(s).
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        model_kwargs["quantization_config"] = quantization_config
        # Force all layers onto the first GPU so nothing is offloaded to CPU
        model_kwargs["device_map"] = {"": 0}
    
    if config.max_memory and not config.use_quantization:
        model_kwargs["max_memory"] = config.max_memory
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_path,
        **model_kwargs
    )
    
    logger.info("Enabling gradient checkpointing...")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    
    logger.info("Policy model loaded successfully.")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    logger.info("Loading reference model...")
    # Reference model should use the same model as policy (instruction-tuned if available)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_path,
        **model_kwargs
    )
    for param in ref_model.parameters():
        param.requires_grad = False
    logger.info("Reference model loaded successfully.")
    
    return model, ref_model


def initialize_tokenizer(config):
    """Initialize the tokenizer."""
    logger.info("Loading tokenizer...")
    # Use instruction-tuned model path if available, otherwise use base model
    tokenizer_path = config.instruction_tuned_model_path if config.instruction_tuned_model_path else config.policy_model_name
    
    if config.instruction_tuned_model_path:
        logger.info(f"Loading tokenizer from instruction-tuned model: {tokenizer_path}")
    else:
        logger.info(f"Loading tokenizer from base model: {tokenizer_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Llama 3 specific tokenizer setup
    if tokenizer.pad_token is None:
        # For Llama 3, use eos_token as pad_token
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Set padding side to left for decoder-only models
    tokenizer.padding_side = "left"
    
    logger.info("Tokenizer loaded successfully.")
    logger.info(f"Vocab size: {len(tokenizer)}, PAD token: {tokenizer.pad_token}")
    
    return tokenizer


def initialize_ppo_trainer(config, model, ref_model, tokenizer, wandb_available=False):
    """Initialize the PPO trainer."""
    logger.info("Initializing PPO trainer...")
    
    ppo_config = PPOConfig(
        model_name=config.policy_model_name,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        mini_batch_size=config.mini_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        ppo_epochs=config.ppo_epochs,
        cliprange=config.cliprange,
        cliprange_value=config.cliprange_value,
        gamma=config.gamma,
        lam=config.lam,
        log_with="wandb" if wandb_available else None,
        project_kwargs={"logging_dir": config.output_dir},
        optimize_cuda_cache=True,
        gradient_checkpointing=True,
        init_kl_coef=getattr(config, 'init_kl_coef', 0.1),
        adap_kl_ctrl=True,       # Keeps KL target stable automatically
        target_kl=getattr(config, 'target_kl', 6.0),
        max_grad_norm=getattr(config, 'max_grad_norm', 0.5),
    )
    
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )
    
    logger.info("PPO trainer initialized successfully.")
    
    return ppo_trainer

