"""
Common utilities for training.
"""
import re
import torch
import gc
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def training_step_context():
    """Context manager for proper cleanup after each training step."""
    try:
        yield
    finally:
        # Consolidated cleanup (no synchronize to avoid illegal access if tensors were freed elsewhere)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def log_memory_usage(stage: str = ""):
    """Log current GPU memory usage for debugging."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        logger.debug(f"GPU Memory {stage}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Max={max_allocated:.2f}GB")


def clean_decoded_text(decoded_text: str) -> str:
    """Clean up decoded text from model output."""
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
    
    return decoded_text


def extract_question(decoded_text: str) -> str:
    """Extract question from decoded text."""
    question_mark_idx = decoded_text.find("?")
    if question_mark_idx != -1:
        end_idx = min(question_mark_idx + 50, len(decoded_text))
        decoded_text = decoded_text[:end_idx].strip()
        last_q = decoded_text.rfind("?")
        if last_q != -1:
            decoded_text = decoded_text[:last_q + 1].strip()
    return decoded_text

