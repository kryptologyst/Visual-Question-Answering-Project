"""
Visual Question Answering Package

A modern implementation of Visual Question Answering using state-of-the-art
transformer models from Hugging Face.
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__email__ = "ai@example.com"

from .vqa_model import VQAModel, VQAResult, VQADataset, create_sample_images
from .config import ConfigManager, ModelConfig, AppConfig, UIConfig

__all__ = [
    "VQAModel",
    "VQAResult", 
    "VQADataset",
    "create_sample_images",
    "ConfigManager",
    "ModelConfig",
    "AppConfig",
    "UIConfig"
]
