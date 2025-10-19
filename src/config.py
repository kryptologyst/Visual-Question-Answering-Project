"""
Configuration management for VQA project.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Configuration for VQA model."""
    model_name: str = "Salesforce/blip-vqa-base"
    device: Optional[str] = None
    use_pipeline: bool = False
    max_length: int = 50
    num_beams: int = 5
    temperature: float = 1.0


@dataclass
class AppConfig:
    """Configuration for the application."""
    debug: bool = False
    log_level: str = "INFO"
    data_dir: str = "data"
    models_dir: str = "models"
    output_dir: str = "outputs"


@dataclass
class UIConfig:
    """Configuration for UI components."""
    title: str = "Visual Question Answering Demo"
    theme: str = "light"
    sidebar_width: int = 300
    max_image_size: int = 512


class ConfigManager:
    """Manages application configuration."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or Path("config/config.yaml")
        self.model_config = ModelConfig()
        self.app_config = AppConfig()
        self.ui_config = UIConfig()
        
        if self.config_path.exists():
            self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix.lower() == '.yaml':
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            # Update configurations
            if 'model' in config_data:
                self.model_config = ModelConfig(**config_data['model'])
            if 'app' in config_data:
                self.app_config = AppConfig(**config_data['app'])
            if 'ui' in config_data:
                self.ui_config = UIConfig(**config_data['ui'])
                
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        self.config_path.parent.mkdir(exist_ok=True)
        
        config_data = {
            'model': asdict(self.model_config),
            'app': asdict(self.app_config),
            'ui': asdict(self.ui_config)
        }
        
        try:
            with open(self.config_path, 'w') as f:
                if self.config_path.suffix.lower() == '.yaml':
                    yaml.dump(config_data, f, default_flow_style=False)
                else:
                    json.dump(config_data, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        return self.model_config
    
    def get_app_config(self) -> AppConfig:
        """Get application configuration."""
        return self.app_config
    
    def get_ui_config(self) -> UIConfig:
        """Get UI configuration."""
        return self.ui_config
