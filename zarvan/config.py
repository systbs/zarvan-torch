# zarvan/config.py
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any

@dataclass
class ZarvanConfig:
    """
    Configuration class for the Zarvan model.

    This class stores the configuration of a Zarvan model, defining the model's
    architecture and hyperparameters.
    """
    vocab_size: int = 30522
    embed_dim: int = 256
    hidden_dim: int = 1024
    num_heads: int = 4
    num_layers: int = 6
    num_classes: int = 2
    max_len: int = 512
    dropout_prob: float = 0.1
    initializer_range: float = 0.02

    def to_dict(self) -> Dict[str, Any]:
        """Serializes this instance to a Python dictionary."""
        return asdict(self)

    def save_pretrained(self, save_directory: str):
        """
        Saves the configuration object to a file in JSON format.
        
        Args:
            save_directory (str): Directory where the configuration file will be saved.
        """
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        config_path = path / "config.json"
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: str):
        """
        Instantiates a configuration object from a file saved in a directory.
        
        Args:
            save_directory (str): Directory where the configuration file is located.
            
        Returns:
            ZarvanConfig: The configuration object.
        """
        path = Path(save_directory)
        config_path = path / "config.json"
        
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
            
        return cls(**config_dict)