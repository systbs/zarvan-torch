# zarvan/model.py
import math
import json
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ZarvanConfig

class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""
    def __init__(self, embed_dim: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
        Returns:
            torch.Tensor: Tensor with added positional information.
        """
        # x is (batch, seq_len, embed_dim), pe is (max_len, 1, embed_dim)
        # We need to transpose x to (seq_len, batch, embed_dim) for broadcasting
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]
        return x.permute(1, 0, 2)

class HolisticExtractor(nn.Module):
    """Captures the global "gist" of the sequence using a multi-head weighted sum."""
    def __init__(self, config: ZarvanConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"Embedding dimension ({self.embed_dim}) must be divisible by the number of heads ({self.num_heads})."
            )
        self.head_dim = self.embed_dim // self.num_heads
        
        self.score_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.value_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.output_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, E = x.shape
        scores = self.score_proj(x).view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        values = self.value_proj(x).view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        weights = F.softmax(scores, dim=2) # Softmax over sequence length
        context = torch.sum(weights * values, dim=2) # (B, num_heads, head_dim)
        
        # Reshape and project to get the final holistic context vector
        context = context.reshape(B, E)
        return self.output_proj(context).unsqueeze(1) # (B, 1, E)

class AssociativeExtractor(nn.Module):
    """Focuses on salient tokens by computing a weighted average."""
    def __init__(self, config: ZarvanConfig):
        super().__init__()
        self.score_proj = nn.Linear(config.embed_dim, 1)
        self.value_proj = nn.Linear(config.embed_dim, config.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.score_proj(x)
        values = self.value_proj(x)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * values, dim=1, keepdim=True)
        return context

class SequentialExtractor(nn.Module):
    """Functions as a parallel state machine to capture historical context."""
    def __init__(self, config: ZarvanConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.value_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.phase_calculator = nn.Linear(config.embed_dim, config.embed_dim)
        self.output_proj = nn.Linear(config.embed_dim * 2, config.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, E = x.shape
        gates = torch.sigmoid(self.gate_proj(x))
        values = self.value_proj(x)
        
        accumulated_state = torch.cumsum(gates * values, dim=1)
        normalized_state = self.norm(self.phase_calculator(accumulated_state / S))
        
        omega = normalized_state * math.pi
        phases = torch.cat([torch.cos(omega), torch.sin(omega)], dim=-1)
        
        return self.output_proj(phases)

class FeedForward(nn.Module):
    """A standard two-layer feed-forward network with GELU activation."""
    def __init__(self, config: ZarvanConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_prob),
            nn.Linear(config.hidden_dim, config.embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ZarvanBlock(nn.Module):
    """A single block of the Zarvan architecture, containing a Mixture-of-Experts."""
    def __init__(self, config: ZarvanConfig):
        super().__init__()
        
        self.input_adapter = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim), 
            nn.GELU(), 
            nn.LayerNorm(config.embed_dim)
        )
        
        self.holistic_extractor = HolisticExtractor(config)
        self.associative_extractor = AssociativeExtractor(config)
        self.sequential_extractor = SequentialExtractor(config)

        self.expert_gate = nn.Sequential(
            nn.Linear(config.embed_dim, 3),
            nn.SiLU()
        )
        
        self.ffn = FeedForward(config)
        self.output_norm = nn.LayerNorm(config.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, E = x.shape
        residual = x
        
        x_adapted = self.input_adapter(x)

        # Get outputs from all three experts
        q_holistic = self.holistic_extractor(x_adapted)
        q_associative = self.associative_extractor(x_adapted)
        q_sequential = self.sequential_extractor(x_adapted)

        # Compute dynamic gates for each token
        gates = self.expert_gate(x_adapted)
        g_h, g_a, g_s = gates.chunk(3, dim=-1)
        
        # Combine expert outputs with learned gates
        # We need to expand the single context vectors to match the sequence length
        h = (
            g_h * q_holistic.expand(-1, S, -1) +
            g_a * q_associative.expand(-1, S, -1) +
            g_s * q_sequential
        )
        
        # Apply the second residual connection
        x = residual + self.ffn(self.output_norm(h))
        
        return x

class Zarvan(nn.Module):
    """
    The main Zarvan model, composed of an embedding layer, positional encoding,
    a stack of ZarvanBlocks, and an output head.
    """
    def __init__(self, config: ZarvanConfig):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_encoder = PositionalEncoding(config.embed_dim, config.max_len)
        self.dropout = nn.Dropout(config.dropout_prob)
        
        self.layers = nn.ModuleList([ZarvanBlock(config) for _ in range(config.num_layers)])
        
        # The output head for classification tasks
        self.output_head = nn.Linear(config.embed_dim, config.num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initializes weights of the model."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Input tensor of shape (batch_size, seq_len).
            
        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, seq_len, num_classes).
        """
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x)
            
        logits = self.output_head(x)
        return logits

    def save_pretrained(self, save_directory: str):
        """
        Saves the model weights and configuration to a directory.
        
        Args:
            save_directory (str): The directory to save the model and config to.
        """
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        
        self.config.save_pretrained(save_directory)
        
        torch.save(self.state_dict(), path / "pytorch_model.bin")
        print(f"Model and config saved to '{save_directory}'")

    @classmethod
    def from_pretrained(cls, save_directory: str):
        """
        Loads a model and its configuration from a directory.
        
        Args:
            save_directory (str): The directory to load the model and config from.
            
        Returns:
            Zarvan: An instance of the Zarvan model with loaded weights.
        """
        path = Path(save_directory)
        
        config = ZarvanConfig.from_pretrained(save_directory)
        
        model = cls(config)
        model_weights_path = path / "pytorch_model.bin"
        
        if model_weights_path.exists():
            model.load_state_dict(torch.load(model_weights_path))
            print(f"Model weights loaded from '{model_weights_path}'")
        else:
            print(f"Warning: No model weights found at '{model_weights_path}'. Model is randomly initialized.")
            
        model.eval() # Set to evaluation mode by default
        return model