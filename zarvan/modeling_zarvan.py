# zarvan/modeling_zarvan.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from transformers import PreTrainedModel, AutoConfig, AutoModel
from transformers.modeling_outputs import BaseModelOutput
from .configuration_zarvan import ZarvanConfig

# --- Helper Classes & Functions (Your original code, slightly adapted) ---

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]
        return x.permute(1, 0, 2)

class _HolisticExtractor(nn.Module):
    def __init__(self, config: ZarvanConfig):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.s_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.combine = nn.Linear(config.embed_dim, config.embed_dim)

    def forward(self, x):
        B, S, E = x.shape
        s, v = self.s_proj(x), self.v_proj(x)
        s = s.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        weights = F.softmax(s, dim=2)
        head_outputs = torch.sum(weights * v, dim=2, keepdim=True)
        return self.combine(head_outputs.reshape(B, 1, E))

class _AssociativeExtractor(nn.Module):
    def __init__(self, config: ZarvanConfig):
        super().__init__()
        self.s_proj = nn.Linear(config.embed_dim, 1)
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim)

    def forward(self, x):
        s, v = self.s_proj(x), self.v_proj(x)
        weights = F.softmax(s, dim=1)
        return torch.sum(weights * v, dim=1, keepdim=True)

class _SequentialExtractor(nn.Module):
    def __init__(self, config: ZarvanConfig):
        super().__init__()
        self.s_proj = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim), 
            nn.Sigmoid()
        )
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.angle_calculator = nn.Linear(config.embed_dim, config.embed_dim)
        self.out_proj = nn.Linear(config.embed_dim * 2, config.embed_dim)

    def forward(self, x):
        B, S, E = x.shape
        s, v = self.s_proj(x), self.v_proj(x)
        weights = torch.cumsum(s * v, dim=1)
        alpha = self.norm(self.angle_calculator(weights / S))
        omega = alpha * math.pi
        phases = torch.cat([torch.cos(omega), torch.sin(omega)], dim=-1)
        return self.out_proj(phases)

class _ZarvanBlock(nn.Module):
    def __init__(self, config: ZarvanConfig):
        super().__init__()
        self.input_adapter = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim), 
            nn.GELU(), 
            nn.LayerNorm(config.embed_dim)
        )
        self.holistic_ctx = _HolisticExtractor(config)
        self.associative_ctx = _AssociativeExtractor(config)
        self.sequential_ctx = _SequentialExtractor(config)
        self.expert_gate = nn.Sequential(
            nn.Linear(config.embed_dim, 3), 
            nn.SiLU()
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim), 
            nn.GELU(), 
            nn.Linear(config.hidden_dim, config.embed_dim)
        )
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, x):
        B, S, E = x.shape
        x_residual = x
        x_adapted = self.input_adapter(x)
        q_holistic = self.holistic_ctx(x_adapted)
        q_associative = self.associative_ctx(x_adapted)
        q_sequential = self.sequential_ctx(x_adapted)
        gates = self.expert_gate(x_adapted)
        g_h, g_a, g_s = gates.chunk(3, dim=-1)
        h_candidate = (
            g_h * q_holistic.expand(-1, S, -1) +
            g_a * q_associative.expand(-1, S, -1) +
            g_s * q_sequential
        )
        out = x_residual + self.ffn(self.norm(h_candidate))
        return out

# --- Main Model Classes for Hugging Face Integration ---

class ZarvanPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for
    downloading and loading pretrained models.
    """
    config_class = ZarvanConfig
    base_model_prefix = "zarvan"

    def _init_weights(self, module):
        """Initializes the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range if hasattr(self.config, 'initializer_range') else 0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range if hasattr(self.config, 'initializer_range') else 0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class Zarvan(ZarvanPreTrainedModel):
    def __init__(self, config: ZarvanConfig):
        super().__init__(config)
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_encoder = PositionalEncoding(config.embed_dim, config.max_len)
        self.layers = nn.ModuleList([_ZarvanBlock(config) for _ in range(config.num_layers)])
        self.output_head = nn.Linear(config.embed_dim, config.num_classes)
        
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None, # Not used by Zarvan, but kept for API consistency
        token_type_ids: Optional[torch.LongTensor] = None, # Not used by Zarvan
        position_ids: Optional[torch.LongTensor] = None,   # Not used by Zarvan
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            h = self.embedding(input_ids)
        elif inputs_embeds is not None:
            h = inputs_embeds
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        h = self.pos_encoder(h)
        for layer in self.layers:
            h = layer(h)
        
        logits = self.output_head(h)

        if not return_dict:
            return (logits,)

        return BaseModelOutput(
            last_hidden_state=h,
            hidden_states=None, # You can populate this if you save intermediate layers' outputs
            attentions=None,    # You can populate this if your model produces attention weights
        )

# --- Register with AutoModel ---
AutoConfig.register("zarvan", ZarvanConfig)
AutoModel.register(ZarvanConfig, Zarvan)