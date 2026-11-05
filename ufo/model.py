""" Decoder-only transformer used for UFO modular-division experiments"""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import D_MODEL, MAX_CONTEXT_LENGTH, N_HEADS, N_LAYERS, Modality
from .data import VOCABULARY

class AttentionHead(nn.Module):
    def __init__(self, d_model: int, d_key: int) -> None:
        super().__init__()
        self.Wq = nn.Linear(d_model, d_key, bias=False)
        self.Wk = nn.Linear(d_model, d_key, bias=False)
        self.Wv = nn.Linear(d_model, d_key, bias=False)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_key = d_model // n_heads
        self.attn_heads = nn.ModuleList(
            [AttentionHead(d_model, self.d_key) for _ in range(n_heads)]
        )
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def _project(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, context_length = queries.shape[:2]
        q_weight = torch.cat([head.Wq.weight for head in self.attn_heads])
        k_weight = torch.cat([head.Wk.weight for head in self.attn_heads])
        v_weight = torch.cat([head.Wv.weight for head in self.attn_heads])
        shape = (batch_size, context_length, self.n_heads, self.d_key)
        return (
            F.linear(queries, q_weight).view(shape).transpose(1, 2),
            F.linear(keys, k_weight).view(shape).transpose(1, 2),
            F.linear(values, v_weight).view(shape).transpose(1, 2),
        )

    def attention_probabilities(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        queries, keys, _ = self._project(inputs, inputs, inputs)
        logits = queries @ keys.transpose(-2, -1)
        logits = logits / math.sqrt(self.d_key)
        logits = logits.masked_fill(mask == 0, float("-inf"))
        return F.softmax(logits, dim=-1)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, context_length = inputs.shape[:2]
        queries, keys, values = self._project(inputs, inputs, inputs)
        logits = queries @ keys.transpose(-2, -1)
        logits = logits / math.sqrt(self.d_key)
        logits = logits.masked_fill(mask == 0, float("-inf"))
        attention = F.softmax(logits, dim=-1)
        result = attention @ values
        result = result.transpose(1, 2).reshape(batch_size, context_length, -1)
        return self.Wo(result)

class FeedForward(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.input = nn.Linear(d_model, 4 * d_model, bias = False)
        self.output = nn.Linear(4 * d_model, d_model, bias = False)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        activations = F.relu(self.input(inputs))
        return self.output(activations), activations

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor, capture: Modality | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        attention_output = self.self_attn(inputs, mask)
        captured = None
        if capture == "attention":
            captured = self.self_attn.attention_probabilities(inputs, mask)
        attended = self.self_attn_norm(inputs + attention_output)
        ffn_output, activations = self.ffn(attended)
        if capture == "mlp":
            captured = activations
        return self.ffn_norm(attended + ffn_output), captured

@dataclass(frozen=True)
class TransformerOutput:
    logits: torch.Tensor
    layer_representations: tuple[torch.Tensor, ...]

class ArithmeticTransformer(nn.Module):
    def __init__(self, n_layers: int = N_LAYERS, n_heads: int = N_HEADS, d_model: int = D_MODEL, max_context_length: int = MAX_CONTEXT_LENGTH) -> None:
        super().__init__()
        self.embedding = nn.Embedding(len(VOCABULARY), d_model)
        self.register_buffer("position_encoding", self._position_encoding(max_context_length, d_model))
        self.register_buffer("self_attn_mask", torch.ones(max_context_length, max_context_length).tril())
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads) for _ in range(n_layers)])
        self.output = nn.Linear(d_model, len(VOCABULARY), bias=False)

    @staticmethod
    def _position_encoding(
        context_length: int,
        d_model: int,
    ) -> torch.Tensor:
        rows = [
            torch.tensor(
                [
                    math.sin(position / (10_000 ** (index / d_model)))
                    if index % 2 == 0
                    else math.cos(
                        position / (10_000 ** ((index - 1) / d_model))
                    )
                    for index in range(d_model)
                ]
            )
            for position in range(context_length)
        ]
        return torch.stack(rows)

    def forward(self, tokens: torch.Tensor, capture: Modality | None = None) -> TransformerOutput:
        context_length = tokens.shape[-1]
        hidden = (self.embedding(tokens) + self.position_encoding[:context_length])
        mask = self.self_attn_mask[:context_length, :context_length]
        layer_representations = []
        for block in self.blocks:
            hidden, captured = block(hidden, mask, capture)
            if captured is not None:
                layer_representations.append(captured)
        return TransformerOutput(logits=self.output(hidden), layer_representations=tuple(layer_representations))
