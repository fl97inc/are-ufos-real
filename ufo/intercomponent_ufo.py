"""Intercomponent UFO autoencoder and training protocol"""
from dataclasses import dataclass
import torch
import torch.nn as nn
from .config import(
    BOTTLENECK_DIM,
    D_MLP,
    INTERCOMPONENT_ATTENTION_HIDDEN_DIM,
    INTERCOMPONENT_MLP_HIDDEN_DIM,
    N_LAYERS,
    MAX_CONTEXT_LENGTH,
    N_HEADS,
    NORMALIZATION_EPSILON,
    UFO_LEARNING_RATE,
    ExperimentConfig,
    Modality,
)
from .data import readout_position
from .model import TransformerOutput
from .selectivity import scheduled_fraction, lowest_fraction_per_group

METADATA_DIM = N_LAYERS
MLP_SIGNAL_DIM = D_MLP
NUM_ATTN_WEIGHTS_PER_HEAD_AT_EQUALS = MAX_CONTEXT_LENGTH - 1 # Inputs are [<|eos|>, x, /, y, =, z]; the "=" query excludes the answer z
ATTENTION_SIGNAL_DIM = NUM_ATTN_WEIGHTS_PER_HEAD_AT_EQUALS * N_HEADS

class IntercomponentAutoencoder(nn.Module):
    def __init__(self, signal_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(METADATA_DIM + signal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, BOTTLENECK_DIM)
        )
        self.decoder = nn.Sequential(
            nn.Linear(BOTTLENECK_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, signal_dim)
        )

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(rows))

def build_autoencoder(modality: Modality, seed: int, device: torch.device) -> IntercomponentAutoencoder:
    match modality:
        case "mlp":
            signal_dim = MLP_SIGNAL_DIM
            hidden_dim = INTERCOMPONENT_MLP_HIDDEN_DIM
        case "attention":
            signal_dim = ATTENTION_SIGNAL_DIM
            hidden_dim = INTERCOMPONENT_ATTENTION_HIDDEN_DIM
        case _:
            raise ValueError(f"Invalid modality: {modality}")
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return IntercomponentAutoencoder(signal_dim, hidden_dim,).to(device)

def representation_rows(output: TransformerOutput, tokens: torch.Tensor, modality: Modality) -> torch.Tensor:
    position = readout_position(tokens)
    rows = []
    for layer, representation in enumerate(output.layer_representations):
        match modality:
            case "mlp":
                signal = representation[:, position]
            case "attention":
                signal = representation[:, :, position, : position + 1].reshape(representation.shape[0], -1)
            case _:
                raise ValueError(f"Invalid modality: {modality}")
        metadata = signal.new_zeros((signal.shape[0], METADATA_DIM))
        metadata[:, layer] = 1.0
        rows.append(torch.cat((metadata, signal), dim=1))
    return torch.cat(rows)

def normalize_rows(rows: torch.Tensor) -> torch.Tensor:
    metadata = rows[:, :METADATA_DIM]
    signal = rows[:, METADATA_DIM:]
    layer_ids = metadata.argmax(dim=1)
    means = []
    standard_deviations = []
    for layer in range(N_LAYERS):
        values = signal[layer_ids == layer]
        means.append(values.mean(dim=0))
        standard_deviations.append(values.std(dim=0, unbiased=False).clamp_min(NORMALIZATION_EPSILON))
    mean = torch.stack(means).index_select(0, layer_ids)
    std = torch.stack(standard_deviations).index_select(0, layer_ids)
    normalized_signal = (signal - mean) / std
    return torch.cat((metadata, normalized_signal), dim=1)

@dataclass(frozen=True)
class Reconstruction:
    loss: torch.Tensor
    normalized_rows: torch.Tensor

def reconstruction(autoencoder: IntercomponentAutoencoder, rows: torch.Tensor) -> Reconstruction:
    normalized_rows = normalize_rows(rows)
    target = normalized_rows[:, METADATA_DIM:]
    decoded = autoencoder(normalized_rows)
    loss = (target - decoded).square().mean()
    return Reconstruction(loss, normalized_rows)

def row_reconstruction_losses(autoencoder: IntercomponentAutoencoder, normalized_rows: torch.Tensor) -> torch.Tensor:
    target = normalized_rows[:, METADATA_DIM:]
    decoded = autoencoder(normalized_rows)
    return (target - decoded).square().mean(dim=1)

class IntercomponentUFO:
    def __init__(self, config: ExperimentConfig, device: torch.device) -> None:
        assert config.modality is not None
        assert config.ufo_seed is not None
        self.config = config
        self.autoencoder = build_autoencoder(config.modality, config.ufo_seed, device)
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=UFO_LEARNING_RATE, weight_decay=0.0)
        self.completed_updates = 0

    def transformer_reconstruction(self, output: TransformerOutput, tokens: torch.Tensor) -> Reconstruction:
        self.autoencoder.requires_grad_(False)
        rows = representation_rows(output, tokens, self.config.modality)
        return reconstruction(self.autoencoder, rows)

    def update(self, normalized_rows: torch.Tensor) -> torch.Tensor | None:
        self.autoencoder.requires_grad_(True)
        if not self.config.ufo_updates_enabled:
            return None
        self.optimizer.zero_grad(set_to_none=True)
        row_losses = row_reconstruction_losses(self.autoencoder,normalized_rows.detach())
        if self.config.schedule is None:
            accepted = torch.ones_like(row_losses, dtype=torch.bool)
        else:
            fraction = scheduled_fraction(self.config.schedule, self.completed_updates)
            grouped_losses = row_losses.reshape(N_LAYERS, -1)
            accepted = lowest_fraction_per_group(grouped_losses, fraction).reshape_as(row_losses)
        loss = row_losses.masked_select(accepted).mean()
        loss.backward()
        self.optimizer.step()
        self.completed_updates += 1
        return loss.detach()
