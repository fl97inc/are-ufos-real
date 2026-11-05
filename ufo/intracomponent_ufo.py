""" Intracomponent UFO autoencoder and training protocol"""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import(
    BATCH_SIZE,
    BOTTLENECK_DIM,
    INTRACOMPONENT_HIDDEN_DIM,
    MODULUS,
    N_LAYERS,
    NORMALIZATION_EPSILON,
    UFO_LEARNING_RATE,
    ExperimentConfig,
    Modality,
)
from .data import readout_position, TOKEN_TO_ID
from .model import TransformerOutput
from .selectivity import accepted_count, scheduled_fraction, lowest_fraction_per_group

METADATA_DIM = N_LAYERS
DOMAIN_SIZE = MODULUS * (MODULUS - 1)
FUNCTION_CHUNK_SIZE = 512
NUMERIC_TOKEN_OFFSET = TOKEN_TO_ID["0"]

class IntracomponentAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(METADATA_DIM + DOMAIN_SIZE, INTRACOMPONENT_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(INTRACOMPONENT_HIDDEN_DIM, BOTTLENECK_DIM)
        )
        self.decoder = nn.Sequential(
            nn.Linear(BOTTLENECK_DIM, INTRACOMPONENT_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(INTRACOMPONENT_HIDDEN_DIM, DOMAIN_SIZE)
        )

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(rows))

def build_autoencoder(seed: int, device: torch.device) -> IntracomponentAutoencoder:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return IntracomponentAutoencoder().to(device)

def domain_positions(tokens: torch.Tensor, position: int) -> torch.Tensor:
    numerator = tokens[:, position - 3] - NUMERIC_TOKEN_OFFSET
    denominator = tokens[:, position - 1] - NUMERIC_TOKEN_OFFSET
    return numerator * (MODULUS - 1) + denominator - 1

@dataclass(frozen=True)
class FunctionRows:
    metadata: torch.Tensor
    responses: torch.Tensor
    domain_positions: torch.Tensor

    def slice(self, start: int, stop: int) -> "FunctionRows":
        return FunctionRows(metadata=self.metadata[start:stop], responses=self.responses[start:stop], domain_positions=self.domain_positions)

    def detach(self) -> "FunctionRows":
        return FunctionRows(metadata=self.metadata.detach(), responses=self.responses.detach(), domain_positions=self.domain_positions.detach())

def function_rows(output: TransformerOutput, tokens: torch.Tensor, modality: Modality, position: int) -> FunctionRows:
    per_layer = []
    layer_ids = []
    for layer, representation in enumerate(output.layer_representations):
        match modality:
            case "mlp":
                raw_responses = representation[:, position].transpose(0, 1)
            case "attention":
                attention = representation[:, :, position, : position + 1]
                raw_responses = attention.permute(1, 2, 0).reshape(-1, representation.shape[0])
            case _:
                raise ValueError(f"Invalid modality: {modality}")
        per_layer.append(raw_responses)
        layer_ids.append(torch.full((raw_responses.shape[0],), layer, dtype=torch.long, device=raw_responses.device))
    raw_responses = torch.cat(per_layer)
    layer_ids_to_tensor = torch.cat(layer_ids)
    means = raw_responses.mean(dim=1, keepdim=True)
    standard_deviations = raw_responses.std(dim=1, unbiased=False, keepdim=True).clamp_min(NORMALIZATION_EPSILON)
    responses = (raw_responses - means) / standard_deviations
    metadata = F.one_hot(layer_ids_to_tensor, num_classes=METADATA_DIM).to(responses.dtype)
    return FunctionRows(metadata=metadata, responses=responses, domain_positions=domain_positions(tokens, position))

def dense_inputs(rows: FunctionRows) -> torch.Tensor:
    dense = rows.responses.new_zeros((rows.responses.shape[0], DOMAIN_SIZE))
    positions = rows.domain_positions.unsqueeze(0).expand(rows.responses.shape[0], -1)
    dense = dense.scatter(1, positions, rows.responses)
    return torch.cat((rows.metadata, dense), dim=1)

def observed_reconstruction(autoencoder: IntracomponentAutoencoder, rows: FunctionRows) -> torch.Tensor:
    decoded = autoencoder(dense_inputs(rows))
    positions = rows.domain_positions.unsqueeze(0).expand(decoded.shape[0], -1)
    return decoded.gather(1, positions)

@dataclass(frozen=True)
class Reconstruction:
    loss: torch.Tensor
    loss_contributions: tuple[torch.Tensor, ...]
    function_rows: FunctionRows

class IntracomponentUFO:
    def __init__(self, config: ExperimentConfig, device: torch.device) -> None:
        assert config.modality is not None
        assert config.ufo_seed is not None
        self.config = config
        self.autoencoder = build_autoencoder(config.ufo_seed, device)
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=UFO_LEARNING_RATE, weight_decay=0.0)
        self.completed_updates = 0
        self._readout_position: int | None = None

    @staticmethod
    def eligible(batch_size: int) -> bool:
        return batch_size == BATCH_SIZE

    def transformer_reconstruction(self, output: TransformerOutput, tokens: torch.Tensor) -> Reconstruction:
        self.autoencoder.requires_grad_(False)
        if self._readout_position is None:
            self._readout_position = readout_position(tokens)
        rows = function_rows(output, tokens, self.config.modality, self._readout_position)
        total_elements = rows.responses.numel()
        contributions = []
        for start in range(0, rows.responses.shape[0], FUNCTION_CHUNK_SIZE):
            local = rows.slice(start, start + FUNCTION_CHUNK_SIZE)
            decoded = observed_reconstruction(self.autoencoder, local)
            contributions.append((decoded - local.responses).square().sum() / total_elements)
        return Reconstruction(loss=sum(contributions), loss_contributions=tuple(contributions), function_rows=rows)

    def update(self, rows: FunctionRows) -> torch.Tensor | None:
        self.autoencoder.requires_grad_(True)
        if not self.config.ufo_updates_enabled:
            return None
        rows = rows.detach()
        element_losses = rows.responses.new_empty(rows.responses.shape)
        with torch.no_grad():
            for start in range(0, rows.responses.shape[0], FUNCTION_CHUNK_SIZE):
                local = rows.slice(start, start + FUNCTION_CHUNK_SIZE)
                decoded = observed_reconstruction(self.autoencoder, local)
                element_losses[start : start + local.responses.shape[0]] = (decoded - local.responses).square()
        if self.config.schedule is None:
            accepted = torch.ones_like(element_losses, dtype=torch.bool)
            selected_count = accepted.numel()
        else:
            fraction = scheduled_fraction(self.config.schedule, self.completed_updates)
            accepted = lowest_fraction_per_group(element_losses, fraction)
            selected_count = accepted.shape[0] * accepted_count(accepted.shape[1], fraction)
        self.optimizer.zero_grad(set_to_none=True)
        loss_contributions = []
        for start in range(0, rows.responses.shape[0], FUNCTION_CHUNK_SIZE):
            local = rows.slice(start, start + FUNCTION_CHUNK_SIZE)
            decoded = observed_reconstruction(self.autoencoder, local)
            losses = (decoded - local.responses).square()
            local_mask = accepted[start : start + local.responses.shape[0]]
            contribution = losses.masked_select(local_mask).sum() / selected_count
            contribution.backward()
            loss_contributions.append(contribution.detach())
        self.optimizer.step()
        self.completed_updates += 1
        return sum(loss_contributions)
