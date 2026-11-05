"""Configuration for the UFO experimental setup."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

PROTOCOL_VERSION = "ufo-first-crossing-v1"

# We study UFO influence on grokking modular division with modulus 97 and
# a 50% training-data split.
Operator = Literal["/"]
MODULUS = 97
TRAIN_DATA_PERCENT = 50.0

# Grok tracking
EVALUATE_EVERY_STEPS = 1_000
GROK_ACCURACY_THRESHOLD = 99.0

# Transformer specs: 2 layers, model dimension of 128, 4 attention heads per
# layer, and MLP dimension of 512.
N_LAYERS = 2
N_HEADS = 4
D_MODEL = 128
D_MLP = 4 * D_MODEL

# Transformer training specs
PRECOMPUTE_BATCHES = 10_000
BATCH_SIZE = 512
MAX_CONTEXT_LENGTH = 6
MAX_LEARNING_RATE = 1e-3
LEARNING_RATE_WARMUP_STEPS = 10
TRANSFORMER_ADAM_BETAS = (0.9, 0.98)
TRANSFORMER_ADAM_EPSILON = 1e-8

# For each treatment, we test 8 seeds.
SEEDS = tuple(range(1, 9))

# We test treatments with and without weight decay.
WEIGHT_DECAYS = (0.03, 0.0)

# Treatments include scheduled selective updating, non-selective updating
# (updates1), non-updating UFOs (updates0), and baselines.
Modality = Literal["mlp", "attention"]
UFOFamily = Literal["intercomponent", "intracomponent"]
Treatment = Literal["baseline", "updates0", "updates1", "scheduled"]
MODALITIES: tuple[Modality, ...] = ("mlp", "attention")
UFO_FAMILIES: tuple[UFOFamily, ...] = ("intercomponent", "intracomponent")

# UFO autoencoder specs
BOTTLENECK_DIM = 2
INTERCOMPONENT_MLP_HIDDEN_DIM = 128
INTERCOMPONENT_ATTENTION_HIDDEN_DIM = 8
INTRACOMPONENT_HIDDEN_DIM = 128

# UFO training specs
NORMALIZATION_EPSILON = 1e-6
UFO_LEARNING_RATE = 1e-3

# UFO regularization warmup boundaries
LAMBDA_WARMUP_START = 250
LAMBDA_WARMUP_END = 500

# Selectivity schedules for intercomponent and intracomponent UFOs
@dataclass(frozen=True)
class SelectivitySchedule:
    start_fraction: float
    floor_fraction: float
    num_ufo_steps: int


def canonical_schedule(family: UFOFamily) -> SelectivitySchedule:
    match family:
        case "intercomponent":
            return SelectivitySchedule(
                start_fraction=0.25,
                floor_fraction=0.10,
                num_ufo_steps=2_000,
            )
        case "intracomponent":
            return SelectivitySchedule(
                start_fraction=0.50,
                floor_fraction=0.10,
                num_ufo_steps=30_000,
            )
        case _:
            raise ValueError(f"Unsupported UFO family: {family!r}")


# Steady-state UFO regularization strengths
def canonical_lambda(
    family: UFOFamily,
    modality: Modality,
) -> float:
    values = {
        ("intercomponent", "mlp"): 1.0,
        ("intercomponent", "attention"): 5.0,
        ("intracomponent", "mlp"): 1.0,
        ("intracomponent", "attention"): 3.0,
    }
    return values[(family, modality)]


# UFO regularization strength as a function of training step
def lambda_at_step(maximum: float, step: int) -> float:
    progress = (step - LAMBDA_WARMUP_START) / (
        LAMBDA_WARMUP_END - LAMBDA_WARMUP_START
    )
    return maximum * min(max(progress, 0.0), 1.0)


# Maximum steps by weight-decay condition
def canonical_max_steps(weight_decay: float) -> int:
    if weight_decay == 0.03:
        return 150_000
    if weight_decay == 0.0:
        return 4_000_000


# Derive seeds to vary transformer parameters, UFO parameters, or data shuffling.
def derive_seed(replicate_seed: int, seed_role: str) -> int:
    payload = f"open-grok/{seed_role}/{replicate_seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (
        2**63 - 1
    )


# Configuration for one experiment
@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    treatment: Treatment
    weight_decay: float
    max_steps: int
    family: UFOFamily | None = None
    modality: Modality | None = None
    lambda_value: float = 0.0
    schedule: SelectivitySchedule | None = None

    @property
    def model_seed(self) -> int:
        return self.seed

    @property
    def ufo_seed(self) -> int | None:
        if self.family is None:
            return None
        seed_role = "ae" if self.family == "intercomponent" else "intra_ae"
        return derive_seed(self.seed, seed_role)

    @property
    def batch_seed(self) -> int:
        return derive_seed(self.seed, "batch")

    @property
    def data_seed(self) -> int:
        return 0

    @property
    def ufo_updates_enabled(self) -> bool:
        return self.treatment in {"updates1", "scheduled"}
