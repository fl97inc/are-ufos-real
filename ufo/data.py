"""Data and tokens for modular-division grokking experiments.

Modular-division sequences have the form
``<|eos|> x / y = z <|eos|>``.
"""
import itertools
import math
from collections.abc import Iterator
from dataclasses import dataclass
import numpy as np
import torch
from .config import BATCH_SIZE, MODULUS, PRECOMPUTE_BATCHES, TRAIN_DATA_PERCENT

EOS_TOKEN = "<|eos|>"
EQUALS_TOKEN = "="
_OPERATOR_TOKENS = (
    "+",
    "-",
    "*",
    "/",
    "**2+",
    "**3+",
    "x**2+y**2_mod_97",
    "x**2+y**2+x*y_mod_97",
    "x**2+y**2+x*y+x_mod_97",
    "x**3+x*y_mod_97",
    "x**3+x*y**2+y_mod_97",
    "(x._value//y)if(y._value%2==1)else(x-y)_mod_97",
    "s5",
    "s5conj",
    "s5aba",
    "+*",
    "+-",
    "sort",
    "reverse",
    "copy",
)

def build_vocabulary() -> tuple[str, ...]:
    permutation_tokens = tuple(
        "".join(str(value) for value in permutation)
        for permutation in itertools.permutations(range(5))
    )
    return (
        EOS_TOKEN,
        EQUALS_TOKEN,
        *sorted(_OPERATOR_TOKENS),
        *map(str, range(MODULUS)),
        *permutation_tokens,
    )

VOCABULARY = build_vocabulary()
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCABULARY)}

def division_equations() -> list[str]:
    equations = []
    for quotient in range (MODULUS):
        for denominator in range(1, MODULUS):
            numerator = (denominator * quotient) % MODULUS
            equations.append(f"{numerator} / {denominator} = {quotient}")
    return equations

def encode_equation(equation: str) -> torch.Tensor:
    tokens = f"{EOS_TOKEN} {equation} {EOS_TOKEN}".split()
    return torch.tensor([TOKEN_TO_ID[token] for token in tokens], dtype=torch.long)

def readout_position(tokens: torch.Tensor) -> int:
    positions = torch.nonzero(tokens[0] == TOKEN_TO_ID[EQUALS_TOKEN])
    return int(positions.item())

@dataclass
class ArithmeticData:
    train: torch.Tensor
    validation: torch.Tensor

def build_data(train_percent: float = TRAIN_DATA_PERCENT, seed: int = 0) -> ArithmeticData:
    equations = division_equations()
    np.random.RandomState(seed).shuffle(equations)
    split = round(len(equations) * train_percent / 100)
    encoded = torch.stack([encode_equation(equation) for equation in equations])
    return ArithmeticData(encoded[:split], encoded[split:])

Batch = tuple[torch.Tensor, torch.Tensor]

def shift_sequences(sequences: torch.Tensor) -> Batch:
    return sequences[:, :-1], sequences[:, 1:]

class BatchStream:
    def __init__(
        self,
        sequences: torch.Tensor,
        device: torch.device,
        seed: int,
    ) -> None:
        self.data = sequences.to(device)
        self.device = device
        self.batch_size = min(BATCH_SIZE, len(sequences))
        self.batches_per_epoch = math.ceil(len(sequences) / self.batch_size)
        self.epochs_per_chunk = math.ceil(PRECOMPUTE_BATCHES / self.batches_per_epoch)
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

    def __iter__(self) -> Iterator[Batch]:
        while True:
            scores = torch.rand(
                (self.epochs_per_chunk, len(self.data)),
                device=self.device,
                generator=self.generator,
            )
            permutations = torch.argsort(scores, dim=1)
            for permutation in permutations:
                shuffled = self.data[permutation]
                for start in range(0, len(shuffled), self.batch_size):
                    yield shift_sequences(shuffled[start:start+self.batch_size])

def full_batch(sequences: torch.Tensor, device: torch.device) -> Batch:
    return shift_sequences(sequences.to(device))
