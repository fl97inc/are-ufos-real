"""Experiment matrix for the 144 modular-division runs."""

from collections.abc import Iterator
from functools import partial
from itertools import product

from .config import (
    MODALITIES,
    SEEDS,
    UFO_FAMILIES,
    WEIGHT_DECAYS,
    ExperimentConfig,
    Treatment,
    canonical_lambda,
    canonical_max_steps,
    canonical_schedule,
)

EXPECTED_EXPERIMENT_COUNT = 144
ABLATION_TREATMENTS: tuple[Treatment, ...] = ("updates0", "updates1")


def _experiment_cells() -> Iterator[ExperimentConfig]:
    for seed, weight_decay in product(SEEDS, WEIGHT_DECAYS):
        make_run = partial(
            ExperimentConfig,
            seed=seed,
            weight_decay=weight_decay,
            max_steps=canonical_max_steps(weight_decay),
        )
        yield make_run(treatment="baseline")
        for family, modality in product(UFO_FAMILIES, MODALITIES):
            make_ufo_run = partial(
                make_run,
                family=family,
                modality=modality,
                lambda_value=canonical_lambda(family, modality),
            )
            yield make_ufo_run(
                treatment="scheduled",
                schedule=canonical_schedule(family),
            )
            if weight_decay == 0.03:
                for treatment in ABLATION_TREATMENTS:
                    yield make_ufo_run(treatment=treatment)


def experiment_matrix() -> list[ExperimentConfig]:
    """Return all 144 experiment cells."""
    experiments = list(_experiment_cells())
    assert len(experiments) == EXPECTED_EXPERIMENT_COUNT
    return experiments
