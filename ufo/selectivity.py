"""Helpers for selective UFO updates.

Intercomponent updates select the easiest problems per layer.
Intracomponent updates select the easiest problems per component.
"""
import math
import torch
from .config import SelectivitySchedule

def scheduled_fraction(schedule: SelectivitySchedule, completed_updates: int) -> float:
    progress = min(completed_updates / schedule.num_ufo_steps, 1.0)
    return schedule.floor_fraction + (schedule.start_fraction - schedule.floor_fraction) * (1.0 - progress)

def accepted_count(group_size: int, fraction: float) -> int:
    count = math.floor(group_size * fraction + 0.5)
    return max(1, min(group_size, count))

def lowest_fraction_per_group(losses: torch.Tensor, fraction: float) -> torch.Tensor:
    count = accepted_count(losses.shape[1], fraction)
    order = torch.argsort(losses.detach(), dim=1, stable=True)
    selected = order[:, :count]
    accepted = torch.zeros_like(losses, dtype=torch.bool)
    return accepted.scatter_(1, selected, True)