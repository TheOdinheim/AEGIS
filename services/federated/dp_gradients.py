"""
DP Gradient Protection — Differential Privacy for Model Updates.

Clips gradient L2 norms and adds calibrated Gaussian noise before sharing.
Tracks cumulative privacy budget across training rounds.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_noise_scale(
    epsilon: float,
    delta: float,
    sensitivity: float,
    num_samples: int,
) -> float:
    """Compute Gaussian noise sigma for (epsilon, delta)-DP.

    Uses the Gaussian mechanism:
        sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon

    The per-sample sensitivity is sensitivity / num_samples.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0, 1)")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    per_sample_sensitivity = sensitivity / num_samples
    return per_sample_sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


def clip_gradients(
    gradients: list[np.ndarray],
    max_norm: float = 1.0,
) -> list[np.ndarray]:
    """Clip L2 norm of each gradient array.

    If the L2 norm of any gradient exceeds max_norm, it is scaled down.
    """
    clipped = []
    for g in gradients:
        norm = float(np.linalg.norm(g))
        if norm > max_norm:
            clipped.append(g * (max_norm / norm))
        else:
            clipped.append(g.copy())
    return clipped


def add_noise_to_gradients(
    gradients: list[np.ndarray],
    noise_scale: float,
    epsilon: float = 3.0,
    delta: float = 1e-5,
) -> list[np.ndarray]:
    """Add calibrated Gaussian noise to gradients.

    Args:
        gradients: List of gradient arrays.
        noise_scale: Standard deviation of Gaussian noise (sigma).
        epsilon: Privacy parameter (for reference, not recomputed).
        delta: Privacy parameter (for reference, not recomputed).

    Returns:
        List of noised gradient arrays.
    """
    noised = []
    for g in gradients:
        noise = np.random.normal(0.0, noise_scale, size=g.shape)
        noised.append(g + noise)
    return noised


class GradientPrivacyTracker:
    """Tracks cumulative privacy budget across training rounds.

    Uses simple composition: epsilons add across rounds.
    """

    def __init__(
        self,
        *,
        epsilon_per_round: float = 3.0,
        delta: float = 1e-5,
        total_budget: float = 100.0,
        sensitivity: float = 1.0,
    ) -> None:
        self._epsilon_per_round = epsilon_per_round
        self._delta = delta
        self._total_budget = total_budget
        self._sensitivity = sensitivity
        self._spent_epsilon: float = 0.0
        self._rounds_tracked: int = 0

    @property
    def is_budget_exhausted(self) -> bool:
        return self._spent_epsilon >= self._total_budget

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self._total_budget - self._spent_epsilon)

    def protect_gradients(
        self,
        gradients: list[np.ndarray],
        num_samples: int,
        max_norm: float = 1.0,
    ) -> list[np.ndarray]:
        """Clip and add DP noise to gradients. Tracks budget.

        Raises RuntimeError if budget is exhausted.
        """
        if self.is_budget_exhausted:
            raise RuntimeError("Privacy budget exhausted")

        clipped = clip_gradients(gradients, max_norm=max_norm)
        sigma = compute_noise_scale(
            epsilon=self._epsilon_per_round,
            delta=self._delta,
            sensitivity=self._sensitivity,
            num_samples=max(num_samples, 1),
        )
        noised = add_noise_to_gradients(clipped, noise_scale=sigma)

        self._spent_epsilon += self._epsilon_per_round
        self._rounds_tracked += 1

        logger.debug(
            "DP gradient protection: sigma=%.6f, budget=%.2f/%.2f",
            sigma, self._spent_epsilon, self._total_budget,
        )
        return noised

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "epsilon_per_round": self._epsilon_per_round,
            "total_budget": self._total_budget,
            "spent_epsilon": round(self._spent_epsilon, 4),
            "remaining_budget": round(self.remaining_budget, 4),
            "rounds_tracked": self._rounds_tracked,
            "is_exhausted": self.is_budget_exhausted,
        }
