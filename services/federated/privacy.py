"""
Differential Privacy Engine — Privacy-Preserving Federated Intelligence.

Biological Analog: The immune system shares pathogen information between cells
via cytokines and antigen presentation, but never exposes the infected cell's
internal state. Similarly, differential privacy adds calibrated noise to shared
data, ensuring no individual training example can be reverse-engineered from
the shared model updates or threat indicators.

Implements the Gaussian mechanism for (ε, δ)-differential privacy:
    σ = sensitivity × √(2 × ln(1.25 / δ)) / ε

Privacy budget tracking prevents unbounded information leakage over time.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PrivacyBudgetRecord:
    """A single privacy budget expenditure."""
    epsilon_spent: float
    delta_spent: float
    operation: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PrivacyBudgetStatus:
    """Current privacy budget status."""
    total_epsilon: float
    total_delta: float
    spent_epsilon: float
    spent_delta: float
    remaining_epsilon: float
    remaining_delta: float
    num_operations: int
    is_exhausted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_epsilon": self.total_epsilon,
            "total_delta": self.total_delta,
            "spent_epsilon": round(self.spent_epsilon, 6),
            "spent_delta": self.spent_delta,
            "remaining_epsilon": round(self.remaining_epsilon, 6),
            "remaining_delta": self.remaining_delta,
            "num_operations": self.num_operations,
            "is_exhausted": self.is_exhausted,
        }


class DifferentialPrivacyEngine:
    """Gaussian mechanism for (ε, δ)-differential privacy.

    Adds calibrated Gaussian noise to numerical data (model weights, embeddings)
    before sharing. Tracks cumulative privacy budget to prevent unbounded leakage.

    Parameters:
        epsilon: Privacy parameter (lower = more private). Default 3.0.
        delta: Probability of privacy breach. Default 1e-5.
        total_budget: Maximum cumulative epsilon before operations are refused.
        sensitivity: L2 sensitivity of the query/data. Default 1.0.
    """

    def __init__(
        self,
        *,
        epsilon: float = 3.0,
        delta: float = 1e-5,
        total_budget: float = 100.0,
        sensitivity: float = 1.0,
    ) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if delta <= 0 or delta >= 1:
            raise ValueError("delta must be in (0, 1)")
        if total_budget <= 0:
            raise ValueError("total_budget must be positive")
        if sensitivity <= 0:
            raise ValueError("sensitivity must be positive")

        self._epsilon = epsilon
        self._delta = delta
        self._total_budget = total_budget
        self._sensitivity = sensitivity
        self._spent_epsilon: float = 0.0
        self._spent_delta: float = 0.0
        self._history: list[PrivacyBudgetRecord] = []
        self._sigma = self._compute_sigma()

    def _compute_sigma(self) -> float:
        """Compute Gaussian noise scale: σ = sensitivity × √(2 × ln(1.25/δ)) / ε."""
        return self._sensitivity * math.sqrt(2.0 * math.log(1.25 / self._delta)) / self._epsilon

    @property
    def sigma(self) -> float:
        """Current noise standard deviation."""
        return self._sigma

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def delta(self) -> float:
        return self._delta

    @property
    def is_budget_exhausted(self) -> bool:
        """Check if privacy budget is exhausted."""
        return self._spent_epsilon >= self._total_budget

    def add_noise(
        self,
        data: np.ndarray,
        *,
        operation: str = "generic",
        epsilon_override: float | None = None,
    ) -> np.ndarray:
        """Add calibrated Gaussian noise to data.

        Args:
            data: NumPy array to privatize.
            operation: Description for budget tracking.
            epsilon_override: Use a different epsilon for this operation.

        Returns:
            Noised copy of data.

        Raises:
            RuntimeError: If privacy budget is exhausted.
        """
        if self.is_budget_exhausted:
            raise RuntimeError(
                f"Privacy budget exhausted: spent {self._spent_epsilon:.4f} "
                f"of {self._total_budget:.4f} total epsilon"
            )

        eps = epsilon_override if epsilon_override is not None else self._epsilon
        if eps <= 0:
            raise ValueError("epsilon must be positive")

        # Compute sigma for this operation
        sigma = self._sensitivity * math.sqrt(2.0 * math.log(1.25 / self._delta)) / eps

        # Generate noise
        noise = np.random.normal(0.0, sigma, size=data.shape)
        noised = data + noise

        # Track budget (simple composition: epsilons add)
        self._spent_epsilon += eps
        self._spent_delta += self._delta
        self._history.append(PrivacyBudgetRecord(
            epsilon_spent=eps,
            delta_spent=self._delta,
            operation=operation,
        ))

        logger.debug(
            "DP noise added (op=%s, σ=%.4f, ε=%.4f, budget=%.4f/%.4f)",
            operation, sigma, eps, self._spent_epsilon, self._total_budget,
        )

        return noised

    def add_noise_to_weights(self, weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Add DP noise to model weight dictionary.

        Args:
            weights: Dict mapping layer names to weight arrays.

        Returns:
            Dict with noised weight arrays.
        """
        noised = {}
        for name, w in weights.items():
            noised[name] = self.add_noise(w, operation=f"weight_{name}")
        return noised

    def add_noise_to_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Add DP noise to a single embedding vector.

        Args:
            embedding: 1D embedding vector.

        Returns:
            Noised embedding.
        """
        return self.add_noise(embedding, operation="embedding")

    def get_budget_status(self) -> PrivacyBudgetStatus:
        """Get current privacy budget status."""
        remaining_eps = max(0.0, self._total_budget - self._spent_epsilon)
        return PrivacyBudgetStatus(
            total_epsilon=self._total_budget,
            total_delta=self._delta,
            spent_epsilon=self._spent_epsilon,
            spent_delta=self._spent_delta,
            remaining_epsilon=remaining_eps,
            remaining_delta=self._delta,  # delta per operation stays constant
            num_operations=len(self._history),
            is_exhausted=self.is_budget_exhausted,
        )

    def reset_budget(self) -> None:
        """Reset privacy budget (e.g., at start of new accounting period)."""
        self._spent_epsilon = 0.0
        self._spent_delta = 0.0
        self._history.clear()
        logger.info("Privacy budget reset")

    @property
    def stats(self) -> dict[str, Any]:
        """Engine statistics."""
        return {
            "epsilon": self._epsilon,
            "delta": self._delta,
            "sigma": round(self._sigma, 6),
            "sensitivity": self._sensitivity,
            "total_budget": self._total_budget,
            "spent_epsilon": round(self._spent_epsilon, 6),
            "remaining_epsilon": round(max(0, self._total_budget - self._spent_epsilon), 6),
            "num_operations": len(self._history),
            "is_exhausted": self.is_budget_exhausted,
        }
