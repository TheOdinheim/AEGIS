"""
Configuration Validator — Startup Safety Checks.

Biological Analog: Positive and negative selection in the thymus. Immature
T-cells are tested: those that cannot recognize MHC (positive selection fail)
are eliminated, and those that attack self-antigens (negative selection fail)
are eliminated. Only properly configured cells enter circulation.

Validates AEGIS configuration before startup. Errors prevent startup.
Warnings are logged but do not block.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from aegis.config import AegisConfig

logger = logging.getLogger(__name__)

_WEAK_API_KEYS = {"aegis-dev-key", "test", "example", "changeme", "password", "secret"}


@dataclass
class ConfigValidationResult:
    """Result of configuration validation."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "checked_at": self.checked_at.isoformat(),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }


class ConfigValidator:
    """Validates AEGIS configuration for production readiness.

    Checks are categorized as errors (fatal, prevent startup) or warnings
    (non-fatal but risky, logged and continued).
    """

    def validate(self, config: AegisConfig) -> ConfigValidationResult:
        """Run all validation checks against the configuration.

        Args:
            config: The AegisConfig to validate.

        Returns:
            ConfigValidationResult with errors and warnings.
        """
        errors: list[str] = []
        warnings: list[str] = []

        self._check_api_key(config, errors, warnings)
        self._check_canary_secret(config, warnings)
        self._check_agent_signing_key(warnings)
        self._check_thresholds(config, errors)
        self._check_rate_limits(config, errors)
        self._check_token_limit(config, errors)
        self._check_upstream_url(config, errors)
        self._check_production_database(warnings)
        self._check_production_redis(warnings)
        self._check_model_files(warnings)
        self._check_dp_epsilon(warnings)
        self._check_policy_backend(config, errors)

        result = ConfigValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

        if errors:
            for err in errors:
                logger.error("Config validation ERROR: %s", err)
        if warnings:
            for warn in warnings:
                logger.warning("Config validation WARNING: %s", warn)

        return result

    def _check_api_key(
        self, config: AegisConfig, errors: list[str], warnings: list[str],
    ) -> None:
        """AEGIS_API_KEY must be set and not a weak default."""
        if not config.api_key:
            errors.append(
                "AEGIS_API_KEY is empty — all requests will be rejected. "
                "Set a strong API key for client authentication."
            )
            return

        key_lower = config.api_key.lower()
        if any(weak in key_lower for weak in ("test", "example", "changeme")):
            warnings.append(
                f"AEGIS_API_KEY appears to be a test/example value. "
                f"Use a cryptographically random key in production."
            )
        if config.api_key in _WEAK_API_KEYS:
            warnings.append(
                "AEGIS_API_KEY matches a known weak default. Change it immediately."
            )

    def _check_canary_secret(self, config: AegisConfig, warnings: list[str]) -> None:
        """AEGIS_CANARY_SECRET_KEY should not be the default."""
        default = "aegis-canary-default-secret-change-in-production"
        if config.canary.secret_key == default:
            warnings.append(
                "AEGIS_CANARY_SECRET_KEY is still the default value. "
                "Set a unique secret for canary token integrity."
            )

    def _check_agent_signing_key(self, warnings: list[str]) -> None:
        """AEGIS_AGENT_SIGNING_KEY should be set for persistence."""
        key = os.environ.get("AEGIS_AGENT_SIGNING_KEY", "")
        if not key:
            warnings.append(
                "AEGIS_AGENT_SIGNING_KEY is not set — an ephemeral key will be "
                "auto-generated. Agent tokens will not persist across restarts."
            )

    def _check_thresholds(self, config: AegisConfig, errors: list[str]) -> None:
        """Block and alert thresholds must be in valid range."""
        bt = config.innate.block_threshold
        at = config.innate.alert_threshold
        if not (0.0 <= bt <= 1.0):
            errors.append(
                f"block_threshold ({bt}) must be in range [0.0, 1.0]."
            )
        if not (0.0 <= at <= 1.0):
            errors.append(
                f"alert_threshold ({at}) must be in range [0.0, 1.0]."
            )
        if 0.0 <= at <= 1.0 and 0.0 <= bt <= 1.0 and at >= bt:
            errors.append(
                f"alert_threshold ({at}) must be less than block_threshold ({bt})."
            )

    def _check_rate_limits(self, config: AegisConfig, errors: list[str]) -> None:
        """Rate limits must be positive."""
        if config.barrier.rate_limit_rpm <= 0:
            errors.append(
                f"rate_limit_rpm ({config.barrier.rate_limit_rpm}) must be positive."
            )
        if config.barrier.rate_limit_burst <= 0:
            errors.append(
                f"rate_limit_burst ({config.barrier.rate_limit_burst}) must be positive."
            )

    def _check_token_limit(self, config: AegisConfig, errors: list[str]) -> None:
        """Max tokens per request must be positive."""
        if config.barrier.max_tokens_per_request <= 0:
            errors.append(
                f"max_tokens_per_request ({config.barrier.max_tokens_per_request}) "
                f"must be positive."
            )

    def _check_upstream_url(self, config: AegisConfig, errors: list[str]) -> None:
        """Upstream URL must not be empty."""
        if not config.upstream_url:
            errors.append(
                "AEGIS_UPSTREAM_URL is empty — no upstream model provider configured."
            )

    def _check_production_database(self, warnings: list[str]) -> None:
        """Production should use SSL for database."""
        env = os.environ.get("AEGIS_ENV", "").lower()
        db_url = os.environ.get("DATABASE_URL", "")
        if env == "production" and db_url and "sslmode=" not in db_url:
            warnings.append(
                "DATABASE_URL does not include sslmode= parameter. "
                "Use sslmode=require or sslmode=verify-full in production."
            )

    def _check_production_redis(self, warnings: list[str]) -> None:
        """Production should have Redis for rate limiting and event bus."""
        env = os.environ.get("AEGIS_ENV", "").lower()
        redis_url = os.environ.get("REDIS_URL", "")
        if env == "production" and not redis_url:
            warnings.append(
                "REDIS_URL is not set in production. In-memory rate limiting "
                "and event bus will not persist across restarts."
            )

    def _check_model_files(self, warnings: list[str]) -> None:
        """Warn if models are expected but might not exist."""
        skip = os.environ.get("AEGIS_SKIP_MODEL_LOAD", "").lower() in ("true", "1", "yes")
        if not skip:
            # Models download on first run, so just warn
            try:
                from pathlib import Path
                cache_dir = Path.home() / ".cache" / "huggingface"
                if not cache_dir.exists():
                    warnings.append(
                        "HuggingFace cache directory not found. DeBERTa and MiniLM "
                        "models will be downloaded on first startup (~500MB)."
                    )
            except Exception:
                pass

    def _check_dp_epsilon(self, warnings: list[str]) -> None:
        """Differential privacy epsilon should be in reasonable range."""
        eps_str = os.environ.get("AEGIS_DP_EPSILON", "")
        if eps_str:
            try:
                eps = float(eps_str)
                if eps < 0.1 or eps > 100.0:
                    warnings.append(
                        f"AEGIS_DP_EPSILON ({eps}) is outside recommended range "
                        f"[0.1, 100.0]. Values < 0.1 add excessive noise; "
                        f"values > 100.0 provide minimal privacy."
                    )
            except ValueError:
                warnings.append(
                    f"AEGIS_DP_EPSILON ('{eps_str}') is not a valid number."
                )

    def _check_policy_backend(self, config: AegisConfig, errors: list[str]) -> None:
        """Policy backend must be 'python' or 'opa'."""
        if config.policy.backend not in ("python", "opa"):
            errors.append(
                f"AEGIS_POLICY_BACKEND ('{config.policy.backend}') must be "
                f"'python' or 'opa'."
            )
