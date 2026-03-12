"""
Vault Backup and Restore — Operational Resilience.

Biological Analog: Bone marrow stores hematopoietic stem cells that can
regenerate the entire immune system if needed. Similarly, vault backups
preserve the complete immune memory (threat indicators and detection
signatures) for disaster recovery.

Exports threat indicators and signatures as portable JSON. Embeddings are
NOT exported — they are rebuilt from indicator text on restore.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BackupResult:
    """Result of a vault backup operation."""
    success: bool
    output_path: str = ""
    indicator_count: int = 0
    signature_count: int = 0
    file_size_bytes: int = 0
    duration_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output_path": self.output_path,
            "indicator_count": self.indicator_count,
            "signature_count": self.signature_count,
            "file_size_bytes": self.file_size_bytes,
            "duration_ms": round(self.duration_ms, 2),
            "warnings": self.warnings,
        }


@dataclass
class RestoreResult:
    """Result of a vault restore operation."""
    success: bool
    indicators_restored: int = 0
    indicators_skipped: int = 0
    signatures_restored: int = 0
    duration_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "indicators_restored": self.indicators_restored,
            "indicators_skipped": self.indicators_skipped,
            "signatures_restored": self.signatures_restored,
            "duration_ms": round(self.duration_ms, 2),
            "warnings": self.warnings,
        }


class VaultBackupManager:
    """Manages backup and restore of the Threat Vault and Signature Store.

    Exports indicators and signatures as a portable JSON file.
    Embeddings are NOT included — they are rebuilt on restore from
    the indicator payload text using the embedding model.
    """

    AEGIS_VERSION = "0.1.0"

    async def backup_vault(
        self,
        vault: Any,
        output_path: Path,
        signature_store: Any | None = None,
    ) -> BackupResult:
        """Export vault indicators and signatures to a JSON file.

        Args:
            vault: ThreatVault instance.
            output_path: Destination file path.
            signature_store: Optional SignatureStore instance.

        Returns:
            BackupResult with counts and file size.
        """
        start = time.perf_counter()
        warnings: list[str] = []

        try:
            # Export indicators
            indicators_data = []
            indicators = vault.get_indicators()
            for ind in indicators:
                indicators_data.append({
                    "indicator_id": ind.indicator_id,
                    "source": ind.source.value if hasattr(ind.source, "value") else str(ind.source),
                    "threat_category": ind.threat_category.value if hasattr(ind.threat_category, "value") else str(ind.threat_category),
                    "confidence": ind.confidence,
                    "severity": ind.severity,
                    "payload_hash": ind.payload_hash,
                    "payload_summary": ind.payload_summary,
                    "confirmed": ind.confirmed,
                    "phase": ind.phase.value if hasattr(ind.phase, "value") else str(ind.phase),
                    "frequency": ind.frequency,
                    "seen_by_tenants": ind.seen_by_tenants,
                    "first_seen": ind.first_seen.isoformat() if ind.first_seen else None,
                    "last_seen": ind.last_seen.isoformat() if ind.last_seen else None,
                    "mitre_atlas_id": ind.mitre_atlas_id,
                    "affected_models": ind.affected_models,
                    "metadata": ind.metadata,
                })

            # Export signatures
            signatures_data = []
            if signature_store:
                try:
                    sigs = signature_store.get_all(include_deprecated=True)
                    for sig in sigs:
                        signatures_data.append({
                            "signature_id": sig.signature_id,
                            "pattern": sig.pattern,
                            "category": sig.category,
                            "description": sig.description,
                            "source_indicator_id": sig.source_indicator_id,
                            "affinity_score": sig.affinity_score,
                            "deprecated": sig.deprecated,
                            "tp_rate": sig.tp_rate,
                            "fp_rate": sig.fp_rate,
                            "match_count": sig.match_count,
                            "false_positive_count": sig.false_positive_count,
                        })
                except Exception as e:
                    warnings.append(f"Signature export failed: {e}")

            # Build backup structure
            vault_stats = vault.get_stats() if hasattr(vault, "get_stats") else {}
            backup = {
                "metadata": {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "aegis_version": self.AEGIS_VERSION,
                    "indicator_count": len(indicators_data),
                    "signature_count": len(signatures_data),
                    "vault_stats": vault_stats,
                },
                "indicators": indicators_data,
                "signatures": signatures_data,
            }

            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            content = json.dumps(backup, indent=2, default=str)
            output_path.write_text(content, encoding="utf-8")
            file_size = output_path.stat().st_size

            duration = (time.perf_counter() - start) * 1000

            logger.info(
                "Vault backup complete: %d indicators, %d signatures → %s (%.1f KB)",
                len(indicators_data), len(signatures_data),
                output_path, file_size / 1024,
            )

            return BackupResult(
                success=True,
                output_path=str(output_path),
                indicator_count=len(indicators_data),
                signature_count=len(signatures_data),
                file_size_bytes=file_size,
                duration_ms=duration,
                warnings=warnings,
            )

        except PermissionError as e:
            return BackupResult(
                success=False,
                warnings=[f"Permission denied: {e}"],
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        except OSError as e:
            return BackupResult(
                success=False,
                warnings=[f"I/O error: {e}"],
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as e:
            logger.error("Vault backup failed: %s", e)
            return BackupResult(
                success=False,
                warnings=[f"Backup failed: {e}"],
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    async def restore_vault(
        self,
        vault: Any,
        backup_path: Path,
        signature_store: Any | None = None,
    ) -> RestoreResult:
        """Restore vault from a JSON backup file.

        Args:
            vault: ThreatVault instance to restore into.
            backup_path: Path to backup JSON file.
            signature_store: Optional SignatureStore for signature restore.

        Returns:
            RestoreResult with counts and warnings.
        """
        start = time.perf_counter()
        warnings: list[str] = []

        try:
            # Load and validate backup file
            if not backup_path.exists():
                return RestoreResult(
                    success=False,
                    warnings=[f"Backup file not found: {backup_path}"],
                    duration_ms=(time.perf_counter() - start) * 1000,
                )

            content = backup_path.read_text(encoding="utf-8")
            if not content.strip():
                return RestoreResult(
                    success=False,
                    warnings=["Backup file is empty"],
                    duration_ms=(time.perf_counter() - start) * 1000,
                )

            try:
                backup = json.loads(content)
            except json.JSONDecodeError as e:
                return RestoreResult(
                    success=False,
                    warnings=[f"Invalid JSON: {e}"],
                    duration_ms=(time.perf_counter() - start) * 1000,
                )

            # Validate structure
            for key in ("metadata", "indicators", "signatures"):
                if key not in backup:
                    return RestoreResult(
                        success=False,
                        warnings=[f"Missing required key: '{key}'"],
                        duration_ms=(time.perf_counter() - start) * 1000,
                    )

            # Restore indicators
            from aegis.models.threat_indicator import (
                ThreatIndicator, IndicatorSource, MemoryPhase, ThreatCategory,
            )

            indicators_restored = 0
            indicators_skipped = 0

            for ind_data in backup["indicators"]:
                try:
                    # Parse enums
                    source = IndicatorSource(ind_data.get("source", "seed"))
                    phase = MemoryPhase(ind_data.get("phase", "acute"))

                    # Parse threat category
                    cat_val = ind_data.get("threat_category", "PROMPT_INJECTION")
                    try:
                        category = ThreatCategory(cat_val)
                    except ValueError:
                        category = ThreatCategory.PROMPT_INJECTION

                    # Parse dates
                    first_seen = datetime.fromisoformat(ind_data["first_seen"]) if ind_data.get("first_seen") else datetime.now(timezone.utc)
                    last_seen = datetime.fromisoformat(ind_data["last_seen"]) if ind_data.get("last_seen") else first_seen

                    # Create indicator (embedding will be empty — vault.add handles embedding)
                    indicator = ThreatIndicator(
                        indicator_id=ind_data.get("indicator_id", ""),
                        source=source,
                        threat_category=category,
                        confidence=ind_data.get("confidence", 0.5),
                        severity=ind_data.get("severity", 0.5),
                        payload_hash=ind_data.get("payload_hash", ""),
                        payload_summary=ind_data.get("payload_summary", ""),
                        confirmed=ind_data.get("confirmed", False),
                        phase=phase,
                        frequency=ind_data.get("frequency", 1),
                        seen_by_tenants=ind_data.get("seen_by_tenants", 1),
                        first_seen=first_seen,
                        last_seen=last_seen,
                        mitre_atlas_id=ind_data.get("mitre_atlas_id", ""),
                        affected_models=ind_data.get("affected_models", []),
                        metadata=ind_data.get("metadata", {}),
                        embedding=[0.0] * 384,  # Placeholder — needs re-embedding
                    )

                    # Check for duplicates by payload_hash
                    existing = vault.get_indicator(indicator.indicator_id) if hasattr(vault, "get_indicator") else None
                    if existing:
                        indicators_skipped += 1
                        continue

                    vault.add(indicator)
                    indicators_restored += 1

                except Exception as e:
                    warnings.append(f"Failed to restore indicator {ind_data.get('indicator_id', '?')}: {e}")
                    indicators_skipped += 1

            # Restore signatures
            signatures_restored = 0
            if signature_store and backup.get("signatures"):
                for sig_data in backup["signatures"]:
                    try:
                        signature_store.add(
                            pattern=sig_data.get("pattern", ""),
                            category=sig_data.get("category", "prompt_injection"),
                            description=sig_data.get("description", ""),
                            source_indicator_id=sig_data.get("source_indicator_id", ""),
                            affinity_score=sig_data.get("affinity_score", 0.0),
                            tp_rate=sig_data.get("tp_rate", 0.0),
                            fp_rate=sig_data.get("fp_rate", 0.0),
                        )
                        signatures_restored += 1
                    except Exception as e:
                        warnings.append(f"Failed to restore signature: {e}")

            duration = (time.perf_counter() - start) * 1000

            logger.info(
                "Vault restore complete: %d indicators restored, %d skipped, %d signatures",
                indicators_restored, indicators_skipped, signatures_restored,
            )

            return RestoreResult(
                success=True,
                indicators_restored=indicators_restored,
                indicators_skipped=indicators_skipped,
                signatures_restored=signatures_restored,
                duration_ms=duration,
                warnings=warnings,
            )

        except Exception as e:
            logger.error("Vault restore failed: %s", e)
            return RestoreResult(
                success=False,
                warnings=[f"Restore failed: {e}"],
                duration_ms=(time.perf_counter() - start) * 1000,
            )
