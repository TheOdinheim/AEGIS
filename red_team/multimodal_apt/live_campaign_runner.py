"""
Live Campaign Runner — sends real HTTP requests to a live AEGIS instance.

Orchestrates Campaigns 13-19 (PRISM, DEEP COVER, WHISPER, CHIMERA,
PUPPET MASTER, ADAPTATION, TOTAL WAR) with full result tracking.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

from red_team.multimodal_apt import (
    AttackPayload,
    AttackResult,
    CampaignResult,
    Modality,
    MultimodalAPTAssessment,
    _grade,
)
from red_team.multimodal_apt.adaptive_attacker import AdaptiveMultimodalAttacker
from red_team.multimodal_apt.cross_modal_attacks import build_request_body
from red_team.multimodal_apt.model_behavior_validator import ModelBehaviorValidator
from red_team.multimodal_apt.payload_factory import PayloadFactory

logger = logging.getLogger(__name__)


class LiveCampaignRunner:
    """Sends real HTTP requests to a live AEGIS instance.

    Each campaign sends attacks via httpx, records full responses,
    and computes per-modality and per-technique detection rates.
    """

    def __init__(
        self,
        aegis_url: str = "http://localhost:8000",
        api_key: str = "",
        timeout: float = 30.0,
        concurrency: int = 5,
        model: str = "gpt-4",
    ) -> None:
        self._url = aegis_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._concurrency = concurrency
        self._model = model
        self._factory = PayloadFactory()

    async def send_attack(self, attack: AttackPayload) -> AttackResult:
        """Send a single attack to AEGIS and record the result."""
        body = build_request_body(attack, self._model)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._url}/v1/chat/completions",
                    json=body,
                    headers=headers,
                )
            elapsed = (time.monotonic() - start) * 1000
            return self._classify(attack, resp, elapsed)
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            return AttackResult(
                attack_id=attack.attack_id,
                status_code=0,
                response_body=str(e),
                latency_ms=elapsed,
                was_blocked=False,
                error=str(e),
            )

    async def send_attacks(
        self,
        attacks: list[AttackPayload],
    ) -> list[AttackResult]:
        """Send multiple attacks with concurrency control."""
        sem = asyncio.Semaphore(self._concurrency)

        async def _send(atk: AttackPayload) -> AttackResult:
            async with sem:
                return await self.send_attack(atk)

        return await asyncio.gather(*[_send(a) for a in attacks])

    def send_attacks_sync(self, attacks: list[AttackPayload]) -> list[AttackResult]:
        """Synchronous wrapper for send_attacks (for testing)."""
        # For mock/test use - caller provides mock results
        raise NotImplementedError("Use send_attacks for live campaigns or mock in tests")

    # ── Campaign 13: PRISM ──

    async def run_prism(self) -> CampaignResult:
        """Campaign 13 — PRISM: Image modality attacks."""
        attacks = self._factory.image_attacks()
        return await self._run_campaign("PRISM", attacks)

    # ── Campaign 14: DEEP COVER ──

    async def run_deep_cover(self) -> CampaignResult:
        """Campaign 14 — DEEP COVER: Document hidden content."""
        attacks = self._factory.document_attacks()
        return await self._run_campaign("DEEP COVER", attacks)

    # ── Campaign 15: WHISPER ──

    async def run_whisper(self) -> CampaignResult:
        """Campaign 15 — WHISPER: Audio adversarial."""
        attacks = self._factory.audio_attacks()
        return await self._run_campaign("WHISPER", attacks)

    # ── Campaign 16: CHIMERA ──

    async def run_chimera(self) -> CampaignResult:
        """Campaign 16 — CHIMERA: Cross-modal convergence."""
        attacks = self._factory.cross_modal_attacks()
        return await self._run_campaign("CHIMERA", attacks)

    # ── Campaign 17: PUPPET MASTER ──

    async def run_puppet_master(self) -> CampaignResult:
        """Campaign 17 — PUPPET MASTER: Tool exploitation."""
        attacks = self._factory.tool_attacks()
        return await self._run_campaign("PUPPET MASTER", attacks)

    # ── Campaign 18: ADAPTATION ──

    async def run_adaptation(self, rounds: int = 3) -> CampaignResult:
        """Campaign 18 — ADAPTATION: Adaptive attacker rounds."""
        # Start with the attacks most likely to evade
        all_attacks = self._factory.all_attacks()
        # Pick top 20 by difficulty
        sorted_attacks = sorted(all_attacks, key=lambda a: a.difficulty, reverse=True)
        initial = sorted_attacks[:20]

        start = time.monotonic()
        results = await self.send_attacks(initial)

        attacker = AdaptiveMultimodalAttacker()

        async def _send_fn(attacks: list[AttackPayload]) -> list[AttackResult]:
            return await self.send_attacks(attacks)

        # Run adaptive rounds using async
        report = attacker.run_adaptive_rounds(
            initial, results,
            send_fn=lambda atks: asyncio.get_event_loop().run_until_complete(self.send_attacks(atks)),
            rounds=rounds,
        )

        elapsed = time.monotonic() - start
        all_results = results  # initial results

        blocked = sum(1 for r in all_results if r.was_blocked)
        total = len(all_results)

        return CampaignResult(
            campaign_name="ADAPTATION",
            total_attacks=total,
            blocked=blocked,
            allowed=total - blocked,
            detection_rate=blocked / total if total else 0,
            duration_seconds=elapsed,
            narrative=report.adaptation_narrative,
            results=all_results,
        )

    # ── Campaign 19: TOTAL WAR ──

    async def run_total_war(self) -> CampaignResult:
        """Campaign 19 — TOTAL WAR: Full spectrum integration."""
        start = time.monotonic()
        all_results: list[AttackResult] = []
        benign_count = 0
        attack_count = 0

        # 80 benign text-only
        benign_texts = self._factory.benign_text_prompts(80)
        benign_attacks = [
            AttackPayload(
                attack_id=f"benign_text_{i:03d}",
                modality=Modality.TEXT,
                text_content=t,
                technique="benign",
                difficulty=0,
                metadata={"is_benign": True},
            )
            for i, t in enumerate(benign_texts)
        ]

        # 20 benign text + benign image
        benign_img = self._factory.benign_image()
        benign_img_attacks = [
            AttackPayload(
                attack_id=f"benign_img_{i:03d}",
                modality=Modality.IMAGE,
                payload_bytes=benign_img,
                text_content=self._factory.benign_text_prompts(20)[i],
                technique="benign",
                difficulty=0,
                metadata={"is_benign": True},
            )
            for i in range(20)
        ]

        # 10 benign text + benign document
        benign_doc = self._factory.benign_document()
        benign_doc_attacks = [
            AttackPayload(
                attack_id=f"benign_doc_{i:03d}",
                modality=Modality.DOCUMENT,
                payload_bytes=benign_doc,
                text_content=f"Summarize this report section {i+1}",
                technique="benign",
                difficulty=0,
                metadata={"is_benign": True},
            )
            for i in range(10)
        ]

        benign_total = len(benign_attacks) + len(benign_img_attacks) + len(benign_doc_attacks)

        # Attack payloads
        image_attacks = self._factory.image_attacks()
        doc_attacks = self._factory.document_attacks()
        audio_attacks = self._factory.audio_attacks()
        xm_attacks = self._factory.cross_modal_attacks()
        tool_attacks = self._factory.tool_attacks()

        all_payloads = (
            benign_attacks + benign_img_attacks + benign_doc_attacks
            + image_attacks + doc_attacks + audio_attacks
            + xm_attacks + tool_attacks
        )

        all_results = await self.send_attacks(all_payloads)

        # Count false positives (benign traffic that was blocked)
        benign_ids = {a.attack_id for a in benign_attacks + benign_img_attacks + benign_doc_attacks}
        attack_ids = {a.attack_id for a in image_attacks + doc_attacks + audio_attacks + xm_attacks + tool_attacks}

        false_positives = sum(1 for r in all_results if r.attack_id in benign_ids and r.was_blocked)
        true_positives = sum(1 for r in all_results if r.attack_id in attack_ids and r.was_blocked)
        false_negatives = sum(1 for r in all_results if r.attack_id in attack_ids and not r.was_blocked)

        total_attacks = len(attack_ids)
        detection_rate = true_positives / total_attacks if total_attacks else 0
        fpr = false_positives / benign_total if benign_total else 0

        elapsed = time.monotonic() - start

        # Per-modality breakdown
        per_mod = self._per_modality_breakdown(all_payloads, all_results)
        per_tech = self._per_technique_breakdown(all_payloads, all_results)

        return CampaignResult(
            campaign_name="TOTAL WAR",
            total_attacks=len(all_payloads),
            blocked=sum(1 for r in all_results if r.was_blocked),
            allowed=sum(1 for r in all_results if not r.was_blocked),
            false_positives=false_positives,
            detection_rate=detection_rate,
            false_positive_rate=fpr,
            per_modality_breakdown=per_mod,
            per_technique_breakdown=per_tech,
            duration_seconds=elapsed,
            narrative=f"Full spectrum: {total_attacks} attacks + {benign_total} benign. Detection: {detection_rate:.1%}. FPR: {fpr:.1%}.",
            results=all_results,
        )

    # ── Helpers ──

    async def _run_campaign(
        self,
        name: str,
        attacks: list[AttackPayload],
    ) -> CampaignResult:
        """Run a single-modality campaign."""
        start = time.monotonic()
        results = await self.send_attacks(attacks)
        elapsed = time.monotonic() - start

        blocked = sum(1 for r in results if r.was_blocked)
        total = len(results)
        detection_rate = blocked / total if total else 0

        per_mod = self._per_modality_breakdown(attacks, results)
        per_tech = self._per_technique_breakdown(attacks, results)

        return CampaignResult(
            campaign_name=name,
            total_attacks=total,
            blocked=blocked,
            allowed=total - blocked,
            detection_rate=detection_rate,
            per_modality_breakdown=per_mod,
            per_technique_breakdown=per_tech,
            duration_seconds=elapsed,
            narrative=f"{name}: {blocked}/{total} blocked ({detection_rate:.1%} detection rate)",
            results=results,
        )

    def _classify(
        self,
        attack: AttackPayload,
        resp: Any,
        latency_ms: float,
    ) -> AttackResult:
        """Classify HTTP response as blocked or allowed."""
        was_blocked = resp.status_code == 403
        detection_layer = None
        confidence = 0.0
        model_output = None
        response_body = ""

        try:
            data = resp.json()
            response_body = str(data)[:500]
            if was_blocked:
                error = data.get("error", {})
                detail = error.get("message", "") if isinstance(error, dict) else str(data.get("detail", ""))
                detection_layer = self._extract_layer(detail)
                meta = error.get("metadata", {}) if isinstance(error, dict) else {}
                confidence = float(meta.get("confidence", 0.0)) if isinstance(meta, dict) else 0.0
            else:
                choices = data.get("choices", [])
                if choices:
                    model_output = choices[0].get("message", {}).get("content", "")
        except Exception:
            response_body = resp.text[:500] if hasattr(resp, "text") else ""

        return AttackResult(
            attack_id=attack.attack_id,
            status_code=resp.status_code,
            response_body=response_body,
            latency_ms=latency_ms,
            was_blocked=was_blocked,
            detection_layer=detection_layer,
            confidence=confidence,
            model_output=model_output,
        )

    @staticmethod
    def _extract_layer(detail: str) -> str | None:
        """Extract detection layer from error detail."""
        d = detail.lower()
        if "innate" in d or "regex" in d or "blocklist" in d:
            return "L2"
        if "adaptive" in d or "classifier" in d or "semantic" in d:
            return "L3"
        if "output" in d or "pii" in d:
            return "L5"
        if "policy" in d:
            return "L6"
        if "multimodal" in d or "image" in d or "document" in d or "audio" in d:
            return "multimodal"
        if "tool" in d:
            return "tool_scanner"
        if "rate" in d:
            return "L1"
        return "unknown"

    @staticmethod
    def _per_modality_breakdown(
        attacks: list[AttackPayload],
        results: list[AttackResult],
    ) -> dict[str, dict[str, int]]:
        """Compute per-modality detection breakdown."""
        breakdown: dict[str, dict[str, int]] = {}
        for attack, result in zip(attacks, results):
            mod = attack.modality.value
            if mod not in breakdown:
                breakdown[mod] = {"total": 0, "blocked": 0, "allowed": 0}
            breakdown[mod]["total"] += 1
            if result.was_blocked:
                breakdown[mod]["blocked"] += 1
            else:
                breakdown[mod]["allowed"] += 1
        return breakdown

    @staticmethod
    def _per_technique_breakdown(
        attacks: list[AttackPayload],
        results: list[AttackResult],
    ) -> dict[str, dict[str, int]]:
        """Compute per-technique detection breakdown."""
        breakdown: dict[str, dict[str, int]] = {}
        for attack, result in zip(attacks, results):
            tech = attack.technique
            if tech not in breakdown:
                breakdown[tech] = {"total": 0, "blocked": 0, "allowed": 0}
            breakdown[tech]["total"] += 1
            if result.was_blocked:
                breakdown[tech]["blocked"] += 1
            else:
                breakdown[tech]["allowed"] += 1
        return breakdown
