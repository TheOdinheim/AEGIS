"""
Cross-modal attack message builders.

Constructs OpenAI-compatible multimodal messages that combine benign text
with malicious media content (images, documents, audio) targeting the seams
between AEGIS modality-specific scanners.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from red_team.multimodal_apt import AttackPayload, Modality
from red_team.multimodal_apt.payload_factory import PayloadFactory


def build_image_message(
    text: str,
    image_bytes: bytes,
    image_format: str = "png",
) -> list[dict[str, Any]]:
    """Build an OpenAI multimodal message with text + image."""
    b64 = base64.b64encode(image_bytes).decode()
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{image_format};base64,{b64}"},
                },
            ],
        }
    ]


def build_document_message(
    text: str,
    doc_bytes: bytes,
    mime_type: str = "text/html",
) -> list[dict[str, Any]]:
    """Build an OpenAI multimodal message with text + document."""
    b64 = base64.b64encode(doc_bytes).decode()
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "file",
                    "file": {"data": b64, "mime_type": mime_type},
                },
            ],
        }
    ]


def build_audio_message(
    text: str,
    audio_bytes: bytes,
    audio_format: str = "wav",
) -> list[dict[str, Any]]:
    """Build an OpenAI multimodal message with text + audio."""
    b64 = base64.b64encode(audio_bytes).decode()
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "input_audio",
                    "input_audio": {"data": b64, "format": audio_format},
                },
            ],
        }
    ]


def build_tool_message(
    text: str,
    tool_defs: list[dict[str, Any]] | None = None,
    tool_outputs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build an OpenAI request body with tool definitions and/or tool outputs.

    Returns a full request body dict (not just messages).
    """
    messages: list[dict[str, Any]] = [{"role": "user", "content": text}]

    if tool_outputs:
        for output in tool_outputs:
            messages.append({
                "role": "tool",
                "content": output.get("content", ""),
                "tool_call_id": output.get("tool_call_id", "call_1"),
            })

    body: dict[str, Any] = {
        "model": "gpt-4",
        "messages": messages,
    }

    if tool_defs:
        body["tools"] = tool_defs

    return body


def build_multi_image_message(
    text: str,
    images: list[bytes],
    image_format: str = "png",
) -> list[dict[str, Any]]:
    """Build message with multiple images."""
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for img in images:
        b64 = base64.b64encode(img).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/{image_format};base64,{b64}"},
        })
    return [{"role": "user", "content": content}]


def build_request_body(
    attack: AttackPayload,
    model: str = "gpt-4",
) -> dict[str, Any]:
    """Build a complete OpenAI-compatible request body from an AttackPayload.

    Routes to the appropriate message builder based on modality and metadata.
    """
    # Tool attacks
    if attack.modality == Modality.TOOL:
        tool_def = attack.metadata.get("tool_def")
        tool_output = attack.metadata.get("tool_output_content")
        tool_call = attack.metadata.get("tool_call")

        tool_defs = [tool_def] if tool_def else None

        tool_outputs = None
        if tool_output:
            tool_outputs = [{"content": tool_output, "tool_call_id": "call_1"}]
        elif tool_call:
            # Include the tool call arguments as text content
            messages = [{"role": "user", "content": f"{attack.text_content} {json.dumps(tool_call)}"}]
            body: dict[str, Any] = {"model": model, "messages": messages}
            if tool_defs:
                body["tools"] = tool_defs
            return body

        return build_tool_message(attack.text_content, tool_defs, tool_outputs)

    # Image attacks
    if attack.modality == Modality.IMAGE and attack.payload_bytes:
        fmt = "jpeg" if attack.payload_bytes[:2] == b'\xff\xd8' else "png"
        messages = build_image_message(attack.text_content or "Describe this image", attack.payload_bytes, fmt)
        return {"model": model, "messages": messages}

    # Document attacks
    if attack.modality == Modality.DOCUMENT and attack.payload_bytes:
        mime = "text/html"
        if attack.payload_bytes[:5] == b"%PDF-":
            mime = "application/pdf"
        elif attack.payload_bytes[:4] == b"PK\x03\x04":
            mime = "application/zip"
        messages = build_document_message(
            attack.text_content or "Summarize this document",
            attack.payload_bytes,
            mime,
        )
        return {"model": model, "messages": messages}

    # Audio attacks
    if attack.modality == Modality.AUDIO and attack.payload_bytes:
        messages = build_audio_message(
            attack.text_content or "Transcribe this audio",
            attack.payload_bytes,
        )
        return {"model": model, "messages": messages}

    # Cross-modal attacks — route based on metadata
    if attack.modality == Modality.CROSS_MODAL:
        return _build_cross_modal_body(attack, model)

    # Default: text-only
    return {
        "model": model,
        "messages": [{"role": "user", "content": attack.text_content}],
    }


def _build_cross_modal_body(attack: AttackPayload, model: str) -> dict[str, Any]:
    """Build cross-modal request body based on attack metadata."""
    media_type = attack.metadata.get("media_type", "image")

    if media_type == "image" and attack.payload_bytes:
        messages = build_image_message(attack.text_content, attack.payload_bytes)
        return {"model": model, "messages": messages}
    elif media_type == "document" and attack.payload_bytes:
        messages = build_document_message(attack.text_content, attack.payload_bytes)
        return {"model": model, "messages": messages}
    elif media_type == "audio" and attack.payload_bytes:
        messages = build_audio_message(attack.text_content, attack.payload_bytes)
        return {"model": model, "messages": messages}

    # Volume anomaly or tool-related cross-modal
    if attack.metadata.get("total_images"):
        factory = PayloadFactory()
        n = attack.metadata["total_images"]
        benign = factory.benign_image()
        images = [benign] * n
        if attack.payload_bytes:
            idx = min(attack.metadata.get("injection_index", n - 1), n - 1)
            images[idx] = attack.payload_bytes
        messages = build_multi_image_message(attack.text_content, images)
        return {"model": model, "messages": messages}

    if attack.metadata.get("document_count"):
        factory = PayloadFactory()
        doc = attack.payload_bytes or factory.benign_document()
        content: list[dict[str, Any]] = [{"type": "text", "text": attack.text_content}]
        for _ in range(attack.metadata["document_count"]):
            b64 = base64.b64encode(doc).decode()
            content.append({
                "type": "file",
                "file": {"data": b64, "mime_type": "text/html"},
            })
        return {"model": model, "messages": [{"role": "user", "content": content}]}

    # Tool-related cross-modal
    tool_def = attack.metadata.get("tool_description")
    if tool_def or attack.metadata.get("category") == "tool_poisoning":
        tool_defs = None
        if attack.metadata.get("tool_name"):
            tool_defs = [{
                "type": "function",
                "function": {
                    "name": attack.metadata["tool_name"],
                    "description": attack.metadata.get("tool_description", ""),
                    "parameters": {"type": "object", "properties": {}},
                },
            }]
        return build_tool_message(attack.text_content, tool_defs)

    tool_output = attack.metadata.get("tool_output_content")
    if tool_output:
        return build_tool_message(
            attack.text_content,
            tool_outputs=[{"content": tool_output, "tool_call_id": "call_1"}],
        )

    # Fallback to text + image if we have bytes
    if attack.payload_bytes:
        messages = build_image_message(attack.text_content, attack.payload_bytes)
        return {"model": model, "messages": messages}

    # Pure text fallback
    return {
        "model": model,
        "messages": [{"role": "user", "content": attack.text_content}],
    }
