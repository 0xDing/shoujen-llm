from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


STORED_ROLES = {"user", "assistant"}
REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}
THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
OPEN_THINK_RE = re.compile(r"<think\b[^>]*>.*", re.IGNORECASE | re.DOTALL)


def parse_models(value: str | None) -> list[str]:
    """Parse a comma-separated model list while preserving order."""
    seen: set[str] = set()
    models: list[str] = []
    for raw_item in (value or "").replace("\n", ",").split(","):
        item = raw_item.strip()
        if item and item not in seen:
            seen.add(item)
            models.append(item)
    return models


def parse_prompt_lines(value: str | None) -> list[str]:
    return [line for line in (raw_line.strip() for raw_line in (value or "").splitlines()) if line]


def render_topic_prompt(template: str, topic: str) -> str:
    clean_template = (template or "").strip()
    clean_topic = (topic or "").strip()
    if "{topic}" in clean_template:
        return clean_template.replace("{topic}", clean_topic)

    target = "本次生成的对话主题是：主题"
    if target in clean_template:
        return clean_template.replace(target, f"本次生成的对话主题是：{clean_topic}")

    return f"{clean_template}\n\n本次生成的对话主题是：{clean_topic}".strip()


def normalize_message(message: dict[str, Any]) -> dict[str, str] | None:
    role = str(message.get("role", "")).strip()
    if role not in STORED_ROLES:
        return None

    content = message.get("content", "")
    if content is None:
        content = ""

    return {"role": role, "content": str(content)}


def normalize_conversation(messages: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages or []:
        clean_message = normalize_message(message)
        if clean_message is not None:
            normalized.append(clean_message)
    return normalized


def strip_think_blocks(content: str) -> str:
    filtered = THINK_BLOCK_RE.sub("", content)
    filtered = OPEN_THINK_RE.sub("", filtered)
    if filtered == content:
        return content

    filtered = re.sub(r"[ \t]+\n", "\n", filtered)
    filtered = re.sub(r"\n{3,}", "\n\n", filtered)
    return filtered.strip()


def prepare_conversation_for_storage(
    conversation: list[dict[str, Any]] | None,
    include_think: bool = True,
) -> list[dict[str, str]]:
    messages = normalize_conversation(conversation)
    if include_think:
        return messages

    prepared: list[dict[str, str]] = []
    for message in messages:
        content = message["content"]
        if message["role"] == "assistant":
            content = strip_think_blocks(content)
        prepared.append({"role": message["role"], "content": content})
    return prepared


def remove_markdown_fences(text: str) -> str:
    lines = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def validate_strict_chatml_record(
    record: Any,
    min_turns: int = 2,
    max_turns: int = 4,
) -> list[dict[str, str]]:
    if not isinstance(record, dict):
        raise ValueError("record is not an object")

    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages is not a list")
    if len(messages) % 2 != 0:
        raise ValueError("messages must contain user/assistant pairs")

    turns = len(messages) // 2
    if turns < min_turns or turns > max_turns:
        raise ValueError(f"turn count must be between {min_turns} and {max_turns}")

    validated: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError("message is not an object")

        expected_role = "user" if index % 2 == 0 else "assistant"
        role = message.get("role")
        if role != expected_role:
            raise ValueError(f"message role must be {expected_role}")

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("message content is empty")
        validated.append({"role": role, "content": content})

    return validated


def parse_chatml_jsonl_output(
    text: str,
    min_turns: int = 2,
    max_turns: int = 4,
) -> tuple[list[list[dict[str, str]]], list[str]]:
    cleaned = remove_markdown_fences(text)
    if not cleaned:
        return [], ["empty output"]

    raw_records: list[Any] = []
    errors: list[str] = []

    if cleaned.startswith("["):
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                raw_records.extend(parsed)
            else:
                errors.append("top-level JSON is not a list")
        except json.JSONDecodeError as exc:
            errors.append(f"invalid JSON array: {exc}")

    if not raw_records:
        for line_number, line in enumerate(cleaned.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw_records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON: {exc}")

    conversations: list[list[dict[str, str]]] = []
    for index, record in enumerate(raw_records, start=1):
        try:
            conversations.append(validate_strict_chatml_record(record, min_turns=min_turns, max_turns=max_turns))
        except ValueError as exc:
            errors.append(f"record {index}: {exc}")

    return conversations, errors


def build_api_messages(
    conversation: list[dict[str, Any]] | None,
    system_prompt: str | None,
) -> list[dict[str, str]]:
    api_messages: list[dict[str, str]] = []
    clean_system_prompt = (system_prompt or "").strip()
    if clean_system_prompt:
        api_messages.append({"role": "system", "content": clean_system_prompt})

    api_messages.extend(normalize_conversation(conversation))
    return api_messages


def build_single_turn_conversation(prompt: str, assistant_content: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": assistant_content},
    ]


def is_openrouter_base_url(base_url: str | None) -> bool:
    parsed = urlparse((base_url or "").strip())
    return parsed.hostname == "openrouter.ai"


def normalize_reasoning_effort(value: str | None) -> str:
    effort = (value or "").strip().lower()
    if effort in REASONING_EFFORTS:
        return effort
    return ""


def build_openrouter_extra_body(base_url: str | None, reasoning_effort: str | None) -> dict[str, Any] | None:
    effort = normalize_reasoning_effort(reasoning_effort)
    if not effort or not is_openrouter_base_url(base_url):
        return None
    return {"reasoning": {"effort": effort}}


def parse_stop(value: str | None) -> str | list[str] | None:
    clean_value = (value or "").strip()
    if not clean_value:
        return None

    stops = [line for line in (item.strip() for item in clean_value.splitlines()) if line]
    if not stops:
        return None
    if len(stops) == 1:
        return stops[0]
    return stops


def build_completion_kwargs(
    model: str,
    messages: list[dict[str, str]],
    max_completion_tokens: int,
    temperature: float,
    top_p: float,
    stop: str | None,
    frequency_penalty: float,
    presence_penalty: float,
    base_url: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "stream": False,
        "stop": parse_stop(stop),
        "frequency_penalty": float(frequency_penalty),
        "presence_penalty": float(presence_penalty),
    }
    if int(max_completion_tokens) > 0:
        kwargs["max_completion_tokens"] = int(max_completion_tokens)

    extra_body = build_openrouter_extra_body(base_url, reasoning_effort)
    if extra_body is not None:
        kwargs["extra_body"] = extra_body
    return kwargs


def extract_assistant_content(message: Any) -> str:
    content = getattr(message, "content", None)
    reasoning_content = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
    if isinstance(reasoning_content, dict):
        reasoning_content = reasoning_content.get("content") or reasoning_content.get("summary")

    clean_content = "" if content is None else str(content)
    clean_reasoning = "" if reasoning_content is None else str(reasoning_content).strip()
    if clean_reasoning and "<think>" not in clean_content:
        return f"<think>{clean_reasoning}</think>\n{clean_content}".rstrip()
    return clean_content


def build_chatml_record(
    conversation: list[dict[str, Any]] | None,
    include_think: bool = True,
) -> dict[str, list[dict[str, str]]]:
    messages = prepare_conversation_for_storage(conversation, include_think=include_think)
    if not messages:
        raise ValueError("conversation is empty")
    if not any(message["role"] == "assistant" and message["content"].strip() for message in messages):
        raise ValueError("conversation has no assistant reply")

    return {"messages": messages}


def resolve_dataset_path(path_value: str | None, base_dir: Path) -> Path:
    raw_path = (path_value or "").strip()
    if not raw_path:
        raw_path = "datasets/sft.jsonl"

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def append_chatml_record(
    conversation: list[dict[str, Any]] | None,
    output_path: Path,
    include_think: bool = True,
) -> Path:
    record = build_chatml_record(conversation, include_think=include_think)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    return output_path
