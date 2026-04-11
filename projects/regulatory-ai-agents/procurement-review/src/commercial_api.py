from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from openai import OpenAI


@dataclass(frozen=True)
class CommercialModelProfile:
    alias: str
    base_url: str
    api_key_env: str
    model: str


COMMERCIAL_MODELS: dict[str, CommercialModelProfile] = {
    "cohere": CommercialModelProfile(
        alias="cohere",
        base_url="https://api.cohere.ai/v1",
        api_key_env="COHERE_API_KEY",
        model="command-r-plus",
    ),
    "openai": CommercialModelProfile(
        alias="openai",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        model="gpt-4o",
    ),
    "anthropic": CommercialModelProfile(
        alias="anthropic",
        base_url="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
        model="claude-sonnet-4-20250514",
    ),
}


def is_commercial_model_alias(model_name: str | None) -> bool:
    return bool(model_name and model_name in COMMERCIAL_MODELS)


def get_commercial_model_profile(alias: str) -> CommercialModelProfile:
    try:
        return COMMERCIAL_MODELS[alias]
    except KeyError as exc:
        raise ValueError(
            f"Unknown commercial API model alias: {alias}. "
            f"Available: {', '.join(COMMERCIAL_MODELS)}"
        ) from exc


def require_commercial_api_key(profile: CommercialModelProfile) -> str:
    api_key = os.getenv(profile.api_key_env)
    if api_key and api_key.strip():
        return api_key.strip()
    raise ValueError(f"Missing required environment variable: {profile.api_key_env}")


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
            else:
                text = getattr(item, "text", None) or getattr(item, "content", None)
            if text:
                parts.append(str(text))
        return "\n".join(parts)
    return str(content or "")


def _message_role(message: BaseMessage) -> str:
    role = getattr(message, "type", "")
    if role in {"human", "user"}:
        return "user"
    if role in {"ai", "assistant"}:
        return "assistant"
    if role == "system":
        return "system"
    return "user"


def _messages_to_prompt(messages: list[BaseMessage]) -> str:
    role_labels = {"system": "System", "user": "User", "assistant": "Assistant"}
    chunks: list[str] = []
    for message in messages:
        role = _message_role(message)
        text = _content_to_text(message.content).strip()
        if text:
            chunks.append(f"{role_labels.get(role, 'User')}:\n{text}")
    return "\n\n".join(chunks)


def _messages_to_openai(messages: list[BaseMessage]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for message in messages:
        text = _content_to_text(message.content).strip()
        if text:
            output.append({"role": _message_role(message), "content": text})
    return output or [{"role": "user", "content": ""}]


def _messages_to_anthropic(
    messages: list[BaseMessage],
) -> tuple[str | None, list[dict[str, str]]]:
    system_chunks: list[str] = []
    output: list[dict[str, str]] = []

    for message in messages:
        role = _message_role(message)
        text = _content_to_text(message.content).strip()
        if not text:
            continue
        if role == "system":
            system_chunks.append(text)
            continue
        output.append(
            {
                "role": "assistant" if role == "assistant" else "user",
                "content": text,
            }
        )

    return "\n\n".join(system_chunks) or None, output or [{"role": "user", "content": ""}]


def extract_openai_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    content = choices[0].message.content
    return _content_to_text(content)


def extract_anthropic_text(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in payload.get("content", []):
        if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
            parts.append(str(item["text"]))
    return "\n".join(parts)


def call_openai_compatible_api(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: float,
    temperature: float,
    stop: list[str] | None = None,
) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if stop:
        request["stop"] = stop
    response = client.chat.completions.create(**request)
    return extract_openai_text(response).strip()


def call_cohere_api(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    temperature: float,
    stop: list[str] | None = None,
) -> str:
    request: dict[str, Any] = {
        "model": model,
        "message": prompt,
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if stop:
        request["stop_sequences"] = stop

    response = requests.post(
        f"{base_url.rstrip('/')}/chat",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=request,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("text"):
        return str(payload["text"]).strip()
    chat_history = payload.get("chat_history", [])
    if chat_history:
        last_message = chat_history[-1].get("message")
        if last_message:
            return str(last_message).strip()
    return ""


def call_anthropic_api(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system: str | None,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: float,
    temperature: float,
    stop: list[str] | None = None,
) -> str:
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if system:
        request["system"] = system
    if stop:
        request["stop_sequences"] = stop

    response = requests.post(
        f"{base_url.rstrip('/')}/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json=request,
        timeout=timeout,
    )
    response.raise_for_status()
    return extract_anthropic_text(response.json()).strip()


def call_commercial_llm(
    *,
    api_mode: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    temperature: float,
    stop: list[str] | None = None,
) -> str:
    profile = get_commercial_model_profile(api_mode)
    api_key = require_commercial_api_key(profile)

    if api_mode == "cohere":
        return call_cohere_api(
            base_url=profile.base_url,
            api_key=api_key,
            model=profile.model,
            prompt=prompt,
            max_tokens=max_tokens,
            timeout=timeout,
            temperature=temperature,
            stop=stop,
        )
    if api_mode == "anthropic":
        return call_anthropic_api(
            base_url=profile.base_url,
            api_key=api_key,
            model=profile.model,
            system=None,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            timeout=timeout,
            temperature=temperature,
            stop=stop,
        )

    return call_openai_compatible_api(
        base_url=profile.base_url,
        api_key=api_key,
        model=profile.model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        timeout=timeout,
        temperature=temperature,
        stop=stop,
    )


def call_commercial_chat(
    *,
    api_mode: str,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[BaseMessage],
    max_tokens: int,
    timeout: float,
    temperature: float,
    stop: list[str] | None = None,
) -> str:
    if api_mode == "cohere":
        return call_cohere_api(
            base_url=base_url,
            api_key=api_key,
            model=model,
            prompt=_messages_to_prompt(messages),
            max_tokens=max_tokens,
            timeout=timeout,
            temperature=temperature,
            stop=stop,
        )
    if api_mode == "anthropic":
        system, anthropic_messages = _messages_to_anthropic(messages)
        return call_anthropic_api(
            base_url=base_url,
            api_key=api_key,
            model=model,
            system=system,
            messages=anthropic_messages,
            max_tokens=max_tokens,
            timeout=timeout,
            temperature=temperature,
            stop=stop,
        )
    return call_openai_compatible_api(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=_messages_to_openai(messages),
        max_tokens=max_tokens,
        timeout=timeout,
        temperature=temperature,
        stop=stop,
    )


class CommercialChatModel(BaseChatModel):
    api_mode: str
    api_key: str
    base_url: str
    model: str
    request_timeout: float = 600.0
    temperature: float = 0.0
    max_tokens: int = 2048

    @property
    def _llm_type(self) -> str:
        return f"commercial-{self.api_mode}"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "api_mode": self.api_mode,
            "base_url": self.base_url,
            "model": self.model,
        }

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = call_commercial_chat(
            api_mode=self.api_mode,
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            messages=messages,
            max_tokens=int(kwargs.get("max_tokens", self.max_tokens)),
            timeout=float(kwargs.get("request_timeout", self.request_timeout)),
            temperature=float(kwargs.get("temperature", self.temperature)),
            stop=stop,
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
