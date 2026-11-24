"""Raw-completion transports for Agents-A1-4B.

Both backends take an *already rendered* prompt (see :mod:`app.llm.template`) so that the serving
engine never applies its own chat template:

* ``OllamaBackend``  -> ``POST /api/generate`` with ``raw: true``
* ``OpenAIBackend``  -> ``POST /v1/completions`` (llama.cpp ``llama-server``, llama-cpp-python,
  vLLM, SGLang, LM Studio ...)
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from ..config import Settings
from .profile import SamplingParams

log = logging.getLogger(__name__)


class BackendError(RuntimeError):
    """The model server could not be reached or returned an error."""


@dataclass
class StreamPiece:
    text: str = ""
    done: bool = False
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class GenerationRequest:
    prompt: str
    sampling: SamplingParams
    max_tokens: int
    stop: list[str] = field(default_factory=list)
    seed: int | None = None


class LLMBackend(Protocol):
    name: str

    def stream(self, req: GenerationRequest) -> AsyncIterator[StreamPiece]: ...

    async def health(self) -> dict[str, Any]: ...

    async def aclose(self) -> None: ...


def _client(settings: Settings) -> httpx.AsyncClient:
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}
    return httpx.AsyncClient(
        base_url=settings.llm_base_url.rstrip("/"),
        headers=headers,
        timeout=httpx.Timeout(settings.llm_request_timeout, connect=10.0),
    )


class OllamaBackend:
    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.client = _client(settings)

    def _payload(self, req: GenerationRequest) -> dict[str, Any]:
        sp = req.sampling
        options: dict[str, Any] = {
            "num_ctx": self.s.llm_num_ctx,
            "num_predict": req.max_tokens,
            "temperature": sp.temperature,
            "top_p": sp.top_p,
            "top_k": sp.top_k,
            "min_p": sp.min_p,
            "repeat_penalty": sp.repeat_penalty,
            "stop": req.stop,
        }
        if req.seed is not None:
            options["seed"] = req.seed
        return {
            "model": self.s.llm_model,
            "prompt": req.prompt,
            "raw": True,  # <- bypass the package's non-native TEMPLATE
            "stream": True,
            "keep_alive": self.s.llm_keep_alive,
            "options": options,
        }

    async def stream(self, req: GenerationRequest) -> AsyncIterator[StreamPiece]:
        try:
            async with self.client.stream("POST", "/api/generate", json=self._payload(req)) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "replace")[:500]
                    raise BackendError(f"Ollama returned {resp.status_code}: {body}")
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if "error" in data:
                        raise BackendError(f"Ollama error: {data['error']}")
                    yield StreamPiece(
                        text=data.get("response", ""),
                        done=bool(data.get("done")),
                        finish_reason=data.get("done_reason"),
                        prompt_tokens=data.get("prompt_eval_count"),
                        completion_tokens=data.get("eval_count"),
                    )
        except httpx.HTTPError as exc:
            raise BackendError(f"Cannot reach Ollama at {self.s.llm_base_url}: {exc!r}") from exc

    async def health(self) -> dict[str, Any]:
        try:
            r = await self.client.get("/api/tags", timeout=5.0)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            present = any(m == self.s.llm_model or m.split(":")[0] == self.s.llm_model.split(":")[0] for m in models)
            return {"ok": True, "backend": self.name, "model": self.s.llm_model, "model_present": present, "models": models}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "backend": self.name, "error": str(exc)}

    async def aclose(self) -> None:
        await self.client.aclose()


class OpenAIBackend:
    name = "openai"

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.client = _client(settings)

    def _payload(self, req: GenerationRequest) -> dict[str, Any]:
        sp = req.sampling
        payload: dict[str, Any] = {
            "model": self.s.llm_model,
            "prompt": req.prompt,
            "stream": True,
            "max_tokens": req.max_tokens,
            "temperature": sp.temperature,
            "top_p": sp.top_p,
            "top_k": sp.top_k,
            "min_p": sp.min_p,
            "repeat_penalty": sp.repeat_penalty,
            "stop": req.stop,
            "cache_prompt": True,
            "stream_options": {"include_usage": True},
        }
        if req.seed is not None:
            payload["seed"] = req.seed
        return payload

    async def stream(self, req: GenerationRequest) -> AsyncIterator[StreamPiece]:
        try:
            async with self.client.stream("POST", "/v1/completions", json=self._payload(req)) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "replace")[:500]
                    raise BackendError(f"Model server returned {resp.status_code}: {body}")
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    obj = json.loads(data)
                    usage = obj.get("usage") or {}
                    choices = obj.get("choices") or []
                    text, finish = "", None
                    if choices:
                        text = choices[0].get("text") or ""
                        finish = choices[0].get("finish_reason")
                    yield StreamPiece(
                        text=text,
                        done=finish is not None,
                        finish_reason=finish,
                        prompt_tokens=usage.get("prompt_tokens"),
                        completion_tokens=usage.get("completion_tokens"),
                    )
        except httpx.HTTPError as exc:
            raise BackendError(f"Cannot reach model server at {self.s.llm_base_url}: {exc!r}") from exc

    async def health(self) -> dict[str, Any]:
        try:
            r = await self.client.get("/v1/models", timeout=5.0)
            r.raise_for_status()
            models = [m.get("id") for m in r.json().get("data", [])]
            return {"ok": True, "backend": self.name, "model": self.s.llm_model, "model_present": True, "models": models}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "backend": self.name, "error": str(exc)}

    async def aclose(self) -> None:
        await self.client.aclose()


def make_backend(settings: Settings) -> LLMBackend:
    if settings.llm_backend == "ollama":
        return OllamaBackend(settings)
    return OpenAIBackend(settings)
