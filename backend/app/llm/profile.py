"""Model profile: everything we know about Agents-A1-4B-Kimi-heretic in one place."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Settings


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.6
    top_p: float = 0.9
    top_k: int = 35
    min_p: float = 0.05
    repeat_penalty: float = 1.05

    def with_(self, **overrides) -> "SamplingParams":
        data = {**self.__dict__, **{k: v for k, v in overrides.items() if v is not None}}
        return SamplingParams(**data)


@dataclass(frozen=True)
class A1Profile:
    """Facts about the model, sourced from the Ollama package and the Hugging Face repos."""

    ollama_tag: str = "nutboy02/Agents-A1-4B-Kimi-heretic"
    hf_repo: str = "hotdogs/Agents-A1-4B-kimi-Preview-heretic-GGUF"
    base_repo: str = "InternScience/Agents-A1-4B"
    architecture: str = "qwen35 (hybrid linear + full attention), ChatML"
    parameters: str = "4.2B"
    quantization: str = "IQ4_NL"
    native_context: int = 262144
    # Sampling published in the Ollama package's params layer.
    sampling: SamplingParams = field(default_factory=SamplingParams)
    # NOTE: the Ollama package's `</thought>` stop is wrong for this model (it thinks with
    # <think>); in raw mode we only stop on the ChatML end markers and on a hallucinated
    # tool response.
    stop: tuple[str, ...] = ("<|im_end|>", "<|im_start|>", "<tool_response>")
    # Ollama's TEMPLATE for this package is not the model's trained template.
    ollama_template_is_native: bool = False


DEFAULT_PROFILE = A1Profile()


def sampling_from_settings(s: Settings) -> SamplingParams:
    return SamplingParams(
        temperature=s.llm_temperature,
        top_p=s.llm_top_p,
        top_k=s.llm_top_k,
        min_p=s.llm_min_p,
        repeat_penalty=s.llm_repeat_penalty,
    )
