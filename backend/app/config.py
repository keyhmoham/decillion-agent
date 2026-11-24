"""Runtime configuration (12-factor, environment driven)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # ---- service -------------------------------------------------------------------------
    app_name: str = "Decillion Agent"
    environment: Literal["dev", "prod", "test"] = "dev"
    log_level: str = "INFO"
    app_api_key: str = Field("", description="If set, every API/WS call must present it.")
    cors_origins: str = "*"

    # ---- database ------------------------------------------------------------------------
    database_url: str = "postgresql+asyncpg://agent:agent@postgres:5432/agent"

    # ---- model serving -------------------------------------------------------------------
    llm_backend: Literal["ollama", "openai"] = "ollama"
    llm_base_url: str = "http://ollama:11434"
    llm_model: str = "nutboy02/Agents-A1-4B-Kimi-heretic:latest"
    llm_api_key: str = ""
    # All requests must share the same num_ctx, otherwise Ollama reloads the model every call.
    llm_num_ctx: int = 65536
    llm_keep_alive: str = "30m"
    llm_request_timeout: float = 900.0
    llm_temperature: float = 0.6
    llm_top_p: float = 0.9
    llm_top_k: int = 35
    llm_min_p: float = 0.05
    llm_repeat_penalty: float = 1.05
    llm_max_tokens: int = 4096
    # Reasoning budget (tokens) before we force-close <think>.  4B models love to ramble.
    thinking_budget: int = 1800
    enable_thinking: bool = True
    # Prompt size at which history compaction kicks in (model fine-tuned on short traces).
    context_budget: int = 28000
    tokenizer_path: str = str(Path(__file__).resolve().parent / "llm" / "assets" / "tokenizer.json")

    # ---- agent behaviour -----------------------------------------------------------------
    max_steps: int = 40
    worker_max_steps: int = 24
    team_parallelism: int = 2
    max_task_retries: int = 1
    tool_output_limit: int = 8000
    default_mode: Literal["auto", "agent", "team"] = "auto"
    memory_recall_k: int = 5

    # ---- sandbox -------------------------------------------------------------------------
    sandbox_backend: Literal["docker", "local"] = "docker"
    sandbox_image: str = "decillion-sandbox:latest"
    sandbox_network: str = "bridge"
    sandbox_memory: str = "2g"
    sandbox_cpus: float = 2.0
    sandbox_pids_limit: int = 512
    sandbox_user: str = "agent"
    sandbox_idle_minutes: int = 45
    sandbox_command_timeout: int = 120
    sandbox_max_command_timeout: int = 900
    sandbox_workdir: str = "/workspace"
    sandbox_local_root: str = "/tmp/decillion-local-sandboxes"  # only for the `local` test backend

    # ---- features ------------------------------------------------------------------------
    enable_web_tools: bool = True
    web_fetch_max_chars: int = 12000
    mcp_startup_timeout: int = 90

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
