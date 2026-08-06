"""Configuration loader for QQCode.

Reads API credentials and base URLs from environment variables (via .env file).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Global fallback, consulted when no .env exists in the current directory or
# any parent. Without it `qqcode --chat` only works inside the directory tree
# that happens to hold a .env, which is the wrong constraint for a tool meant
# to be pointed at arbitrary repositories.
GLOBAL_CONFIG_DIR = Path.home() / ".config" / "qqcode"
GLOBAL_ENV_FILENAME = "env"


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for one model provider."""

    api_key: str
    base_url: str | None = None


@dataclass(frozen=True)
class Config:
    """Application configuration."""

    anthropic: ProviderConfig | None
    openai: ProviderConfig | None
    default_provider: str
    debug: bool = False

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> Config:
        """Load configuration from environment.

        Args:
            env_path: Optional path to .env file. If None, searches for .env
                in the current directory and parent directories.
        """
        # Load .env file if it exists (simple inline implementation to avoid
        # adding python-dotenv as a hard dependency)
        if env_path is None:
            env_path = cls._find_dotenv()
        if env_path and env_path.is_file():
            cls._load_dotenv(env_path)

        # Read Anthropic config
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        anthropic = (
            ProviderConfig(
                api_key=anthropic_key,
                base_url=os.getenv("ANTHROPIC_BASE_URL"),
            )
            if anthropic_key
            else None
        )

        # Read OpenAI config
        openai_key = os.getenv("OPENAI_API_KEY")
        openai = (
            ProviderConfig(
                api_key=openai_key,
                base_url=os.getenv("OPENAI_BASE_URL"),
            )
            if openai_key
            else None
        )

        default_provider = os.getenv("DEFAULT_PROVIDER", "anthropic")
        if default_provider not in {"anthropic", "openai"}:
            raise ValueError(f"Invalid DEFAULT_PROVIDER: {default_provider}")

        debug = os.getenv("DEBUG", "false").lower() in {"1", "true", "yes"}

        return cls(
            anthropic=anthropic,
            openai=openai,
            default_provider=default_provider,
            debug=debug,
        )

    @staticmethod
    def _find_dotenv() -> Path | None:
        """Locate the environment file.

        A project-local `.env` wins over the global one: a repository that
        pins its own provider or base URL should not be overridden by whatever
        the user configured globally. The global file is the fallback that lets
        `qqcode` run against a repository that has no .env of its own.
        """
        cwd = Path.cwd()
        for parent in [cwd, *cwd.parents]:
            candidate = parent / ".env"
            if candidate.is_file():
                return candidate

        global_env = GLOBAL_CONFIG_DIR / GLOBAL_ENV_FILENAME
        if global_env.is_file():
            return global_env
        return None

    @staticmethod
    def _load_dotenv(path: Path) -> None:
        """Parse .env file and set environment variables."""
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Remove surrounding quotes if present
            if value and value[0] in {'"', "'"} and value[0] == value[-1]:
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value
