"""Config discovery tests: project-local .env versus the global fallback.

These never touch the real ~/.config/qqcode: GLOBAL_CONFIG_DIR is redirected to
a tmp_path for every test that exercises the fallback.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qqcode.config import Config


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config._load_dotenv never overwrites an existing variable, so a real key
    in the ambient environment would mask what the file under test provides."""
    for var in (
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
        "ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "DEFAULT_PROVIDER",
        "DEFAULT_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)


def _write_env(path: Path, key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"ANTHROPIC_API_KEY={key}\n")


class TestGlobalFallback:
    def test_global_env_is_used_when_no_local_env_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The point of the fallback: run against a repo that has no .env."""
        home = tmp_path / "home"
        _write_env(home / ".config" / "qqcode" / "env", "from-global")
        monkeypatch.setattr(
            "qqcode.config.GLOBAL_CONFIG_DIR", home / ".config" / "qqcode"
        )

        workdir = tmp_path / "elsewhere"
        workdir.mkdir()
        monkeypatch.chdir(workdir)

        config = Config.from_env()
        assert config.anthropic is not None
        assert config.anthropic.api_key == "from-global"

    def test_local_env_wins_over_global(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A repo pinning its own provider must not be overridden globally."""
        home = tmp_path / "home"
        _write_env(home / ".config" / "qqcode" / "env", "from-global")
        monkeypatch.setattr(
            "qqcode.config.GLOBAL_CONFIG_DIR", home / ".config" / "qqcode"
        )

        workdir = tmp_path / "project"
        _write_env(workdir / ".env", "from-local")
        monkeypatch.chdir(workdir)

        config = Config.from_env()
        assert config.anthropic is not None
        assert config.anthropic.api_key == "from-local"

    def test_no_env_anywhere_yields_no_providers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("qqcode.config.GLOBAL_CONFIG_DIR", tmp_path / "absent")
        workdir = tmp_path / "bare"
        workdir.mkdir()
        monkeypatch.chdir(workdir)

        config = Config.from_env()
        assert config.anthropic is None
        assert config.openai is None

    def test_explicit_env_path_bypasses_discovery(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        _write_env(home / ".config" / "qqcode" / "env", "from-global")
        monkeypatch.setattr(
            "qqcode.config.GLOBAL_CONFIG_DIR", home / ".config" / "qqcode"
        )

        explicit = tmp_path / "custom.env"
        _write_env(explicit, "from-explicit")

        config = Config.from_env(env_path=explicit)
        assert config.anthropic is not None
        assert config.anthropic.api_key == "from-explicit"


class TestDefaultModel:
    """DEFAULT_MODEL lets one config point qqcode at a backend whose model ids
    differ from the built-in defaults, without passing --model every time."""

    def test_absent_by_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("qqcode.config.GLOBAL_CONFIG_DIR", tmp_path / "nowhere")
        env = tmp_path / ".env"
        env.write_text("ANTHROPIC_API_KEY=k\n")
        assert Config.from_env(env).default_model is None

    def test_read_from_env_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("qqcode.config.GLOBAL_CONFIG_DIR", tmp_path / "nowhere")
        env = tmp_path / ".env"
        env.write_text("ANTHROPIC_API_KEY=k\nDEFAULT_MODEL=some-model-id\n")
        assert Config.from_env(env).default_model == "some-model-id"

    def test_blank_is_treated_as_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty value must not pin every tier to the empty string."""
        monkeypatch.setattr("qqcode.config.GLOBAL_CONFIG_DIR", tmp_path / "nowhere")
        env = tmp_path / ".env"
        env.write_text("ANTHROPIC_API_KEY=k\nDEFAULT_MODEL=\n")
        assert Config.from_env(env).default_model is None
