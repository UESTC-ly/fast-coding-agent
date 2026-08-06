"""Event tests: progress visibility, and its isolation from execution."""

from __future__ import annotations

from qqcode.events import (
    MAX_DETAIL_CHARS,
    AgentEvent,
    describe_tool_call,
    emit,
)


class TestEmit:
    def test_none_callback_is_a_noop(self) -> None:
        """Batch mode passes None; it must cost nothing and raise nothing."""
        emit(None, AgentEvent(kind="turn_start", turn=1))

    def test_callback_receives_the_event(self) -> None:
        seen: list[AgentEvent] = []
        emit(seen.append, AgentEvent(kind="tool_end", tool="read_file"))
        assert len(seen) == 1
        assert seen[0].tool == "read_file"

    def test_a_throwing_callback_cannot_break_the_agent(self) -> None:
        """A rendering bug must not abort a run that is otherwise succeeding."""
        def broken(_event: AgentEvent) -> None:
            raise RuntimeError("terminal exploded")

        emit(broken, AgentEvent(kind="turn_start", turn=1))  # must not raise


class TestDescribeToolCall:
    def test_path_is_preferred(self) -> None:
        assert describe_tool_call("read_file", {"path": "src/a.py"}) == "src/a.py"

    def test_command_list_is_joined(self) -> None:
        """`cmd` is the real parameter name — `command` silently renders as keys."""
        detail = describe_tool_call("run_command", {"cmd": ["pytest", "-q"]})
        assert detail == "pytest -q"

    def test_every_builtin_tool_renders_its_value_not_its_keys(self) -> None:
        """Pinned to the real schemas.

        The fallback returns key names, so a mismatch between this module's
        expected parameter names and the registry's actual ones degrades quietly
        rather than failing. This asserts against the registry itself.
        """
        from qqcode.tools.builtins import default_registry

        probes: dict[str, object] = {
            "path": "src/a.py",
            "cmd": ["pytest", "-q"],
            "pattern": "def main",
            "name": "some-skill",
            "preset": "explorer",
            "artifact_id": "art-1",
            "summary": "did the work",
            "old_string": "a",
            "new_string": "b",
            "content": "body",
            "glob": "*.py",
            "task": "look into it",
            "timeout": 30,
            "files_changed": ["a.py"],
        }

        for spec in default_registry().specs_for(tier="fullagent"):
            props = spec.input_schema.get("properties", {})
            args = {k: probes[k] for k in props if k in probes}
            detail = describe_tool_call(spec.name, args)
            # The fallback joins sorted key names; a correct render never equals it.
            assert detail != ", ".join(sorted(args)), (
                f"{spec.name} fell through to the key-name fallback; "
                f"describe_tool_call does not know its parameters {list(props)}"
            )

    def test_pattern_is_used_for_search(self) -> None:
        assert describe_tool_call("grep", {"pattern": "def main"}) == "def main"

    def test_empty_input_yields_empty_detail(self) -> None:
        assert describe_tool_call("finish", {}) == ""

    def test_unknown_tool_shows_keys_not_values(self) -> None:
        """An unknown tool's values may be arbitrarily large."""
        detail = describe_tool_call("mystery", {"payload": "x" * 10_000})
        assert detail == "payload"

    def test_long_detail_is_truncated(self) -> None:
        detail = describe_tool_call("write_file", {"path": "a/" * 500})
        assert len(detail) <= MAX_DETAIL_CHARS + 1
        assert detail.endswith("…")

    def test_newlines_are_collapsed(self) -> None:
        """A multi-line detail would break the one-line-per-event rendering."""
        detail = describe_tool_call("run_command", {"command": "echo a\nb"})
        assert "\n" not in detail

    def test_write_file_does_not_leak_content(self) -> None:
        detail = describe_tool_call(
            "write_file", {"path": "a.py", "content": "SECRET BODY " * 100}
        )
        assert detail == "a.py"
        assert "SECRET" not in detail
