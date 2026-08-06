"""Tests for MCP client: stdio connection, lazy startup, crash isolation."""

import json
import subprocess
from unittest.mock import Mock, patch

import pytest

from qqcode.tools.mcp import MCPCapability, MCPServerConfig
from qqcode.tools.mcp_client import MCPClient

# --- Fixtures ------------------------------------------------------------------


@pytest.fixture
def mock_stdio_server():
    """Mock subprocess.Popen for a well-behaved stdio server."""
    mock_proc = Mock(spec=subprocess.Popen)
    mock_proc.poll.return_value = None  # Still running
    mock_proc.stdin = Mock()
    mock_proc.stdout = Mock()
    mock_proc.stderr = Mock()

    # Mock initialization message
    init_msg = {
        "tools": [
            {
                "name": "read_file",
                "description": "Read a file",
                "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
            {
                "name": "write_file",
                "description": "Write a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            },
        ]
    }
    mock_proc.stdout.readline.return_value = (json.dumps(init_msg) + "\n").encode("utf-8")

    return mock_proc


@pytest.fixture
def read_only_config():
    """A read-only stdio server config."""
    return MCPServerConfig(
        name="test-server",
        transport="stdio",
        capabilities=frozenset({MCPCapability.READ}),
        command=["python", "-m", "test_server"],
        enabled_tiers=frozenset({"fullagent"}),
    )


@pytest.fixture
def write_capable_config():
    """A write-capable stdio server config."""
    return MCPServerConfig(
        name="fs-server",
        transport="stdio",
        capabilities=frozenset({MCPCapability.READ, MCPCapability.WRITE}),
        command=["python", "-m", "fs_server"],
        shadow_root_arg="--root",
        enabled_tiers=frozenset({"fullagent"}),
    )


# --- Lazy startup --------------------------------------------------------------


def test_lazy_startup_deferred_until_register(read_only_config):
    """Servers are not started at client construction."""
    client = MCPClient(shadow_root="/tmp/workspace")
    assert len(client._sessions) == 0


def test_register_server_launches_stdio(mock_stdio_server, read_only_config):
    """register_server starts a stdio process and reads tool schemas."""
    with patch("subprocess.Popen", return_value=mock_stdio_server):
        client = MCPClient()
        tools = client.register_server(read_only_config)

    assert len(tools) == 2
    assert tools[0].name == "read_file"
    assert tools[1].name == "write_file"
    assert "test-server" in client._sessions
    assert client._sessions["test-server"].process is mock_stdio_server


def test_register_server_idempotent(mock_stdio_server, read_only_config):
    """Calling register_server twice returns cached schemas without relaunch."""
    with patch("subprocess.Popen", return_value=mock_stdio_server) as popen_mock:
        client = MCPClient()
        tools1 = client.register_server(read_only_config)
        tools2 = client.register_server(read_only_config)

    assert tools1 == tools2
    assert popen_mock.call_count == 1


def test_register_server_respects_tool_allowlist(mock_stdio_server):
    """Only tools in the allowlist are registered."""
    config = MCPServerConfig(
        name="filtered",
        transport="stdio",
        capabilities=frozenset({MCPCapability.READ}),
        command=["python", "-m", "server"],
        tool_allowlist=frozenset({"read_file"}),
        enabled_tiers=frozenset({"fullagent"}),
    )

    with patch("subprocess.Popen", return_value=mock_stdio_server):
        client = MCPClient()
        tools = client.register_server(config)

    assert len(tools) == 1
    assert tools[0].name == "read_file"


# --- Shadow root confinement ---------------------------------------------------


def test_write_capable_server_receives_shadow_root(mock_stdio_server, write_capable_config):
    """A write-capable server is launched with --root <shadow_root>."""
    shadow = "/tmp/shadow_workspace_123"

    with patch("subprocess.Popen", return_value=mock_stdio_server) as popen_mock:
        client = MCPClient(shadow_root=shadow)
        client.register_server(write_capable_config)

    call_args = popen_mock.call_args[0][0]
    assert "--root" in call_args
    assert shadow in call_args


def test_read_only_server_no_shadow_arg(mock_stdio_server, read_only_config):
    """A read-only server does not receive the shadow root arg."""
    with patch("subprocess.Popen", return_value=mock_stdio_server) as popen_mock:
        client = MCPClient(shadow_root="/tmp/shadow")
        client.register_server(read_only_config)

    call_args = popen_mock.call_args[0][0]
    assert "/tmp/shadow" not in call_args


# --- Tool calls ----------------------------------------------------------------


def test_call_tool_sends_request_and_parses_response(mock_stdio_server, read_only_config):
    """call_tool sends JSON over stdin and reads the response."""
    response_msg = {"content": "file contents here"}
    mock_stdio_server.stdout.readline.side_effect = [
        (json.dumps({"tools": []}) + "\n").encode("utf-8"),  # init
        (json.dumps(response_msg) + "\n").encode("utf-8"),  # tool response
    ]

    with patch("subprocess.Popen", return_value=mock_stdio_server):
        client = MCPClient()
        client.register_server(read_only_config)
        result = client.call_tool("test-server", "read_file", {"path": "src/main.py"})

    assert result == "file contents here"
    mock_stdio_server.stdin.write.assert_called_once()
    written = mock_stdio_server.stdin.write.call_args[0][0].decode("utf-8")
    req = json.loads(written.strip())
    assert req["tool"] == "read_file"
    assert req["arguments"]["path"] == "src/main.py"


def test_call_tool_raises_on_server_error(mock_stdio_server, read_only_config):
    """An error response from the server raises RuntimeError."""
    error_msg = {"error": "file not found"}
    mock_stdio_server.stdout.readline.side_effect = [
        (json.dumps({"tools": []}) + "\n").encode("utf-8"),
        (json.dumps(error_msg) + "\n").encode("utf-8"),
    ]

    with patch("subprocess.Popen", return_value=mock_stdio_server):
        client = MCPClient()
        client.register_server(read_only_config)

        with pytest.raises(RuntimeError, match="file not found"):
            client.call_tool("test-server", "read_file", {"path": "missing.txt"})


def test_call_tool_raises_if_server_not_registered():
    """Calling a tool on an unregistered server raises."""
    client = MCPClient()
    with pytest.raises(RuntimeError, match="not registered"):
        client.call_tool("unknown-server", "some_tool", {})


# --- Crash isolation -----------------------------------------------------------


def test_call_tool_detects_dead_process(read_only_config):
    """A crashed server is detected and reported as an error."""
    # Setup: register a server first, then mark it as dead
    mock_proc = Mock(spec=subprocess.Popen)
    mock_proc.poll.return_value = None  # Alive during startup
    mock_proc.stdin = Mock()
    mock_proc.stdout = Mock()
    mock_proc.stderr = Mock()
    init_msg = {"tools": []}
    mock_proc.stdout.readline.return_value = (json.dumps(init_msg) + "\n").encode("utf-8")

    with patch("subprocess.Popen", return_value=mock_proc):
        client = MCPClient()
        client.register_server(read_only_config)

        # Now mark the process as dead
        mock_proc.poll.return_value = 1  # Exited

        with pytest.raises(RuntimeError, match="died"):
            client.call_tool("test-server", "read_file", {"path": "x"})


def test_startup_timeout_kills_hung_server(read_only_config):
    """A server that never sends an init message is killed after timeout."""
    mock_proc = Mock(spec=subprocess.Popen)
    mock_proc.poll.return_value = None
    mock_proc.stdout = Mock()
    mock_proc.stdout.readline.return_value = b""  # Never completes

    config = MCPServerConfig(
        name=read_only_config.name,
        transport=read_only_config.transport,
        capabilities=read_only_config.capabilities,
        command=read_only_config.command,
        startup_timeout=0.1,
    )

    with patch("subprocess.Popen", return_value=mock_proc):
        client = MCPClient()
        with pytest.raises(RuntimeError, match="no init message"):
            client.register_server(config)

    mock_proc.kill.assert_called_once()


def test_server_crash_during_startup_raises(read_only_config):
    """A server that exits during init raises with stderr."""
    mock_proc = Mock(spec=subprocess.Popen)
    mock_proc.poll.return_value = 127
    mock_proc.communicate.return_value = (b"", b"command not found")

    with patch("subprocess.Popen", return_value=mock_proc):
        client = MCPClient()
        with pytest.raises(RuntimeError, match="exited during startup.*command not found"):
            client.register_server(read_only_config)


# --- Shutdown ------------------------------------------------------------------


def test_shutdown_all_terminates_stdio_servers(mock_stdio_server, read_only_config):
    """shutdown_all gracefully terminates all managed processes."""
    with patch("subprocess.Popen", return_value=mock_stdio_server):
        client = MCPClient()
        client.register_server(read_only_config)
        client.shutdown_all()

    mock_stdio_server.terminate.assert_called_once()
    mock_stdio_server.wait.assert_called_once()
    assert len(client._sessions) == 0


def test_shutdown_all_kills_unresponsive_servers(read_only_config):
    """If terminate+wait times out, kill is called."""
    mock_proc = Mock(spec=subprocess.Popen)
    mock_proc.poll.return_value = None
    mock_proc.stdin = Mock()
    mock_proc.stdout = Mock()
    mock_proc.stderr = Mock()
    init_msg = {"tools": []}
    mock_proc.stdout.readline.return_value = (json.dumps(init_msg) + "\n").encode("utf-8")
    mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="", timeout=2.0)

    with patch("subprocess.Popen", return_value=mock_proc):
        client = MCPClient()
        client.register_server(read_only_config)
        client.shutdown_all()

    mock_proc.terminate.assert_called_once()
    mock_proc.kill.assert_called_once()


# --- SSE transport (not implemented) -------------------------------------------


def test_sse_transport_not_implemented():
    """SSE servers raise NotImplementedError."""
    config = MCPServerConfig(
        name="sse-server",
        transport="sse",
        capabilities=frozenset({MCPCapability.READ}),
        url="https://example.com/mcp",
        enabled_tiers=frozenset({"fullagent"}),
    )

    client = MCPClient()
    with pytest.raises(NotImplementedError, match="SSE transport not yet implemented"):
        client.register_server(config)
