---
name: pytest-patterns
description: pytest conventions, AAA structure, fixtures, and common assertions
globs: ["tests/**", "**/*_test.py", "**/test_*.py"]
keywords: ["test", "pytest", "fixture", "assert", "mock", "coverage"]
fastpath_safe: true
routing_hint: fast
---

## Structure (AAA)

```python
def test_returns_greeting() -> None:
    # Arrange
    user = User(name="Alice")
    # Act
    result = greet(user)
    # Assert
    assert result == "Hello, Alice!"
```

## Naming

- File: `test_<module>.py`
- Function: `test_<what>_<condition>`
- Class: `TestFoo` — groups related tests, no `__init__`

## Fixtures

```python
@pytest.fixture
def db(tmp_path):
    conn = connect(tmp_path / "test.db")
    yield conn
    conn.close()
```

## Common assertions

```python
assert result == expected
assert "substr" in text
assert value is None
with pytest.raises(ValueError, match="too short"):
    validate("")
```

## Mocking

```python
from unittest.mock import patch
with patch("mymod.requests.get") as mock_get:
    mock_get.return_value.json.return_value = {"ok": True}
    result = fetch_data()
mock_get.assert_called_once()
```

## Running

```bash
pytest -q          # quiet
pytest -x          # stop on first failure
pytest --tb=short  # compact tracebacks
```
