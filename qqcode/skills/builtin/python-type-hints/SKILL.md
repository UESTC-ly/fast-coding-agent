---
name: python-type-hints
description: How to add and fix Python type annotations
globs: ["**/*.py"]
keywords: ["type hint", "type annotation", "typing", "mypy", "annotate"]
fastpath_safe: true
routing_hint: fast
---

## Standard annotations

```python
from __future__ import annotations  # enables forward refs without quotes

def process(items: list[str], limit: int = 10) -> dict[str, int]:
    ...

# Optional / union (Python 3.10+)
def find(name: str) -> str | None:
    ...

# Callable
from collections.abc import Callable
def apply(fn: Callable[[int], str], value: int) -> str:
    ...
```

## Common patterns

| Want | Annotation |
|------|-----------|
| Optional arg | `x: int \| None = None` |
| Variable-length positional | `*args: str` |
| Keyword-only | `**kwargs: int` |
| Generator | `Generator[YieldType, SendType, ReturnType]` |

## Minimal change principle

Add annotations only where the task asks. Do not refactor logic, rename
variables, or change runtime behaviour.

## Running mypy

```bash
mypy --strict <file>
mypy --ignore-missing-imports <file>   # when stubs are absent
```

Fix errors in the order reported — earlier errors often cascade.
