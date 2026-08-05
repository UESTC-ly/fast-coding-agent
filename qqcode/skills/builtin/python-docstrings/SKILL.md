---
name: python-docstrings
description: How to write and place Python docstrings correctly
globs: ["**/*.py"]
keywords: ["docstring", "doc", "document", "comment", "description"]
fastpath_safe: true
routing_hint: fast
---

## Docstring format

Use Google style. One-line docstrings for simple functions; multi-line for
anything with arguments, return values, or raised exceptions.

```python
def greet(name: str) -> str:
    """Return a personalised greeting."""
    return f"Hello, {name}!"


def divide(a: float, b: float) -> float:
    """Divide a by b.

    Args:
        a: Numerator.
        b: Denominator.

    Returns:
        Result of the division.

    Raises:
        ZeroDivisionError: When b is zero.
    """
    return a / b
```

## Placement rules

- Module docstring: first statement of the file, before imports
- Class docstring: immediately after `class Foo:`
- Method/function: immediately after `def foo():`
- No blank line between `def` and the docstring

## Minimal change principle

When adding a docstring, change only the docstring line(s). Do not reformat
surrounding code, adjust spacing, or rename anything.
