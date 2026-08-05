---
name: fix-imports
description: How to resolve Python import errors and reorganise imports
globs: ["**/*.py"]
keywords: ["import", "ImportError", "ModuleNotFoundError", "circular import", "missing import"]
fastpath_safe: true
routing_hint: fast
---

## Diagnose first

```bash
python -c "import <module>"
python -m py_compile <file>   # syntax + import check
```

## Common fixes

### ModuleNotFoundError
```python
# Run from project root with package on PYTHONPATH, or use explicit path:
from src import utils   # instead of bare `import utils`
```

### Circular import — break with deferred import
```python
def get_value():
    from mypackage.other import helper  # import inside function
    return helper()
```

### Missing `__init__.py`
```bash
touch src/mypackage/__init__.py
```

### Wrong relative import
```python
from . import sibling       # same package
from .. import parent_mod   # one level up
```

## Fix import order with ruff

```bash
ruff check --select I --fix <file>
```

Standard order: `__future__` → stdlib → third-party → local.

## Minimal change principle

Fix only the broken import(s). Do not reorganise unrelated imports unless
the task explicitly asks for it.
