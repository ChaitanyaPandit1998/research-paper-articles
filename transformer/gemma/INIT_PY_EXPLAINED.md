# `__init__.py` Explained — Lazy Loading

```python
from typing import TYPE_CHECKING

from ...utils import _LazyModule
from ...utils.import_utils import define_import_structure


if TYPE_CHECKING:
    from .configuration_gemma import *
    from .modeling_gemma import *
    from .tokenization_gemma import *
    from .tokenization_gemma_fast import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
```

## The Problem This Solves

`transformers` ships hundreds of models. If `import transformers` eagerly imported every model's `modeling_*.py` (which pulls in PyTorch, builds classes, etc.), just importing the library would be very slow — even if you only ever use Gemma.

This file makes the `gemma` package **lazy**: nothing inside `configuration_gemma.py`, `modeling_gemma.py`, etc. is actually imported until you ask for a specific name (e.g. `from transformers import GemmaModel`).

## Two Branches: `TYPE_CHECKING` vs Runtime

### `if TYPE_CHECKING:` branch — for static type checkers / IDEs only

```python
if TYPE_CHECKING:
    from .configuration_gemma import *
    from .modeling_gemma import *
    from .tokenization_gemma import *
    from .tokenization_gemma_fast import *
```

`TYPE_CHECKING` is `False` at actual runtime — it's a constant from `typing` that's only ever `True` when a tool like `mypy`, `pyright`, or your IDE's language server is statically analyzing the code (not running it).

So this branch **never executes when you run Python**. Its only purpose is to tell type checkers and editors "pretend these names are available here," so:
- Autocomplete works (typing `transformers.Gemma` and seeing `GemmaModel`, `GemmaConfig`, etc. suggested)
- `mypy`/`pyright` can resolve `from transformers import GemmaModel` without flagging it as an unknown import
- "Go to definition" in an IDE jumps to the real class

### `else:` branch — what actually runs

```python
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
```

This is the real runtime path. Breaking it down line by line:

1. **`_file = globals()["__file__"]`**
   Gets the absolute path of this `__init__.py` file itself (e.g. `.../transformers/models/gemma/__init__.py`). Needed so the lazy loader knows which directory to scan for submodules.

2. **`define_import_structure(_file)`**
   Scans the `gemma` package directory (`configuration_gemma.py`, `modeling_gemma.py`, `tokenization_gemma.py`, `tokenization_gemma_fast.py`, …) and builds a mapping of:
   ```
   { "configuration_gemma": {"GemmaConfig"},
     "modeling_gemma": {"GemmaModel", "GemmaForCausalLM", ...},
     "tokenization_gemma": {"GemmaTokenizer"},
     ... }
   ```
   It does this by parsing each file's AST (without importing it) to find top-level class/function names, plus the file's `__all__` if defined. It also tracks **optional-dependency requirements** — e.g. `tokenization_gemma_fast` needs the `tokenizers` library, so its entry is tagged so the loader knows to raise a helpful error only if someone actually tries to use it without `tokenizers` installed.

3. **`sys.modules[__name__] = _LazyModule(...)`**
   This is the actual trick. `__name__` here is the dotted module path, e.g. `"transformers.models.gemma"`. Normally, after `__init__.py` runs, Python leaves a `ModuleType` object holding whatever was executed in `sys.modules["transformers.models.gemma"]`.

   Instead, this line **replaces that module object with a custom `_LazyModule` instance** — a class that subclasses `ModuleType` but overrides `__getattr__`. So when later code does:
   ```python
   from transformers.models.gemma import GemmaModel
   ```
   Python looks up `GemmaModel` as an attribute on the `_LazyModule` object. `_LazyModule.__getattr__("GemmaModel")` then:
   - Looks up which submodule defines it (`modeling_gemma`, from the structure built in step 2)
   - **Only now** actually runs `import .modeling_gemma` (triggering the real PyTorch class definitions)
   - Caches the result so the next access is instant

## Why `module_spec=__spec__`

`__spec__` is the `ModuleSpec` Python's import system already created for this package (containing things like the loader, the package's `__path__`, etc.). Passing it through to `_LazyModule` lets the replacement module object still behave correctly under `importlib` introspection (e.g. `importlib.reload`, `pkgutil` tools) — without it, swapping `sys.modules[__name__]` for a different object could confuse tools that expect a module's `__spec__` to be consistent.

## Putting It Together

| Branch | Runs when? | Purpose |
|---|---|---|
| `if TYPE_CHECKING:` | Never at runtime — only for static analysis | Lets editors/type-checkers see real names for autocomplete & type hints |
| `else:` | Always, at actual import time | Installs a `_LazyModule` so submodules are only imported the moment a name from them is actually accessed |

Net effect: `import transformers` (or even `import transformers.models.gemma`) is cheap — no PyTorch classes are built, no tokenizer files are parsed — until you write code that actually touches a specific name like `GemmaModel`, at which point just that one submodule loads.

---

## Deep Dive: `TYPE_CHECKING` — Why It's Always `False` at Runtime

`TYPE_CHECKING` is just a plain `bool` constant defined in the `typing` module:

```python
# inside cpython's typing.py (simplified)
TYPE_CHECKING = False
```

It is *hardcoded to `False`* in the real interpreter. Static analysis tools (mypy, pyright, Pylance, etc.) special-case this exact name: when they parse your code, they treat `TYPE_CHECKING` as `True` instead of actually evaluating the `typing` module. They don't run your code at all — they just read the AST and follow the `if` branch they've decided to treat as "true."

So there are really two completely different programs reading this file:

| Reader | Sees `TYPE_CHECKING` as | Therefore runs |
|---|---|---|
| Python interpreter (`python script.py`) | `False` (the real value) | the `else:` branch — installs `_LazyModule` |
| mypy / pyright / IDE language server | `True` (special-cased) | the `if:` branch — sees `from .modeling_gemma import *` and resolves real names |

This is why the `if TYPE_CHECKING:` branch can safely contain a **wildcard import of every submodule** (`modeling_gemma`, `tokenization_gemma`, etc.) without ever paying the cost of actually importing torch or building model classes — that code is dead weight at runtime, parsed by Python's compiler but never executed, since the `else` always wins.

**Why not just always import everything?** Because then `import transformers` would transitively import every model's `modeling_*.py`, instantiating hundreds of PyTorch class hierarchies and parsing every tokenizer's vocab-handling code — multi-second startup cost for a library with hundreds of models, even if you only want Gemma.

**Why not just skip the `TYPE_CHECKING` block entirely?** Then static checkers and IDEs would have *no* way to know that `transformers.GemmaModel` is a valid name with a particular type — `from transformers import GemmaModel` would show as an unresolved import in your editor, autocomplete would show nothing, and `mypy` would error. The `TYPE_CHECKING` block exists purely to keep tooling happy without affecting real execution.

---

## Deep Dive: How `_LazyModule.__getattr__` Actually Resolves a Name

This relies on a Python feature called **module-level `__getattr__`** (PEP 562, Python 3.7+): if a module defines a `__getattr__(name)` function (or, as here, the module object itself is replaced by an instance of a class with `__getattr__`), Python calls it whenever an attribute lookup on that module *fails* through the normal route (i.e. the name isn't already a real attribute set on the module).

Walking through `from transformers.models.gemma import GemmaModel`:

1. Python imports `transformers.models.gemma` as usual. Its `__init__.py` runs, hits the `else:` branch, and does:
   ```python
   sys.modules["transformers.models.gemma"] = _LazyModule(...)
   ```
   This **overwrites** the module object Python just created with a custom one. From this point on, anyone who does `import transformers.models.gemma` gets this `_LazyModule` instance, not a normal module.

2. Python then does the equivalent of `getattr(sys.modules["transformers.models.gemma"], "GemmaModel")` to satisfy the `import GemmaModel` part.

3. Since `_LazyModule` doesn't have a real attribute named `GemmaModel` (nothing has actually been imported yet), Python falls through to `_LazyModule.__getattr__(self, "GemmaModel")`.

4. Inside `__getattr__` (see `src/transformers/utils/import_utils.py`):
   - It checks `self._class_to_module` — a dict built ahead of time by `define_import_structure` that maps every exportable name to the submodule that defines it, e.g. `{"GemmaModel": "modeling_gemma", "GemmaConfig": "configuration_gemma", ...}`.
   - It finds `"GemmaModel" → "modeling_gemma"`.
   - It calls `self._get_module("modeling_gemma")`, which does:
     ```python
     importlib.import_module(".modeling_gemma", "transformers.models.gemma")
     ```
     — this is the **first real import** of `modeling_gemma.py`. Only now does PyTorch get pulled in and the `GemmaModel` class actually get defined.
   - It does `getattr(module, "GemmaModel")` to pull the real class out of that now-imported module.
   - It caches the result with `setattr(self, name, value)` so the *next* time anyone accesses `GemmaModel` on this module, it's a normal attribute lookup — no `__getattr__`, no re-import.

5. The resolved `GemmaModel` class is handed back to satisfy the original `from ... import GemmaModel` statement.

### Why `_class_to_module` exists ahead of time without importing anything

`define_import_structure` builds this name → submodule map by **parsing each file's AST** (`ast.parse`), not by importing it. It walks the syntax tree looking for top-level `class`/`def` statements and `__all__` declarations. This is how the library can know "`GemmaModel` lives in `modeling_gemma`" without ever running `modeling_gemma.py` — parsing source text is far cheaper than executing it (no torch import, no class body execution).

### Bonus: optional-dependency handling

The same `__getattr__` also checks `self._object_missing_backend` — names that belong to a submodule whose import was tagged as needing an optional dependency (e.g. `tokenization_gemma_fast` needs the `tokenizers` package). If that dependency isn't installed, instead of raising an import error immediately, it hands back a `Placeholder` class whose `__init__` raises a clear "you need to `pip install tokenizers`" error **only if you actually try to instantiate it** — so merely importing `transformers` (or even `transformers.models.gemma`) never fails due to an optional dependency you don't have, only *using* the specific feature that needs it does.

### Summary of the mechanism

```
import transformers.models.gemma
        │
        ▼
__init__.py runs → replaces sys.modules entry with _LazyModule
        │
        ▼
from transformers.models.gemma import GemmaModel
        │
        ▼
Python: getattr(lazy_module, "GemmaModel")
        │
        ▼
_LazyModule.__getattr__("GemmaModel")
        │
        ▼
look up "GemmaModel" in pre-built _class_to_module map → "modeling_gemma"
        │
        ▼
importlib.import_module(".modeling_gemma", ...)   ← REAL import happens HERE, first time only
        │
        ▼
getattr(modeling_gemma_module, "GemmaModel")  → cache it → return the class
```

Every subsequent access of any name from `modeling_gemma` (e.g. `GemmaForCausalLM` next) re-uses the already-imported module object — only the *first* name from a given submodule triggers that submodule's import.
