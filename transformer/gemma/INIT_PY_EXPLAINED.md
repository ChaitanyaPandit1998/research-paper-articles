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
