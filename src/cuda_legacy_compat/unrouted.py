"""Say plainly when a run ends on an operation the adapter has no Maxwell route for.

The adapter records the operation and the exception it let through; when that exception (or one
raised from it) ends the script unhandled, one line after the traceback names the operation and
the next step. The exception itself, its traceback and the exit code are unchanged.
"""

from __future__ import annotations

import sys
from types import TracebackType
from typing import Callable

from cuda_legacy_compat import __version__

COVERAGE_URL = "https://cudacompat.com/docs/compatibility.html"
REPORT_URL = "https://github.com/zachshallbetter/cuda-compatibility/issues/new?template=reproduction-report.yml"

_recorded: tuple[str, BaseException] | None = None


def record(operation: str, error: BaseException) -> None:
    """Remember the latest operation the adapter could not route, and the error it raised."""
    global _recorded
    _recorded = (operation, error)


def _chain(error: BaseException | None):
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        yield error
        error = error.__cause__ or error.__context__


def hint_for(error: BaseException) -> str | None:
    """The hint for an unhandled error, or None when the adapter did not let it through."""
    if _recorded is None:
        return None
    operation, recorded = _recorded
    if not any(link is recorded for link in _chain(error)):
        return None
    return (
        f"cuda-legacy: {operation} has no sm_52 route in cuda-legacy {__version__}, so this GPU cannot run it. "
        f"See what runs: {COVERAGE_URL} | Ask for it: {REPORT_URL}"
    )


Hook = Callable[[type[BaseException], BaseException, TracebackType | None], object]


def install_excepthook() -> None:
    """Print the hint after the normal traceback of an unhandled error."""
    previous: Hook = sys.excepthook

    def excepthook(kind: type[BaseException], error: BaseException, trace: TracebackType | None) -> None:
        previous(kind, error, trace)
        hint = hint_for(error)
        if hint:
            print(hint, file=sys.stderr)

    sys.excepthook = excepthook
