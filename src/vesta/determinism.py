"""Vesta's numbers do not depend on the machine's threads.

The linear algebra library (BLAS) splits a long dot product across its threads, so the order in
which partial sums are added depends on the machine; an optimiser that stops at a tolerance then
stops one iteration earlier or later, and a published number changes in its last digits, enough
to flip a simulated season. Wherever Vesta fits or
forecasts, the BLAS runs on one thread (`one_thread`), which is also the fastest on these sizes.
"""

import importlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps

THREADS = 1


@contextmanager
def pin_blas_to_one_thread() -> Iterator[None]:
    """Every BLAS loaded in the process limited to one thread for the block, the limits of before
    restored after. The libraries are loaded first: a limit applies only to those already loaded."""
    from threadpoolctl import threadpool_limits

    for name in ("numpy", "scipy.linalg", "scipy.optimize", "scipy.special"):
        importlib.import_module(name)
    with threadpool_limits(limits=THREADS, user_api="blas"):
        yield


def run_on_one_thread[**Arguments, Result](
    function: Callable[Arguments, Result],
) -> Callable[Arguments, Result]:
    """Run `function` with the BLAS on one thread: for every entry point that fits or forecasts."""

    @wraps(function)
    def run(*arguments: Arguments.args, **keywords: Arguments.kwargs) -> Result:
        with pin_blas_to_one_thread():
            return function(*arguments, **keywords)

    return run
