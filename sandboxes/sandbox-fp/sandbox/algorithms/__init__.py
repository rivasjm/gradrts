"""Your workspace: fixed-priority assignment algorithms live here.

Register an algorithm by decorating a function with :func:`register`::

    from algorithms import register

    @register
    def my_policy(system):
        # set task.priority on every task (larger number = higher priority)
        ...

Every module in this package is imported automatically, so adding a new file
here is enough for its algorithms to show up in ``python run.py``.
"""

import importlib
import pkgutil

REGISTRY = {}


def register(function):
    """Register ``function`` under its own name (used as the report label)."""
    REGISTRY[function.__name__] = function
    return function


def load(reload=False):
    """Import every module in this package and return the registry."""
    if reload:
        REGISTRY.clear()
    for module in pkgutil.iter_modules(__path__):
        importlib.import_module(f"{__name__}.{module.name}")
    return dict(REGISTRY)
