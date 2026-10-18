"""Local-deadline assignment techniques.

A technique is a callable that receives a system and sets each task's local
deadline. Register one with :func:`register`::

    from algorithms import register

    @register
    def my_technique(system):
        for task in system.tasks:
            task.deadline = ...

The function name is the label used in the report, and every module in this
package is imported automatically, so adding a file here is enough for its
techniques to be found by the harness.
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
