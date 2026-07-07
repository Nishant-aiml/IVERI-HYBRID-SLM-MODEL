"""Component registry for dynamically discovering and instantiating modules.

The :class:`ComponentRegistry` maintains a global, name-keyed mapping of
component *types*.  Modules register themselves via the ``@register``
decorator and are later retrieved by name through :func:`build_component`
in the factory module.

Example
-------
>>> from core.registry import register
>>>
>>> @register("my_encoder")
... class MyEncoder(BaseEncoder):
...     ...
"""

from __future__ import annotations

from typing import TypeVar

from core.exceptions import RegistryError

_T = TypeVar("_T", bound=type)


class ComponentRegistry:
    """Central name → type mapping for all pluggable IVERI components."""

    _registry: dict[str, type] = {}

    # -- registration ------------------------------------------------------

    @classmethod
    def register(cls, name: str):  # noqa: ANN206 – returns generic decorator
        """Return a decorator that registers *component_cls* under *name*.

        Parameters
        ----------
        name:
            Unique string key used to look up the component later.

        Raises
        ------
        RegistryError
            If *name* is already registered.
        """

        def _decorator(component_cls: _T) -> _T:
            if name in cls._registry:
                raise RegistryError(
                    f"Duplicate registration: '{name}' is already registered "
                    f"to {cls._registry[name].__qualname__}.",
                )
            cls._registry[name] = component_cls
            return component_cls

        return _decorator

    # -- lookup ------------------------------------------------------------

    @classmethod
    def get(cls, name: str) -> type:
        """Retrieve a registered component type by *name*.

        Parameters
        ----------
        name:
            The string key used during registration.

        Returns
        -------
        type
            The registered class.

        Raises
        ------
        RegistryError
            If *name* has not been registered.
        """
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry)) or "(none)"
            raise RegistryError(
                f"Component '{name}' is not registered.",
                details=f"Available components: {available}",
            )
        return cls._registry[name]

    # -- introspection / testing -------------------------------------------

    @classmethod
    def list_registered(cls) -> list[str]:
        """Return a sorted list of all registered component names."""
        return sorted(cls._registry)

    @classmethod
    def clear(cls) -> None:
        """Remove **all** registered components (intended for test teardown)."""
        cls._registry.clear()


# ---------------------------------------------------------------------------
# Module-level convenience alias
# ---------------------------------------------------------------------------
register = ComponentRegistry.register
"""Shorthand decorator — equivalent to ``ComponentRegistry.register``."""
