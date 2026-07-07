"""Factory functions for instantiating registered IVERI CORE components.

This module bridges the :mod:`core.registry` (which stores *types*) with
the rest of the codebase that needs *instances*.  It also provides
lightweight parameter-counting utilities used across training scripts and
experiment notebooks.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from core.registry import ComponentRegistry


def build_component(name: str, **kwargs: Any) -> nn.Module:
    """Instantiate a registered component by *name*.

    Parameters
    ----------
    name:
        Registry key assigned via ``@register(name)``.
    **kwargs:
        Keyword arguments forwarded to the component's constructor.

    Returns
    -------
    nn.Module
        A freshly constructed module instance.

    Raises
    ------
    core.exceptions.RegistryError
        If *name* is not registered (propagated from
        :meth:`ComponentRegistry.get`).
    """
    from typing import cast

    component_cls = ComponentRegistry.get(name)
    return cast(nn.Module, component_cls(**kwargs))


def build_model(config: Any) -> nn.Module:
    """Assemble the full IVERI CORE model from a config object.

    .. note::
        This is a **Phase 0 placeholder**.  Full model assembly will be
        implemented in Phase 1 once all architectural sub-components
        (BLT encoder, Mamba2 backbone, Titans memory, MoE router, …)
        are available.

    Parameters
    ----------
    config:
        A model configuration object (schema TBD in Phase 1).

    Raises
    ------
    NotImplementedError
        Always — this function is not yet implemented.
    """
    raise NotImplementedError("Model assembly available in Phase 1")


# ---------------------------------------------------------------------------
# Parameter introspection utilities
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> int:
    """Return the total number of **trainable** parameters in *model*.

    Parameters
    ----------
    model:
        Any :class:`torch.nn.Module`.

    Returns
    -------
    int
        Scalar count of elements across all ``requires_grad=True`` params.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_by_module(model: nn.Module) -> dict[str, int]:
    """Return trainable parameter counts keyed by top-level sub-module name.

    Parameters
    ----------
    model:
        Any :class:`torch.nn.Module` with named children.

    Returns
    -------
    dict[str, int]
        Mapping from child module name to its trainable parameter count.
        Modules with zero trainable parameters are included for
        completeness.
    """
    counts: dict[str, int] = {}
    for name, child in model.named_children():
        counts[name] = sum(p.numel() for p in child.parameters() if p.requires_grad)
    return counts
