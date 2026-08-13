"""Decorator-based factor registration and discovery."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

from .spec import Category, Factor, FactorSpec, Market

_REGISTRY: dict[str, Factor] = {}
FactorType = TypeVar("FactorType", bound=type[Factor])


def register_factor(factor_class: FactorType) -> FactorType:
    """Instantiate and register a factor class by its unique specification name."""
    factor = factor_class()
    name = factor.spec.name
    if name in _REGISTRY:
        raise ValueError(f"Factor already registered: {name}")
    _REGISTRY[name] = factor
    return factor_class


def get_factor(name: str) -> Factor:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown factor: {name}") from exc


def get_spec(name: str) -> FactorSpec:
    return get_factor(name).spec


def list_factors(
    market: Market | None = None,
    category: Category | None = None,
) -> tuple[FactorSpec, ...]:
    factors: Iterable[Factor] = _REGISTRY.values()
    specs = (
        factor.spec
        for factor in factors
        if (market is None or market in factor.spec.markets) and (category is None or factor.spec.category == category)
    )
    return tuple(sorted(specs, key=lambda spec: spec.name))


def _clear_registry_for_tests() -> None:
    """Clear global registrations for isolated registry tests."""
    _REGISTRY.clear()
