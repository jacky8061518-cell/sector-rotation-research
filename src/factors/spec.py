"""Public factor contracts shared by the library, evaluator, and UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import pandas as pd

if TYPE_CHECKING:
    from .context import DataContext


Category = Literal["momentum", "value", "quality", "growth", "risk", "liquidity", "flow"]
Market = Literal["US", "TW"]


@dataclass(frozen=True)
class FactorSpec:
    """Immutable metadata for a registered cross-sectional factor."""

    name: str
    label: str
    category: Category
    direction: int
    lookback_days: int
    requires: tuple[str, ...]
    markets: tuple[Market, ...]
    description: str
    reference: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ValueError("Factor name must be a non-empty Python identifier.")
        if self.direction not in {-1, 1}:
            raise ValueError("Factor direction must be either -1 or +1.")
        if self.lookback_days <= 0:
            raise ValueError("Factor lookback_days must be positive.")
        if not self.markets:
            raise ValueError("Factor must support at least one market.")


class Factor(Protocol):
    """Computational interface implemented by every factor."""

    spec: FactorSpec

    def compute(self, ctx: DataContext, asof: pd.Timestamp) -> pd.Series:
        """Return raw factor values indexed by ticker, leaving gaps as NaN."""
