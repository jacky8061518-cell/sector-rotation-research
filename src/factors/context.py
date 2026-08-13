"""Point-in-time data access boundary for every factor computation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import time
from typing import Literal

import pandas as pd

from .spec import Market

PriceField = Literal["adj_close", "close", "volume", "traded_value"]


def _empty_panel(columns: Sequence[str] = ()) -> pd.DataFrame:
    index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([], name="date"), pd.Index([], name="ticker")]
    )
    return pd.DataFrame(index=index, columns=list(columns), dtype=float)


def _normalize_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize long data to a sorted (date, ticker) MultiIndex."""
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    if not isinstance(result.index, pd.MultiIndex):
        if not {"date", "ticker"}.issubset(result.columns):
            raise ValueError("Panel data must contain date and ticker dimensions.")
        result["date"] = pd.to_datetime(result["date"])
        result = result.set_index(["date", "ticker"])
    if result.index.nlevels != 2:
        raise ValueError("Panel index must have exactly two levels: date and ticker.")
    dates = pd.to_datetime(result.index.get_level_values(0))
    tickers = result.index.get_level_values(1).astype(str)
    result.index = pd.MultiIndex.from_arrays([dates, tickers], names=["date", "ticker"])
    return result.sort_index()


@dataclass(frozen=True)
class FactorDataStore:
    """In-memory tables loaded by the existing application cache layer."""

    price_data: pd.DataFrame
    universe_data: pd.DataFrame
    market_cap_data: pd.DataFrame = field(default_factory=_empty_panel)
    inst_flow_data: pd.DataFrame = field(default_factory=_empty_panel)
    revenue_data: pd.DataFrame = field(default_factory=_empty_panel)
    adjusted_close_wide: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "price_data", _normalize_panel(self.price_data))
        object.__setattr__(self, "market_cap_data", _normalize_panel(self.market_cap_data))
        object.__setattr__(self, "inst_flow_data", _normalize_panel(self.inst_flow_data))
        object.__setattr__(self, "revenue_data", _normalize_panel(self.revenue_data))
        if self.adjusted_close_wide is not None:
            wide = self.adjusted_close_wide.copy()
            wide.index = pd.to_datetime(wide.index)
            object.__setattr__(self, "adjusted_close_wide", wide.sort_index())


@dataclass(frozen=True)
class DataContext:
    """An as-of-bound view that cannot expose observations from the future."""

    store: FactorDataStore
    market: Market
    asof: pd.Timestamp
    benchmark: str

    def __post_init__(self) -> None:
        timestamp = pd.Timestamp(self.asof)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert(None)
        if timestamp == timestamp.normalize():
            close_time = time(13, 30) if self.market == "TW" else time(16)
            timestamp = timestamp.replace(
                hour=close_time.hour,
                minute=close_time.minute,
            )
        object.__setattr__(self, "asof", timestamp)

    def _slice(
        self,
        frame: pd.DataFrame,
        window: int | None,
        *,
        knowledge_column: str | None = None,
    ) -> pd.DataFrame:
        if window is not None and window <= 0:
            raise ValueError("window must be positive when provided.")
        if frame.empty:
            return frame.copy()
        date_level = frame.index.names.index("date")
        sessions = pd.DatetimeIndex(frame.index.levels[date_level])
        visible_sessions = sessions[sessions <= self.asof]
        if visible_sessions.empty:
            return frame.iloc[0:0].copy()
        start = (
            visible_sessions[0]
            if window is None
            else visible_sessions[max(0, len(visible_sessions) - window)]
        )
        try:
            result = frame.loc[pd.IndexSlice[start : self.asof, :], :].copy()
        except KeyError:
            return frame.iloc[0:0].copy()
        if knowledge_column is not None:
            if knowledge_column not in result:
                raise ValueError(f"Missing point-in-time column: {knowledge_column}")
            known = pd.to_datetime(result[knowledge_column], errors="coerce")
            result = result.loc[known.notna() & known.le(self.asof)]
        allowed = self.universe().union(pd.Index([self.benchmark]))
        return result.loc[result.index.get_level_values("ticker").isin(allowed)]

    def prices(
        self,
        fields: Sequence[PriceField] = ("adj_close",),
        window: int | None = None,
    ) -> pd.DataFrame:
        if tuple(fields) == ("adj_close",) and self.store.adjusted_close_wide is not None:
            wide = self.store.adjusted_close_wide.loc[: self.asof]
            if window is not None:
                if window <= 0:
                    raise ValueError("window must be positive when provided.")
                wide = wide.tail(window)
            allowed = self.universe().union(pd.Index([self.benchmark]))
            selected = wide.reindex(columns=allowed.intersection(wide.columns))
            selected.index.name = "date"
            selected.columns.name = "ticker"
            return selected.stack(future_stack=True).rename("adj_close").to_frame()
        missing = set(fields).difference(self.store.price_data.columns)
        if missing:
            raise ValueError(f"Price fields unavailable: {', '.join(sorted(missing))}")
        return self._slice(self.store.price_data, window).loc[:, list(fields)]

    def market_cap(self, window: int | None = None) -> pd.DataFrame:
        return self._slice(self.store.market_cap_data, window)

    def inst_flow(self, window: int | None = None) -> pd.DataFrame:
        return self._slice(self.store.inst_flow_data, window)

    def revenue(self, window: int | None = None) -> pd.DataFrame:
        return self._slice(self.store.revenue_data, window, knowledge_column="published_at")

    def universe(self) -> pd.Index:
        frame = self.store.universe_data
        if frame.empty:
            return pd.Index([], dtype="object", name="ticker")
        required = {"ticker", "market"}
        if not required.issubset(frame.columns):
            raise ValueError("Universe data must include ticker and market columns.")
        selected = frame[frame["market"].eq(self.market)].copy()
        if "eligible" in selected:
            selected = selected[selected["eligible"].fillna(False).astype(bool)]
        if "valid_from" in selected:
            valid_from = pd.to_datetime(selected["valid_from"], errors="coerce")
            selected = selected[valid_from.isna() | valid_from.le(self.asof)]
        if "valid_to" in selected:
            valid_to = pd.to_datetime(selected["valid_to"], errors="coerce")
            selected = selected[valid_to.isna() | valid_to.ge(self.asof)]
        return pd.Index(selected["ticker"].astype(str).drop_duplicates(), name="ticker")

    def industry_map(self) -> pd.Series:
        frame = self.store.universe_data
        if "industry" not in frame:
            return pd.Series(dtype="string", name="industry")
        mapping = (
            frame[frame["ticker"].astype(str).isin(self.universe())]
            .drop_duplicates("ticker", keep="last")
            .set_index("ticker")["industry"]
        )
        mapping.index = mapping.index.astype(str)
        mapping.name = "industry"
        return mapping


def wide_prices_to_panel(
    adjusted_close: pd.DataFrame,
    *,
    extra_fields: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Adapt the existing wide price cache without adding a download path."""
    fields: dict[str, pd.DataFrame] = {"adj_close": adjusted_close}
    fields.update(extra_fields or {})
    pieces = []
    for name, frame in fields.items():
        normalized = frame.copy()
        normalized.index = pd.to_datetime(normalized.index)
        normalized.index.name = "date"
        normalized.columns.name = "ticker"
        pieces.append(normalized.stack(future_stack=True).rename(name))
    return pd.concat(pieces, axis=1).sort_index()
