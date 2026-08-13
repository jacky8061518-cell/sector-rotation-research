"""Adapters for existing Taiwan caches and forward point-in-time snapshots."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

TWSE_REVENUE_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap05_L"
TPEX_REVENUE_URL = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap05_O"


def _roc_date(value: object) -> pd.Timestamp:
    digits = str(value).strip()
    if len(digits) != 7:
        return pd.NaT
    return pd.Timestamp(year=int(digits[:3]) + 1911, month=int(digits[3:5]), day=int(digits[5:]))


def _roc_period(value: object) -> str:
    digits = str(value).strip()
    if len(digits) != 5:
        return ""
    return f"{int(digits[:3]) + 1911:04d}-{digits[3:]}"


def fetch_current_monthly_revenue(timeout: int = 30) -> pd.DataFrame:
    """Fetch the current official TWSE/TPEx monthly-revenue publications."""
    frames = []
    for market, suffix, url in (("上市", ".TW", TWSE_REVENUE_URL), ("上櫃", ".TWO", TPEX_REVENUE_URL)):
        request = Request(url, headers={"User-Agent": "sector-rotation-research/1.0"})
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
        raw = pd.DataFrame(payload)
        if raw.empty:
            continue
        frame = pd.DataFrame(
            {
                "date": raw["出表日期"].map(_roc_date),
                "ticker": raw["公司代號"].astype(str).str.strip().add(suffix),
                "period": raw["資料年月"].map(_roc_period),
                "name": raw["公司名稱"].astype(str).str.strip(),
                "market": market,
                "revenue": pd.to_numeric(raw["營業收入-當月營收"], errors="coerce") * 1000,
                "revenue_mom": pd.to_numeric(raw["營業收入-上月比較增減(%)"], errors="coerce") / 100,
                "revenue_yoy": pd.to_numeric(raw["營業收入-去年同月增減(%)"], errors="coerce") / 100,
            }
        )
        frame["published_at"] = frame["date"].add(pd.Timedelta(hours=18))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).dropna(subset=["date", "ticker", "period"])


def update_monthly_revenue_cache(path: Path) -> pd.DataFrame:
    """Append official snapshots, keeping the earliest date each value became visible."""
    fresh = fetch_current_monthly_revenue()
    cached = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([cached, fresh], ignore_index=True)
    combined["published_at"] = pd.to_datetime(combined["published_at"])
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values("published_at").drop_duplicates(["ticker", "period"], keep="first")
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False)
    return combined


def _numeric_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    for candidate in candidates:
        if candidate in frame:
            return pd.to_numeric(frame[candidate], errors="coerce")
    return pd.Series(float("nan"), index=frame.index)


def fetch_current_financials(timeout: int = 30) -> pd.DataFrame:
    """Fetch official current-quarter statements with conservative publication dates."""
    frames = []
    endpoints = (
        ("上市", ".TW", "https://openapi.twse.com.tw/v1/opendata", "L"),
        ("上櫃", ".TWO", "https://www.tpex.org.tw/openapi/v1", "O"),
    )
    jobs = []
    for market, suffix, base, market_code in endpoints:
        # General-industry statements share one stable schema. Financial,
        # insurance and securities statements require separate accounting
        # mappings and remain unavailable rather than being coerced incorrectly.
        for industry_code in ("ci",):
            income_url = f"{base}/mopsfin_t187ap06_{market_code}_{industry_code}"
            balance_url = f"{base}/mopsfin_t187ap07_{market_code}_{industry_code}"
            if market_code == "L":
                income_url = income_url.replace("/mopsfin_", "/")
                balance_url = balance_url.replace("/mopsfin_", "/")
            jobs.append((market, suffix, income_url, balance_url))

    def fetch_pair(job: tuple[str, str, str, str]) -> tuple[str, str, pd.DataFrame, pd.DataFrame]:
        market, suffix, income_url, balance_url = job
        request_headers = {"User-Agent": "sector-rotation-research/1.0"}
        with urlopen(Request(income_url, headers=request_headers), timeout=timeout) as response:
            income = pd.DataFrame(json.load(response))
        with urlopen(Request(balance_url, headers=request_headers), timeout=timeout) as response:
            balance = pd.DataFrame(json.load(response))
        return market, suffix, income, balance

    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = [executor.submit(fetch_pair, job) for job in jobs]
        for future in futures:
            try:
                market, suffix, income, balance = future.result()
            except (OSError, TimeoutError, ValueError, json.JSONDecodeError):
                continue
            keys = ["年度", "季別", "公司代號"]
            if income.empty or balance.empty or not set(keys).issubset(income) or not set(keys).issubset(balance):
                continue
            merged = income.merge(balance, on=keys, suffixes=("_income", "_balance"))
            output_date = merged.get("出表日期_income")
            if output_date is None:
                output_date = merged.get("出表日期_balance", merged.get("出表日期"))
            if output_date is None:
                continue
            year = pd.to_numeric(merged["年度"], errors="coerce").add(1911).astype("Int64")
            quarter = pd.to_numeric(merged["季別"], errors="coerce").astype("Int64")
            frame = pd.DataFrame(
                {
                    "date": output_date.map(_roc_date),
                    "ticker": merged["公司代號"].astype(str).str.strip().add(suffix),
                    "period": year.astype(str).add("Q").add(quarter.astype(str)),
                    "market": market,
                    "eps": _numeric_column(merged, ("基本每股盈餘（元）", "基本每股盈餘")),
                    "sales": _numeric_column(merged, ("營業收入", "淨收益", "收益")) * 1000,
                    "gross_profit": _numeric_column(merged, ("營業毛利（毛損）淨額", "營業毛利（毛損）")) * 1000,
                    "net_income": _numeric_column(merged, ("淨利（淨損）歸屬於母公司業主", "本期淨利（淨損）")) * 1000,
                    "total_assets": _numeric_column(merged, ("資產總計",)) * 1000,
                    "debt": _numeric_column(merged, ("負債總計",)) * 1000,
                    "equity": _numeric_column(merged, ("權益總計", "權益總額")) * 1000,
                    "book_value_per_share": _numeric_column(merged, ("每股參考淨值", "每股淨值")),
                }
            )
            frame["published_at"] = frame["date"].add(pd.Timedelta(hours=18))
            frame["roe"] = frame["net_income"].div(frame["equity"])
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).dropna(subset=["date", "ticker", "period"])


def update_financial_cache(path: Path) -> pd.DataFrame:
    fresh = fetch_current_financials()
    cached = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([cached, fresh], ignore_index=True)
    if combined.empty:
        return combined
    combined["date"] = pd.to_datetime(combined["date"])
    combined["published_at"] = pd.to_datetime(combined["published_at"])
    combined = combined.sort_values("published_at").drop_duplicates(["ticker", "period"], keep="first")
    prior = combined[["ticker", "period", "eps", "sales"]].copy()
    year_quarter = prior["period"].str.extract(r"(?P<year>\d{4})Q(?P<quarter>\d)")
    prior["period"] = (
        (pd.to_numeric(year_quarter["year"]) + 1).astype("Int64").astype(str).add("Q").add(year_quarter["quarter"])
    )
    prior = prior.rename(columns={"eps": "eps_prior", "sales": "sales_prior"})
    combined = combined.merge(prior, on=["ticker", "period"], how="left")
    combined["eps_growth_yoy"] = combined["eps"].div(combined["eps_prior"]).sub(1)
    combined["sales_growth_yoy"] = combined["sales"].div(combined["sales_prior"]).sub(1)
    combined = combined.drop(columns=["eps_prior", "sales_prior"])
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False)
    return combined


def save_universe_snapshot(master: pd.DataFrame, directory: Path, asof: pd.Timestamp | None = None) -> Path:
    timestamp = pd.Timestamp(asof or datetime.now().astimezone()).normalize()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{timestamp:%Y-%m-%d}.parquet"
    snapshot = master.copy()
    snapshot["snapshot_date"] = timestamp
    snapshot.to_parquet(path, index=False)
    return path


def normalize_institutional_flows(flows: pd.DataFrame) -> pd.DataFrame:
    if flows.empty:
        return pd.DataFrame(
            columns=["date", "ticker", "foreign_net_shares", "trust_net_shares", "dealer_net_shares", "total_net_shares"]
        )
    return flows.rename(
        columns={
            "Date": "date",
            "Ticker": "ticker",
            "Foreign net shares": "foreign_net_shares",
            "Trust net shares": "trust_net_shares",
            "Dealer net shares": "dealer_net_shares",
            "Total net shares": "total_net_shares",
        }
    )[["date", "ticker", "foreign_net_shares", "trust_net_shares", "dealer_net_shares", "total_net_shares"]]


def load_optional_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame
