"""Download and cache Swiss stock prices, index prices and label snapshots.

CLI: python -m equity_fingerprint.data [--refresh] [--limit N] [--data-dir DIR]
Every step is cached on disk and skipped when its file exists unless --refresh.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd

SIX_URL = "https://www.six-group.com/sheldon/equity_issuers/v1/equity_issuers.csv"
START = "2010-01-01"
INDEX_TICKERS = ["^SSHI", "^SSMI"]
BATCH = 40
PRICE_COLS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
FIELD_MAP = {"Open": "open", "High": "high", "Low": "low", "Close": "close",
             "Adj Close": "adj_close", "Volume": "volume"}
INFO_FIELDS = {
    "name": ("longName", "shortName"),
    "sector": ("sector",),
    "industry": ("industry",),
    "shares_outstanding": ("sharesOutstanding",),
    "market_cap": ("marketCap",),
    "currency": ("currency",),
    "quote_type": ("quoteType",),
}


# --------------------------------------------------------------------------- paths
def _data_dir(data_dir=None) -> Path:
    if data_dir is None:
        data_dir = os.environ.get("EF_DATA_DIR")
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[2] / "data"
    return Path(data_dir)


# --------------------------------------------------------------------------- loaders
def load_universe(data_dir=None) -> pd.DataFrame:
    return pd.read_csv(_data_dir(data_dir) / "universe.csv")


def load_prices(data_dir=None) -> pd.DataFrame:
    df = pd.read_parquet(_data_dir(data_dir) / "raw" / "prices.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df[PRICE_COLS]


def load_index(data_dir=None) -> pd.DataFrame:
    df = pd.read_parquet(_data_dir(data_dir) / "raw" / "index.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df[PRICE_COLS]


def load_info(data_dir=None) -> pd.DataFrame:
    return pd.read_parquet(_data_dir(data_dir) / "raw" / "info.parquet")


# --------------------------------------------------------------------------- universe
def build_universe(raw: Path, out: Path, refresh: bool) -> pd.DataFrame:
    csv_path = raw / "six_issuers.csv"
    if refresh or not csv_path.exists():
        print(f"[universe] downloading {SIX_URL}")
        req = urllib.request.Request(SIX_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            csv_path.write_bytes(r.read())
    uni_path = out / "universe.csv"
    if uni_path.exists() and not refresh:
        print("[universe] cached")
        return pd.read_csv(uni_path)
    df = pd.read_csv(csv_path, sep=";", encoding="latin-1", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    prim = df["Primary listing"].astype(str).str.strip().str.upper() == "TRUE"
    df = df[prim].copy()
    df["symbol"] = df["Symbol"].astype(str).str.replace(" ", "", regex=False)
    df = df[df["symbol"].notna() & (df["symbol"] != "") & (df["symbol"] != "nan")]
    comp_col = next((c for c in df.columns if c.lower() in ("issuer", "company", "issuer name")),
                    df.columns[0])
    uni = pd.DataFrame({
        "ticker": df["symbol"] + ".SW",
        "symbol": df["symbol"],
        "company": df[comp_col].str.strip(),
        "isin": df["ISIN"].str.strip() if "ISIN" in df.columns else None,
        "country": df["Country"].str.strip() if "Country" in df.columns else None,
    }).drop_duplicates("ticker").reset_index(drop=True)
    uni.to_csv(uni_path, index=False)
    print(f"[universe] {len(uni)} primary listings")
    return uni


# --------------------------------------------------------------------------- prices
def _download(tickers: list[str]) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(tickers, start=START, end=str(date.today() + pd.Timedelta(days=1)),
                      auto_adjust=False, group_by="ticker", threads=True,
                      progress=False)
    frames = []
    if raw is None or raw.empty:
        return pd.DataFrame(columns=PRICE_COLS)
    for t in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            if t not in raw.columns.get_level_values(0):
                continue
            sub = raw[t]
        else:  # single ticker, flat columns
            sub = raw
        sub = sub.rename(columns=FIELD_MAP)
        if "close" not in sub.columns:
            continue
        sub = sub.dropna(subset=["close"])
        if sub.empty:
            continue
        sub = sub.reindex(columns=list(FIELD_MAP.values()))
        sub.index.name = "date"
        sub = sub.reset_index()
        sub["ticker"] = t
        frames.append(sub[PRICE_COLS])
    if not frames:
        return pd.DataFrame(columns=PRICE_COLS)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    return out


def download_series(tickers: list[str], label: str) -> pd.DataFrame:
    frames, failed = [], list(tickers)
    for attempt in (1, 2):
        todo = list(failed)
        failed = []
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            try:
                df = _download(chunk)
            except Exception as e:  # noqa: BLE001
                print(f"[{label}] batch error: {e!r}")
                df = pd.DataFrame(columns=PRICE_COLS)
            got = set(df["ticker"].unique())
            failed += [t for t in chunk if t not in got]
            if len(df):
                frames.append(df)
            print(f"[{label}] attempt {attempt} batch {i // BATCH + 1}: "
                  f"{len(got)}/{len(chunk)} ok")
        if not failed:
            break
    if failed:
        print(f"[{label}] no data for {len(failed)}: {failed}")
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PRICE_COLS)
    out = out.drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"])
    return out.reset_index(drop=True)


def build_prices(uni: pd.DataFrame, raw: Path, refresh: bool) -> None:
    p = raw / "prices.parquet"
    if p.exists() and not refresh:
        print("[prices] cached")
    else:
        df = download_series(uni["ticker"].tolist(), "prices")
        if df.empty:
            raise RuntimeError("price download returned no data; not caching")
        df.to_parquet(p, index=False)
        print(f"[prices] saved {len(df)} rows, {df.ticker.nunique()} tickers")
    p = raw / "index.parquet"
    if p.exists() and not refresh:
        print("[index] cached")
    else:
        df = download_series(INDEX_TICKERS, "index")
        if df.empty:
            raise RuntimeError("index download returned no data; not caching")
        df.to_parquet(p, index=False)
        print(f"[index] saved {len(df)} rows, {df.ticker.nunique()} tickers")


# --------------------------------------------------------------------------- info
def _fetch_info(t: str, path: Path) -> tuple[str, str | None]:
    import yfinance as yf

    last = None
    for _ in range(2):
        try:
            info = yf.Ticker(t).info
            if isinstance(info, dict) and len(info) > 1:
                info = dict(info)
                info["_snapshot_date"] = str(date.today())
                path.write_text(json.dumps(info, default=str))
                return t, None
            last = "empty info"
        except Exception as e:  # noqa: BLE001
            last = repr(e)
    return t, last


def build_info(uni: pd.DataFrame, raw: Path, refresh: bool, limit: int | None) -> dict:
    d = raw / "info"
    d.mkdir(parents=True, exist_ok=True)
    errors: dict[str, str] = {}
    todo = [t for t in uni["ticker"] if refresh or not (d / f"{t}.json").exists()]
    if limit is not None:
        todo = todo[:limit]
    print(f"[info] fetching {len(todo)} tickers")
    if todo:
        with ThreadPoolExecutor(max_workers=8) as ex:
            for k, (t, err) in enumerate(ex.map(lambda t: _fetch_info(t, d / f"{t}.json"), todo), 1):
                if err:
                    errors[t] = err
                if k % 25 == 0:
                    print(f"[info] {k}/{len(todo)}")
    rows = []
    for t in uni["ticker"]:
        f = d / f"{t}.json"
        if not f.exists():
            continue
        info = json.loads(f.read_text())
        row = {"ticker": t}
        for k, keys in INFO_FIELDS.items():
            row[k] = next((info[x] for x in keys if info.get(x) not in (None, "")), None)
        row["snapshot_date"] = info.get("_snapshot_date")
        rows.append(row)
    df = pd.DataFrame(rows, columns=["ticker", "name", "sector", "industry", "shares_outstanding",
                                     "market_cap", "currency", "quote_type", "snapshot_date"])
    for c in ("shares_outstanding", "market_cap"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.to_parquet(raw / "info.parquet", index=False)
    print(f"[info] saved {len(df)} rows ({df.sector.notna().sum()} with sector)")
    return errors


# --------------------------------------------------------------------------- manifest
def write_manifest(uni: pd.DataFrame, root: Path, info_errors: dict) -> None:
    raw = root / "raw"
    prices = pd.read_parquet(raw / "prices.parquet")
    index = pd.read_parquet(raw / "index.parquet")
    info = pd.read_parquet(raw / "info.parquet")
    counts = prices.groupby("ticker").size()
    with_prices = set(counts.index)
    failed = {}
    for t in uni["ticker"]:
        reasons = []
        if t not in with_prices:
            reasons.append("no price data")
        elif counts[t] < 500:
            reasons.append(f"only {counts[t]} price rows")
        if t in info_errors:
            reasons.append(f"info: {info_errors[t]}")
        elif t not in set(info["ticker"]):
            reasons.append("info not fetched")
        elif info.loc[info.ticker == t, "sector"].isna().all():
            reasons.append("no sector")
        if reasons:
            failed[t] = "; ".join(reasons)
    manifest = {
        "run_date": str(date.today()),
        "n_universe": int(len(uni)),
        "n_with_prices": int(len(with_prices)),
        "n_with_>=500_rows": int((counts >= 500).sum()),
        "price_date_range": [str(prices.date.min())[:10], str(prices.date.max())[:10]],
        "index_date_ranges": {t: [str(g.date.min())[:10], str(g.date.max())[:10]]
                              for t, g in index.groupby("ticker")},
        "n_with_sector": int(info.sector.notna().sum()),
        "sector_counts": {k: int(v) for k, v in info.sector.value_counts().items()},
        "n_with_shares": int(info.shares_outstanding.notna().sum()),
        "failed_tickers": failed,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] written; {len(failed)} tickers with issues")


# --------------------------------------------------------------------------- main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="re-download even if cached")
    ap.add_argument("--limit", type=int, default=None,
                    help="max number of tickers to fetch .info for in this run (resume later)")
    ap.add_argument("--data-dir", default=None)
    a = ap.parse_args(argv)
    # threaded yfinance downloads leak sockets; macOS default soft limit is 256
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = 8192 if hard == resource.RLIM_INFINITY else min(8192, hard)
    if soft < want:
        resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    root = _data_dir(a.data_dir)
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    uni = build_universe(raw, root, a.refresh)
    build_prices(uni, raw, a.refresh)
    errors = build_info(uni, raw, a.refresh, a.limit)
    write_manifest(uni, root, errors)


if __name__ == "__main__":
    main()
