"""Extra market series for the v9 vol_1m features: VIX and Euro Stoxx 50 daily closes.

``download`` fetches ``TICKERS`` with ``data.download_series`` (same long format as ``index.parquet``) and writes
``<data>/raw/extra_index.parquet``; ``load_extra`` returns the closes as a (date x ticker) frame. The data
directory follows ``data._data_dir`` (argument, else ``EF_DATA_DIR``, else ``<repo>/data``).

Run: env -u PYTHONPATH PYTHONPATH=src python -m equity_fingerprint.extra_market
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import data

TICKERS = ["^VIX", "^STOXX50E"]
FILE = "extra_index.parquet"


def _path(data_dir=None) -> Path:
    return data._data_dir(data_dir) / "raw" / FILE


def download(data_dir=None) -> pd.DataFrame:
    """Download ``TICKERS`` and write ``<data>/raw/extra_index.parquet``; returns the long frame."""
    df = data.download_series(TICKERS, "extra")
    path = _path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"[extra] wrote {len(df)} rows for {sorted(df['ticker'].unique())} -> {path}")
    return df


def load_extra(data_dir=None) -> pd.DataFrame:
    """(date x ticker) close prices, sorted by date, duplicate (ticker, date) rows dropped."""
    path = _path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python -m equity_fingerprint.extra_market` first "
                                f"(downloads {', '.join(TICKERS)})")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    return df.pivot(index="date", columns="ticker", values="close").sort_index()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=None)
    a = ap.parse_args(argv)
    download(a.data_dir)


if __name__ == "__main__":
    main()
