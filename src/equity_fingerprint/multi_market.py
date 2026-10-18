"""v10 data layer: foreign (European) universe, FX to CHF, per-market panels, issuer groups.

Default regions: ``REGIONS = ("EU",)`` (user decision 2026-10-18: US stocks did not help the Swiss rows and
were removed from every v10 result). The selection code stays region-generic (``regions`` argument, the US
floors below) so the synthetic tests can still exercise it.

Foreign inputs come from the v4 pretraining cache (``EF_PRETRAIN_DIR``, default ``data/pretrain``):
``universe.csv``, ``raw/prices.parquet``, ``raw/index.parquet``, ``raw/info.parquet``. A stock's DOMICILE is
the field ``country`` of ``raw/info/<ticker>.json`` (read-only; no other json field, in particular no
company description, is used). FX comes from ``<data>/raw/fx.parquet`` (``--download-fx``).

Steps (design.md, group 1):
  * ``chf_rate`` local->CHF factors, forward filled from earlier dates only and shifted by ``lag`` trading
    days (default 1: Yahoo FX closes after the SIX / US close); GBp = GBP / 100; ``.L`` lines whose
    close x shares is ~100x the market cap are treated as pence whatever the currency field says (logged).
  * ``select_foreign``: equity, sector known, not ``.SW``, not a Swiss name, domiciled in the listing
    country, >= 504 valid returns, median CHF close >= 5, median CHF traded value above the region floor.
  * ``dedupe``: normalised-name pass and weekly CHF return correlation (>= 0.90 over >= 104 weeks) against
    Swiss and other foreign lines; Swiss/foreign -> drop foreign, US/EU -> drop US, else keep the line with
    the higher traded value.
  * ``build_market_panel`` (``features.panel_from_frames`` with CHF prices and the local index),
    ``weekly_resid`` (two-factor), ``load_markets``, ``stack_returns``, ``issuer_groups``.

CLI: python -m equity_fingerprint.multi_market [--download-fx] [--summary] [--data-dir DIR] [--pretrain-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from .features import ARTIFACT_LOGRET, Panel, build_panel, panel_from_frames
from .pretrain_data import INDEX_FALLBACK, MIN_INDEX_ROWS, STOXX_COUNTRY, SWISS_NAMES, _pretrain_dir
from .static_cv import pairwise_corr, resid_on

FX_TICKERS = {"USD": "USDCHF=X", "EUR": "EURCHF=X", "GBP": "GBPCHF=X", "NOK": "NOKCHF=X",
              "SEK": "SEKCHF=X", "DKK": "DKKCHF=X"}
FX_FILE = "fx.parquet"
REGIONS = ("EU",)
PENCE_RATIO = (50.0, 200.0)       # close x shares / market cap in this range => the line is quoted in pence
MIN_RETURNS = 504
# price floor against US penny stocks only: many EU large caps trade below CHF 5 (UK lines in pence)
MIN_CLOSE_CHF = {"US": 5.0, "EU": 0.0}
MIN_TV_CHF = {"US": 1_000_000.0, "EU": 100_000.0}
TV_WINDOW = 252
DEDUPE_CORR = 0.90
DEDUPE_MIN_WEEKS = 104
WEEKLY_MIN_DAYS = 3
LISTING_COUNTRY = {code: name for name, (code, _, _) in STOXX_COUNTRY.items()}
LISTING_COUNTRY["US"] = "United States"
PRICE_COLS_LIGHT = ["date", "ticker", "close", "adj_close", "volume"]
NAME_STOP = {"ag", "sa", "plc", "nv", "se", "inc", "ltd", "corp", "co", "adr", "ads", "holding", "holdings",
             "group"}


# --------------------------------------------------------------------------- FX
def _fx_path(data_dir=None) -> Path:
    return data._data_dir(data_dir) / "raw" / FX_FILE


def download_fx(data_dir=None) -> pd.DataFrame:
    """Download the CHF cross rates of ``FX_TICKERS`` and write ``<data>/raw/fx.parquet`` (long format)."""
    df = data.download_series(list(FX_TICKERS.values()), "fx")
    path = _fx_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"[fx] wrote {len(df)} rows for {sorted(df['ticker'].unique())} -> {path}")
    return df


def load_fx(data_dir=None) -> pd.DataFrame:
    """(date x FX ticker) close rates, sorted by date, duplicate (ticker, date) rows dropped."""
    path = _fx_path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python -m equity_fingerprint.multi_market --download-fx` "
                                f"first (downloads {', '.join(FX_TICKERS.values())})")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    return df.pivot(index="date", columns="ticker", values="close").sort_index()


def chf_rate(fx: pd.DataFrame, currency: str, calendar, lag: int = 1) -> pd.Series:
    """Local->CHF factor on ``calendar``: forward fill of earlier FX dates only (no back fill), then shifted
    by ``lag`` calendar rows (value at row i = rate known at row i - lag). CHF -> 1, GBp -> GBP / 100,
    unknown currency -> NaN."""
    cal = pd.DatetimeIndex(calendar)
    if currency == "CHF":
        return pd.Series(1.0, index=cal)
    scale = 1.0
    if currency in ("GBp", "GBX"):
        currency, scale = "GBP", 0.01
    col = FX_TICKERS.get(currency)
    if col is None or col not in fx.columns:
        return pd.Series(np.nan, index=cal)
    s = fx[col].dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    r = s.reindex(cal.union(s.index)).ffill().reindex(cal) * scale
    return r.shift(lag) if lag else r


def rate_frame(fx: pd.DataFrame, currency_of: pd.Series, calendar, lag: int = 1) -> pd.DataFrame:
    """(calendar x ticker) local->CHF factors from a ticker -> currency series."""
    cols = {}
    for cur, tick in currency_of.groupby(currency_of.fillna("?")):
        r = chf_rate(fx, cur, calendar, lag)
        for t in tick.index:
            cols[t] = r
    out = pd.DataFrame(cols, index=pd.DatetimeIndex(calendar))
    return out[list(currency_of.index)] if len(cols) else out


def effective_currency(table: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Quote currency per ticker with the pence override.

    ``table`` has ticker, currency, last_close (local), shares_outstanding, market_cap. A ``.L`` line whose
    close x shares is 50-200x the market cap is quoted in pence (GBp) whatever ``currency`` says. Returns
    the currency series and a log of every override (ticker, currency_field, ratio)."""
    cur = table.set_index("ticker")["currency"].astype(object).copy()
    cur = cur.where(~cur.isin(["GBX", "GBp"]), "GBp")
    log = []
    for r in table.itertuples():
        if not str(r.ticker).endswith(".L"):
            continue
        ok = all(np.isfinite(x) and x > 0 for x in (r.last_close, r.shares_outstanding, r.market_cap))
        if not ok:
            continue
        ratio = r.last_close * r.shares_outstanding / r.market_cap
        if PENCE_RATIO[0] <= ratio <= PENCE_RATIO[1] and cur[r.ticker] != "GBp":
            log.append(dict(ticker=r.ticker, currency_field=r.currency, ratio=float(ratio)))
            cur[r.ticker] = "GBp"
    return cur, pd.DataFrame(log, columns=["ticker", "currency_field", "ratio"])


# --------------------------------------------------------------------------- regions, names
def region_of(country) -> str:
    """CH, US or EU from a country code or name (every other country is EU)."""
    c = str(country).strip()
    if c in ("CH", "Switzerland"):
        return "CH"
    if c in ("US", "United States"):
        return "US"
    return "EU"


def norm_name(name) -> str:
    """Lower case, punctuation and legal suffixes (ag sa plc nv se inc ltd corp co adr ads holding(s) group)
    and share-class markers ("class A", "series B") removed."""
    s = str(name).lower().replace(".", "")
    s = re.sub(r"\b(class|series)\s+[a-z0-9]\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(w for w in s.split() if w not in NAME_STOP)


def issuer_groups(tickers, universe_csv, names=None) -> np.ndarray:
    """Integer issuer id per ticker: Swiss lines by ``company`` of ``universe_csv`` (share lines of one
    company share an id), foreign lines by normalised name (``names``: ticker -> name); a ticker with
    neither gets its own group."""
    uni = pd.read_csv(universe_csv) if isinstance(universe_csv, (str, Path)) else universe_csv
    company = dict(zip(uni["ticker"], uni["company"]))
    names = {} if names is None else dict(names)
    keys = []
    for t in tickers:
        if t in company and pd.notna(company[t]):
            keys.append("ch:" + str(company[t]).strip())
        elif t in names and norm_name(names[t]):
            keys.append("fx:" + norm_name(names[t]))
        else:
            keys.append("own:" + str(t))
    codes = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    return np.array([codes[k] for k in keys], dtype=int)


# --------------------------------------------------------------------------- readers
def read_domicile(tickers, pretrain_dir=None) -> pd.Series:
    """Ticker -> ``country`` field of ``raw/info/<ticker>.json`` (NaN when the file or field is missing)."""
    d = _pretrain_dir(pretrain_dir) / "raw" / "info"
    out = {}
    for t in tickers:
        f = d / f"{t}.json"
        out[t] = np.nan
        if f.exists():
            c = json.loads(f.read_text()).get("country")
            if c:
                out[t] = c
    return pd.Series(out, dtype=object)


def read_prices(path, tickers, columns=PRICE_COLS_LIGHT) -> pd.DataFrame:
    """Long price frame filtered to ``tickers`` (pyarrow filter, only ``columns``)."""
    tickers = sorted(set(tickers))
    df = pd.read_parquet(path, columns=list(columns), filters=[("ticker", "in", tickers)])
    df["date"] = pd.to_datetime(df["date"])
    return df


def _pretrain_files(pretrain_dir=None):
    root = _pretrain_dir(pretrain_dir)
    return root, root / "raw" / "prices.parquet", root / "raw" / "index.parquet", root / "raw" / "info.parquet"


# --------------------------------------------------------------------------- selection
def _tail_median(wide: pd.DataFrame, n: int = TV_WINDOW) -> pd.Series:
    return pd.Series({c: wide[c].dropna().iloc[-n:].median() for c in wide.columns}, dtype=float)


def select_foreign(universe: pd.DataFrame, info: pd.DataFrame, domicile: pd.Series, prices, fx: pd.DataFrame,
                   regions=REGIONS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Foreign stocks that may be used, plus the count after each filter.

    ``prices``: long frame (date, ticker, close, adj_close, volume) or a callable ``tickers -> frame`` that is
    only called for the tickers that passed the metadata filters. Returns (table, counts); counts has one row
    per filter and the columns all / EU / US. The full-sample filters suit the static sets (2026 snapshot
    labels); forward rows use ``eligible_rows``."""
    u = universe.drop_duplicates("ticker").set_index("ticker")
    u = u.assign(region=[region_of(c) for c in u["country"]])
    inf = info.drop_duplicates("ticker").set_index("ticker")
    steps: list[tuple[str, pd.Index]] = []

    def step(name, idx):
        steps.append((name, pd.Index(idx)))
        return pd.Index(idx)

    keep = step("universe", u.index[u["region"].isin(regions)])
    keep = step("in_info", keep[keep.isin(inf.index)])
    keep = step("equity", keep[(inf.loc[keep, "quote_type"] == "EQUITY").to_numpy()])
    keep = step("sector", keep[inf.loc[keep, "sector"].notna().to_numpy()])
    keep = step("not_sw", keep[~keep.str.endswith(".SW")])
    swissy = [bool(SWISS_NAMES.search(str(u.loc[t, "name"]))) or bool(SWISS_NAMES.search(str(inf.loc[t, "name"])))
              for t in keep]
    keep = step("not_swiss_name", keep[~np.array(swissy, dtype=bool)] if len(keep) else keep)
    want = u.loc[keep, "country"].map(LISTING_COUNTRY)
    dom = domicile.reindex(keep)
    keep = step("domicile", keep[((dom == want) & want.notna()).to_numpy()])

    px = prices(list(keep)) if callable(prices) else prices[prices["ticker"].isin(keep)]
    px = px[px["ticker"].isin(keep)]
    adj = px.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    close = px.pivot(index="date", columns="ticker", values="close").sort_index()
    vol = px.pivot(index="date", columns="ticker", values="volume").sort_index()
    nret = np.log(adj).diff().notna().sum()
    keep = step("returns_504", keep[(nret.reindex(keep).fillna(0) >= MIN_RETURNS).to_numpy()])

    cols = [c for c in keep if c in close.columns]
    last_close = pd.Series({c: close[c].dropna().iloc[-1] if close[c].notna().any() else np.nan for c in cols})
    tab = pd.DataFrame({"ticker": keep})
    tab["currency"] = inf.loc[keep, "currency"].to_numpy()
    tab["last_close"] = last_close.reindex(keep).to_numpy()
    tab["shares_outstanding"] = inf.loc[keep, "shares_outstanding"].to_numpy(float)
    tab["market_cap"] = inf.loc[keep, "market_cap"].to_numpy(float)
    cur, pence_log = effective_currency(tab)
    rate = rate_frame(fx, cur.reindex(keep), close.index, lag=0)
    close_chf = close[list(keep)] * rate
    tv_chf = (close_chf * vol[list(keep)].fillna(0.0))
    med_close, med_tv = _tail_median(close_chf), _tail_median(tv_chf)
    pfloor = u.loc[keep, "region"].map(MIN_CLOSE_CHF)
    keep = step("price_chf", keep[(med_close.reindex(keep) >= pfloor).to_numpy()])
    floor = u.loc[keep, "region"].map(MIN_TV_CHF)
    keep = step("traded_value", keep[(med_tv.reindex(keep) >= floor).to_numpy()])

    table = pd.DataFrame({
        "ticker": keep, "name": u.loc[keep, "name"].to_numpy(), "country": u.loc[keep, "country"].to_numpy(),
        "domicile": domicile.reindex(keep).to_numpy(), "region": u.loc[keep, "region"].to_numpy(),
        "sector": inf.loc[keep, "sector"].to_numpy(), "currency": cur.reindex(keep).to_numpy(),
        "market_index": u.loc[keep, "market_index"].to_numpy(),
        "shares_outstanding": inf.loc[keep, "shares_outstanding"].to_numpy(float),
        "market_cap": inf.loc[keep, "market_cap"].to_numpy(float),
        "median_close_chf": med_close.reindex(keep).to_numpy(), "median_tv_chf": med_tv.reindex(keep).to_numpy(),
        "n_returns": nret.reindex(keep).to_numpy(),
    }).reset_index(drop=True)
    reg = u["region"]
    counts = pd.DataFrame({"filter": [n for n, _ in steps],
                           "all": [len(i) for _, i in steps],
                           "EU": [int((reg.reindex(i) == "EU").sum()) for _, i in steps],
                           "US": [int((reg.reindex(i) == "US").sum()) for _, i in steps]})
    table.attrs["counts"] = counts
    table.attrs["pence_log"] = pence_log
    return table, counts


def eligible_rows(panel: Panel, pos, min_tv_chf: float) -> pd.DataFrame:
    """Forward-row eligibility: True where the trailing ``TV_WINDOW``-day median CHF traded value at the date
    clears ``min_tv_chf`` (uses rows up to and including that date only). Index = panel dates at ``pos``."""
    pos = np.asarray(pos)
    tv = panel.tv.to_numpy(float)
    out = np.zeros((len(pos), tv.shape[1]), dtype=bool)
    for k, i in enumerate(pos):
        if i + 1 < TV_WINDOW:
            continue
        with np.errstate(all="ignore"):
            med = np.nanmedian(tv[i + 1 - TV_WINDOW:i + 1], axis=0)
        out[k] = med >= min_tv_chf
    return pd.DataFrame(out, index=panel.calendar[pos], columns=panel.tickers)


# --------------------------------------------------------------------------- dedupe
def weekly_log_returns(prices: pd.DataFrame, currency_of: pd.Series, fx: pd.DataFrame, lag: int = 0) -> pd.DataFrame:
    """Weekly (W-FRI) log returns of CHF-converted adjusted closes (``chf_rate`` per currency); a weekly
    |return| above the artifact limit is NaN."""
    adj = prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    adj = adj[[c for c in currency_of.index if c in adj.columns]]
    adj = adj * rate_frame(fx, currency_of.reindex(adj.columns), adj.index, lag)
    w = np.log(adj.resample("W-FRI").last()).diff()
    return w.mask(w.abs() > ARTIFACT_LOGRET).astype("float32")


def dedupe(table: pd.DataFrame, wret: pd.DataFrame, swiss_names: pd.Series, min_weeks: int = DEDUPE_MIN_WEEKS,
           threshold: float = DEDUPE_CORR) -> tuple[list[str], pd.DataFrame]:
    """Remove cross-listings of Swiss or other kept stocks from the foreign ``table``.

    ``wret``: weekly CHF log returns, columns = foreign tickers and the Swiss tickers of ``swiss_names``
    (ticker -> company). Pass 1: equal normalised names (against Swiss company names and among foreign
    names). Pass 2: weekly correlation >= ``threshold`` over >= ``min_weeks`` common weeks. A foreign/Swiss
    pair drops the foreign line, a US/EU pair the US line, otherwise the line with the lower median traded
    value goes. Returns (kept foreign tickers, drop log: ticker, reason, partner, corr)."""
    info = table.set_index("ticker")
    region, tv = info["region"], info["median_tv_chf"]
    dropped: dict[str, dict] = {}

    def drop(t, reason, partner, corr=np.nan):
        dropped.setdefault(t, dict(ticker=t, reason=reason, partner=partner, corr=corr))

    def loser(a, b):
        """Foreign pair -> the line to drop."""
        if region[a] != region[b]:
            return a if region[a] == "US" else b
        return a if (tv[a], a) < (tv[b], b) else b

    swiss_by_name = {}
    for t, n in swiss_names.items():
        swiss_by_name.setdefault(norm_name(n), t)
    groups: dict[str, list[str]] = {}
    for t in info.index:
        groups.setdefault(norm_name(info.loc[t, "name"]), []).append(t)
    for key, members in groups.items():
        if not key:
            continue
        if key in swiss_by_name:
            for t in members:
                drop(t, "name_swiss", swiss_by_name[key])
            continue
        alive = list(members)
        while len(alive) > 1:
            a, b = alive[0], alive[1]
            lo = loser(a, b)
            hi = b if lo == a else a
            drop(lo, "name_foreign", hi)
            alive = [t for t in alive if t != lo]

    cols = [c for c in wret.columns if c in info.index and c not in dropped] + \
           [c for c in wret.columns if c in swiss_names.index]
    cols = list(dict.fromkeys(cols))
    if len(cols) > 1:
        C = pairwise_corr(wret[cols].astype(float), min_common=min_weeks)
        ci = C.to_numpy()
        isf = np.array([c in info.index for c in cols])
        hits = np.argwhere(np.triu(np.nan_to_num(ci, nan=-1.0) >= threshold, k=1))
        hits = [(ci[i, j], cols[i], cols[j]) for i, j in hits if isf[i] or isf[j]]
        for c, a, b in sorted(hits, key=lambda x: (-x[0], x[1], x[2])):
            fa, fb = a in info.index, b in info.index
            if fa and fb:
                if a in dropped or b in dropped:
                    continue
                lo = loser(a, b)
                drop(lo, "corr", b if lo == a else a, float(c))
            else:
                f, s = (a, b) if fa else (b, a)
                if f not in dropped:
                    drop(f, "corr_swiss", s, float(c))
    log = pd.DataFrame(list(dropped.values()), columns=["ticker", "reason", "partner", "corr"])
    kept = [t for t in table["ticker"] if t not in dropped]
    return kept, log


# --------------------------------------------------------------------------- panels
def build_market_panel(prices: pd.DataFrame, index: pd.DataFrame, info: pd.DataFrame, market: str,
                       rate: pd.DataFrame | None, region: str) -> Panel:
    """Panel of one market: all tickers of ``prices``, CHF prices via ``rate``, ``mkt`` = ``market`` index.
    ``region`` is stored as ``panel.region``."""
    prices = prices.assign(ticker=prices["ticker"].astype(str))
    p = panel_from_frames(prices, index, info, chf_only=False, market=market, rate=rate)
    p.region = region
    p.market_index = market
    return p


def _weekly(x):
    return x.resample("W-FRI").sum(min_count=WEEKLY_MIN_DAYS)


def weekly_resid(panels: dict) -> pd.DataFrame:
    """Weekly residuals of every stock of ``panels`` ({key: Panel}) on two factors: its local index and the
    equal-weight mean weekly return of all stocks of all panels. Weekly return = sum of the daily log
    returns of a W-FRI week (weeks with < 3 valid days are NaN), on the union weekly calendar."""
    wk = {k: _weekly(p.ret) for k, p in panels.items()}
    wm = {k: _weekly(p.mkt) for k, p in panels.items()}
    cal = None
    for w in wk.values():
        cal = w.index if cal is None else cal.union(w.index)
    allw = pd.concat([w.reindex(cal) for w in wk.values()], axis=1)
    with np.errstate(all="ignore"):
        ew = allw.mean(axis=1).to_numpy(float)
    parts = []
    for k, w in wk.items():
        F = np.column_stack([wm[k].reindex(cal).to_numpy(float), ew])
        parts.append(resid_on(w.reindex(cal).astype(float), F))
    return pd.concat(parts, axis=1)


def stack_returns(panels: dict, freq: str = "D") -> pd.DataFrame:
    """Returns of all stocks of ``panels`` ({key: Panel}) on the union calendar (``freq`` D daily, W weekly
    sums with at least 3 valid days), float32."""
    if freq not in ("D", "W"):
        raise ValueError("freq must be 'D' or 'W'")
    frames = [p.ret if freq == "D" else _weekly(p.ret) for p in panels.values()]
    cal = None
    for f in frames:
        cal = f.index if cal is None else cal.union(f.index)
    return pd.concat([f.reindex(cal) for f in frames], axis=1).astype("float32")


def _index_for(index: pd.DataFrame, market: str) -> str:
    rows = index.groupby("ticker").size()
    alt = INDEX_FALLBACK.get(market)
    if rows.get(market, 0) < MIN_INDEX_ROWS and alt and rows.get(alt, 0) > rows.get(market, 0):
        return alt
    return market


def load_markets(data_dir=None, pretrain_dir=None, regions=REGIONS):
    """(swiss_panel, {market_index: panel}, foreign_table, drop_log) for the kept foreign universe.

    Reads ``data_dir`` (Swiss cache, fx) and ``pretrain_dir``; the foreign price file is read with a ticker
    filter, one market at a time. ``foreign_table.attrs`` holds ``counts`` (per filter) and ``pence_log``."""
    swiss = build_panel(data_dir)
    fx = load_fx(data_dir)
    root, ppath, ipath, infopath = _pretrain_files(pretrain_dir)
    uni, info = pd.read_csv(root / "universe.csv"), pd.read_parquet(infopath)
    dom = read_domicile(uni["ticker"], pretrain_dir)
    table, counts = select_foreign(uni, info, dom, lambda ts: read_prices(ppath, ts), fx, regions)
    pence_log = table.attrs["pence_log"]
    cur = table.set_index("ticker")["currency"]
    swiss_uni = data.load_universe(data_dir).set_index("ticker")["company"]
    swiss_names = swiss_uni.reindex(swiss.tickers).dropna()
    fpx = read_prices(ppath, table["ticker"], ["date", "ticker", "adj_close"])
    wf = weekly_log_returns(fpx, cur, fx)
    ws = np.log(swiss.adj.resample("W-FRI").last()).diff().astype("float32")
    ws = ws.mask(ws.abs() > ARTIFACT_LOGRET)
    del fpx
    kept, drop_log = dedupe(table, pd.concat([wf, ws[swiss_names.index]], axis=1), swiss_names)
    table = table[table["ticker"].isin(kept)].reset_index(drop=True)
    index_all = pd.read_parquet(ipath)
    index_all["date"] = pd.to_datetime(index_all["date"])
    table["market_index"] = [_index_for(index_all, m) for m in table["market_index"]]
    panels = {}
    for mkt, grp in table.groupby("market_index"):
        px = read_prices(ppath, grp["ticker"], data.PRICE_COLS)
        px = px.assign(ticker=px["ticker"].astype(str))
        rate = rate_frame(fx, grp.set_index("ticker")["currency"], sorted(px["date"].unique()), lag=1)
        sub_info = info[info["ticker"].isin(grp["ticker"])]
        panels[mkt] = build_market_panel(px, index_all[index_all["ticker"] == mkt], sub_info, mkt, rate,
                                         grp["region"].iloc[0])
        del px
    table.attrs["counts"], table.attrs["pence_log"] = counts, pence_log
    return swiss, panels, table, drop_log


# --------------------------------------------------------------------------- CLI
def summary(data_dir=None, pretrain_dir=None, regions=REGIONS) -> None:
    """Print counts per filter, region and sector, plus the dedupe and pence logs."""
    swiss, panels, table, drop_log = load_markets(data_dir, pretrain_dir, regions)
    print("[summary] filter counts\n" + table.attrs["counts"].to_string(index=False))
    print(f"[summary] kept foreign stocks: {len(table)}; Swiss stocks: {len(swiss.tickers)}")
    print("[summary] by region and sector\n" + pd.crosstab(table["sector"], table["region"]).to_string())
    print("[summary] by market index\n" + table.groupby(["region", "market_index"]).size().to_string())
    print(f"[summary] dedupe drops: {len(drop_log)}\n" + drop_log.to_string(index=False))
    print(f"[summary] pence overrides: {len(table.attrs['pence_log'])}\n" + table.attrs["pence_log"].to_string(index=False))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--download-fx", action="store_true", help="download the CHF cross rates into raw/fx.parquet")
    ap.add_argument("--summary", action="store_true", help="print filter counts, dedupe and pence logs")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--pretrain-dir", type=Path, default=None)
    ap.add_argument("--regions", default=",".join(REGIONS))
    a = ap.parse_args(argv)
    if a.download_fx:
        download_fx(a.data_dir)
    if a.summary:
        summary(a.data_dir, a.pretrain_dir, tuple(a.regions.split(",")))


if __name__ == "__main__":
    main()
