"""Synthetic CH + EU + US fixture for the v10 tests (groups 1-4). SYNTHETIC data only.

``make_synth(root)`` writes a fake data dir ``root/data`` (Swiss ``universe.csv``, ``raw/prices.parquet``,
``raw/index.parquet`` with ^SSHI / ^SSMI, ``raw/info.parquet``, ``raw/fx.parquet``) and a fake pretrain dir
``root/pretrain`` (``universe.csv``, ``raw/prices.parquet``, ``raw/index.parquet``, ``raw/info.parquet``,
``raw/info/<ticker>.json`` with ``country`` plus a text field that no code may read) and returns a dict with
the two paths, the dates and the ticker lists.

Structure: a global factor, one factor per market, three sector factors shared by all markets (so sector
neighbours exist across markets); stocks are sector ``i % 3``.
  Swiss  S00..S{n-1}.SW (CHF); S00/S01 and S02/S03 are share lines of one company.
  EU     DE (EUR, ^GDAXI), GB (GBp quotes / GBP / ^FTSE), FR (EUR, ^FCHI), SE (SEK, ^OMX).
  US     USD, ^GSPC.
Special lines (``specials=True``; all rejected or deduplicated by ``multi_market`` except the *kept* ones):
  rejected  ABBX.DE (Swiss name), BAD.SW (.SW), ADRX (domicile China), PENNY (price), ILLIQ (traded value),
            SHORT (< 504 returns), NOSEC (no sector), ETFX (ETF), DUPNAME.DE (name = Swiss company S05)
  duplicate DUPW.DE: FX-wedged copy of S10.SW (weekly CHF correlation ~0.9+), dropped, partner S10.SW
  kept      SIML.FR: noisier copy of S11.SW (weekly correlation ~0.8)
  pair      CROSS (US, USD) / CROSS.PA (FR, EUR): one underlying, different names; the US line is dropped
  pence     GBPL.L is quoted in pence but labelled GBP (pence override), POUND.L is quoted in pounds
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SECTORS = ["Industrials", "Healthcare", "Financial Services"]
FX_BASE = {"USDCHF=X": 0.90, "EURCHF=X": 0.95, "GBPCHF=X": 1.12, "NOKCHF=X": 0.085, "SEKCHF=X": 0.088,
           "DKKCHF=X": 0.127}
MARKETS = {  # key: (listing country code, suffix, local index, quote currency, domicile name)
    "DE": ("DE", ".DE", "^GDAXI", "EUR", "Germany"), "GB": ("GB", ".L", "^FTSE", "GBP", "United Kingdom"),
    "FR": ("FR", ".PA", "^FCHI", "EUR", "France"), "SE": ("SE", ".ST", "^OMX", "SEK", "Sweden"),
    "US": ("US", "", "^GSPC", "USD", "United States"),
}
SECRET_TEXT = "SYNTHETIC-DESCRIPTION-MUST-NOT-BE-READ"
COLS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]


def _bars(dates, ticker, close, vol, rng):
    close = np.asarray(close, float)
    n = len(close)
    return pd.DataFrame({
        "date": dates, "ticker": ticker, "open": close * (1 + rng.normal(0, 0.002, n)),
        "high": close * (1 + np.abs(rng.normal(0, 0.006, n)) + 0.001),
        "low": close * (1 - np.abs(rng.normal(0, 0.006, n)) - 0.001),
        "close": close, "adj_close": close, "volume": np.asarray(vol, float)})


def make_synth(root, seed=0, T=1700, start="2018-01-02", n_ch=24, specials=True, dup_noise=0.0035,
               sim_noise=0.009) -> dict:
    root = Path(root)
    data_dir, pre_dir = root / "data", root / "pretrain"
    for d in (data_dir / "raw", pre_dir / "raw" / "info"):
        d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=T)
    g = rng.normal(0, 0.01, T)
    reg = {k: rng.normal(0, 0.01, T) for k in ["CH", *MARKETS]}
    sec = rng.normal(0, 0.01, (T, len(SECTORS)))

    def mkt_ret(k):
        return 0.6 * g + 0.7 * reg[k] + rng.normal(0, 0.003, T)

    mret = {k: mkt_ret(k) for k in ["CH", *MARKETS]}

    def stock_ret(k, s, beta):
        return beta * mret[k] + 0.6 * sec[:, s] + rng.normal(0, 0.008, T)

    def path(r, p0):
        return p0 * np.exp(np.cumsum(r))

    holidays = {k: set(rng.choice(T, 12, replace=False).tolist()) for k in ["CH", *MARKETS]}

    def cal(k, first=0):
        idx = [i for i in range(first, T) if i not in holidays[k]]
        return idx, dates[idx]

    # ---- FX (local -> CHF), a few missing days
    fx_rows = []
    fx_path = {}
    fx_dates = pd.bdate_range(dates[0] - pd.Timedelta(days=14), periods=T + 10)
    miss = set(rng.choice(len(fx_dates), 15, replace=False).tolist())
    for tk, base in FX_BASE.items():
        s = pd.Series(path(rng.normal(0, 0.003, len(fx_dates)), base), index=fx_dates)
        fx_path[tk] = s
        keep = [i for i in range(len(fx_dates)) if i not in miss]
        fx_rows.append(pd.DataFrame({"date": fx_dates[keep], "ticker": tk, "open": s.iloc[keep].to_numpy(),
                                     "high": s.iloc[keep].to_numpy(), "low": s.iloc[keep].to_numpy(),
                                     "close": s.iloc[keep].to_numpy(), "adj_close": s.iloc[keep].to_numpy(),
                                     "volume": 0.0}))
    pd.concat(fx_rows, ignore_index=True).to_parquet(data_dir / "raw" / "fx.parquet", index=False)

    def fx_on(tk, lag=0):
        """FX at the data dates as the panel would see it (ffill; lag in calendar rows)."""
        s = fx_path[tk].reindex(dates.union(fx_path[tk].index)).ffill().reindex(dates)
        return s.shift(lag).bfill().to_numpy()

    # ---- Swiss
    ch_idx, ch_dates = cal("CH")
    swiss, sw_close, rows_p, rows_i, uni_rows = [], {}, [], [], []
    for i in range(n_ch):
        t = f"S{i:02d}.SW"
        s = i % 3
        company = f"Company {i - 1:02d}" if i in (1, 3) else f"Swiss Co {i:02d}"
        c = path(stock_ret("CH", s, rng.uniform(0.6, 1.3)), rng.uniform(20, 200))
        sw_close[t] = c
        vol = rng.lognormal(np.log(2e5), 0.3, T)
        rows_p.append(_bars(dates[ch_idx], t, c[ch_idx], vol[ch_idx], rng))
        shares = float(rng.uniform(1e7, 1e8))
        rows_i.append(dict(ticker=t, name=company, sector=SECTORS[s], industry="x", shares_outstanding=shares,
                           market_cap=c[ch_idx[-1]] * shares, currency="CHF", quote_type="EQUITY"))
        uni_rows.append(dict(ticker=t, symbol=t[:-3], company=company, isin=f"CH{i:010d}", country="Switzerland"))
        swiss.append(t)
    # company of S01 / S03 = the company of S00 / S02
    for i in (1, 3):
        uni_rows[i]["company"] = uni_rows[i - 1]["company"]
        rows_i[i]["name"] = uni_rows[i - 1]["company"]
    pd.concat(rows_p, ignore_index=True).to_parquet(data_dir / "raw" / "prices.parquet", index=False)
    pd.DataFrame(rows_i).to_parquet(data_dir / "raw" / "info.parquet", index=False)
    pd.DataFrame(uni_rows).to_csv(data_dir / "universe.csv", index=False)
    ssh = path(mret["CH"], 1000.0)
    idx_rows = [_bars(dates, "^SSHI", ssh, np.zeros(T), rng), _bars(dates, "^SSMI", path(mret["CH"] * 0.95, 900.0),
                                                                    np.zeros(T), rng)]
    pd.concat(idx_rows, ignore_index=True).to_parquet(data_dir / "raw" / "index.parquet", index=False)

    # ---- foreign
    f_prices, f_info, f_uni, f_json = [], [], [], {}
    clean = {"EU": [], "US": []}
    counts = {"DE": 8, "GB": 6, "FR": 6, "SE": 4, "US": 20}

    def add(ticker, name, mk, close_local, vol, sector, dom=None, currency=None, info_currency=None, quote="EQUITY",
            rows=None, market_cap_ccy=1.0, region_list=None, share_scale=1.0):
        code, _, index, ccy, domicile = MARKETS[mk]
        idx, _d = cal(mk)
        if rows is not None:
            idx = idx[-rows:]
        f_prices.append(_bars(dates[idx], ticker, np.asarray(close_local)[idx], np.asarray(vol)[idx], rng))
        shares = float(rng.uniform(1e7, 1e8))
        last = float(np.asarray(close_local)[idx[-1]])
        f_info.append(dict(ticker=ticker, name=name, sector=sector, industry="x", shares_outstanding=shares,
                           market_cap=last * shares * market_cap_ccy, currency=info_currency or currency or ccy,
                           quote_type=quote))
        f_uni.append(dict(ticker=ticker, name=name, source="synth", country=code, market_index=index))
        f_json[ticker] = {"country": dom or domicile, "quoteType": quote, "longBusinessSummary": SECRET_TEXT}
        if region_list is not None:
            region_list.append(ticker)

    for mk, n in counts.items():
        code, suffix, index, ccy, domicile = MARKETS[mk]
        for i in range(n):
            t = f"{mk}{i:02d}{suffix}"
            s = i % 3
            r = stock_ret(mk, s, rng.uniform(0.6, 1.3))
            vol = rng.lognormal(np.log(3e5), 0.3, T)
            if mk == "GB":
                if i == 4:   # quoted in pence, labelled GBP
                    add(t, f"Firm {mk}{i:02d}", mk, path(r, rng.uniform(2000, 8000)), vol, SECTORS[s],
                        info_currency="GBP", market_cap_ccy=0.01, region_list=clean["EU"])
                elif i == 5:  # quoted in pounds
                    add(t, f"Firm {mk}{i:02d}", mk, path(r, rng.uniform(20, 80)), vol, SECTORS[s],
                        info_currency="GBP", region_list=clean["EU"])
                else:         # quoted in pence, labelled GBp
                    add(t, f"Firm {mk}{i:02d}", mk, path(r, rng.uniform(2000, 8000)), vol, SECTORS[s],
                        info_currency="GBp", market_cap_ccy=0.01, region_list=clean["EU"])
            else:
                lst = clean["US" if mk == "US" else "EU"]
                p0 = rng.uniform(300, 900) if mk == "SE" else rng.uniform(20, 150)
                add(t, f"Firm {mk}{i:02d}", mk, path(r, p0), vol, SECTORS[s], region_list=lst)
    out = dict(data_dir=data_dir, pretrain_dir=pre_dir, dates=dates, swiss=swiss, eu=clean["EU"], us=clean["US"],
               specials={}, sectors=SECTORS, fx=pd.DataFrame(fx_path))

    if specials:
        sp = out["specials"]

        def vol_():
            return rng.lognormal(np.log(3e5), 0.3, T)

        add("ABBX.DE", "ABB Ltd (Germany)", "DE", path(stock_ret("DE", 0, 1.0), 40), vol_(), SECTORS[0])
        add("BAD.SW", "Bad Swiss Line", "DE", path(stock_ret("DE", 1, 1.0), 40), vol_(), SECTORS[1])
        add("ADRX", "Adr Firm", "US", path(stock_ret("US", 2, 1.0), 40), vol_(), SECTORS[2], dom="China")
        add("PENNY", "Penny Firm", "US", path(stock_ret("US", 0, 1.0), 2.0), vol_(), SECTORS[0])
        add("ILLIQ", "Illiquid Firm", "US", path(stock_ret("US", 1, 1.0), 50), np.full(T, 100.0), SECTORS[1])
        add("SHORT", "Short Firm", "US", path(stock_ret("US", 2, 1.0), 50), vol_(), SECTORS[2], rows=300)
        add("NOSEC", "No Sector Firm", "US", path(stock_ret("US", 0, 1.0), 50), vol_(), None)
        add("ETFX", "Some Etf", "US", path(stock_ret("US", 1, 1.0), 50), vol_(), SECTORS[1], quote="ETF")
        add("DUPNAME.DE", "Swiss Co 05 AG ADR", "DE", path(stock_ret("DE", 2, 1.0), 50), vol_(), SECTORS[2])
        # FX-wedged copy of S10.SW: CHF price = Swiss price x noise, quoted in EUR through the previous-day rate
        chf = sw_close["S10.SW"] * np.exp(np.cumsum(rng.normal(0, dup_noise, T)))
        add("DUPW.DE", "Wedge Duplicate", "DE", chf / fx_on("EURCHF=X", lag=1), vol_(), SECTORS[10 % 3])
        chf2 = sw_close["S11.SW"] * np.exp(np.cumsum(rng.normal(0, sim_noise, T)))
        add("SIML.FR", "Similar Not Duplicate", "FR", chf2 / fx_on("EURCHF=X"), vol_(), SECTORS[11 % 3])
        und = path(stock_ret("US", 1, 1.0), 60.0) * 0.9
        us_vol, eu_vol = vol_() * 2.0, vol_()
        add("CROSS", "Crossway Inc", "US", und * np.exp(np.cumsum(rng.normal(0, 0.001, T))), us_vol, SECTORS[1])
        add("CROSS.PA", "Kreuzung SA", "FR", und * 0.9 / 0.95 * np.exp(np.cumsum(rng.normal(0, 0.001, T))), eu_vol,
            SECTORS[1], dom="France")
        for k, v in dict(swiss_name="ABBX.DE", sw="BAD.SW", non_domiciled="ADRX", penny="PENNY", illiquid="ILLIQ",
                         short="SHORT", no_sector="NOSEC", etf="ETFX", name_dup="DUPNAME.DE", wedge_dup="DUPW.DE",
                         wedge_of="S10.SW", similar="SIML.FR", similar_of="S11.SW", cross_us="CROSS",
                         cross_eu="CROSS.PA", pence_gbp="GB04.L", pounds="GB05.L", pence_gbp_label="GB00.L").items():
            sp[k] = v

    pd.concat(f_prices, ignore_index=True).to_parquet(pre_dir / "raw" / "prices.parquet", index=False)
    pd.DataFrame(f_info).to_parquet(pre_dir / "raw" / "info.parquet", index=False)
    pd.DataFrame(f_uni).to_csv(pre_dir / "universe.csv", index=False)
    idx_rows = []
    for mk, (code, suffix, index, ccy, dom) in MARKETS.items():
        idx_rows.append(_bars(dates, index, path(mret[mk], 1000.0), np.zeros(T), rng))
    pd.concat(idx_rows, ignore_index=True).to_parquet(pre_dir / "raw" / "index.parquet", index=False)
    for t, j in f_json.items():
        (pre_dir / "raw" / "info" / f"{t}.json").write_text(json.dumps(j))
    out["foreign_all"] = [r["ticker"] for r in f_info]
    return out
