"""Download and cache the non-Swiss v4 pretraining universe (Europe + US stocks).

CLI: python -m equity_fingerprint.pretrain_data [--refresh] [--limit N] [--pretrain-dir DIR]
         [--sources a,b,...] [--max-per-source SOURCE=N] [--universe-only] [--no-info]
Writes <pretrain-dir>/{universe.csv, manifest.json, raw/prices.parquet, raw/index.parquet,
raw/info.parquet}; EF_PRETRAIN_DIR overrides the default <repo>/data/pretrain. Constituent lists
come from Wikipedia plus the Nasdaq Trader symbol directory (source nasdaqtrader: US common
stocks). Every step is cached on disk: price batches are stored as parquet parts under
raw/parts/ and info snapshots as one json per ticker, so a crash or rate limit resumes without
re-downloading finished work. --limit N caps the universe at about N tickers (spread evenly over
the chosen sources without a --max-per-source); a rerun with more sources downloads only
the tickers not cached yet. The 2021-12-31 cutoff is applied by the pretraining loader, not here.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import re
import time
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd

from .data import FIELD_MAP, INFO_FIELDS, PRICE_COLS, _download

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
WIKI = "https://{lang}.wikipedia.org/wiki/{page}"
BATCH = 50
INFO_WORKERS = 4
MIN_INDEX_ROWS = 1000          # an index with fewer rows is replaced by its fallback
INDEX_FALLBACK = {"OBX.OL": "OSEBX.OL"}   # OBX.OL has a single row on yfinance
SUFFIXES = {"L", "DE", "PA", "AS", "MC", "MI", "ST", "CO", "HE", "OL", "BR", "VI", "LS"}
# companies that sit in non-Swiss index lists but are Swiss-domiciled / Swiss-headquartered
SWISS_NAMES = re.compile(r"\b(ABB|Glencore|Garmin|Chubb|TE Connectivity|Logitech|Alcon|Transocean|"
                         r"Coca-Cola HBC|Weatherford|Sensata)\b", re.I)
STOXX_COUNTRY = {  # STOXX 600 country -> (code, yfinance suffix, market index)
    "United Kingdom": ("GB", ".L", "^FTSE"), "Germany": ("DE", ".DE", "^GDAXI"),
    "France": ("FR", ".PA", "^FCHI"), "Sweden": ("SE", ".ST", "^OMX"),
    "Netherlands": ("NL", ".AS", "^AEX"), "Spain": ("ES", ".MC", "^IBEX"),
    "Italy": ("IT", ".MI", "FTSEMIB.MI"), "Belgium": ("BE", ".BR", "^BFX"),
    "Finland": ("FI", ".HE", "^OMXH25"), "Norway": ("NO", ".OL", "OBX.OL"),
    "Denmark": ("DK", ".CO", "^OMXC25"), "Austria": ("AT", ".VI", "^ATX"),
    "Portugal": ("PT", ".LS", "PSI20.LS"),
}


def _src(page, cols, country, index, suffix="", lang="en", kind="table", min_rows=8):
    """Source spec. cols maps our key -> lowercase column-name prefix on the Wikipedia table."""
    return dict(page=page, cols=cols, country=country, index=index, suffix=suffix, lang=lang,
                kind=kind, min_rows=min_rows)


TK = {"ticker": "ticker", "name": "company"}
# order = dedupe priority (first source wins)
SOURCES = {
    "sp500": _src("List_of_S%26P_500_companies", {"ticker": "symbol", "name": "security",
                  "hq": "headquarters"}, "US", "^GSPC", kind="us", min_rows=100),
    "sp400": _src("List_of_S%26P_400_companies", {"ticker": "symbol", "name": "security",
                  "hq": "headquarters"}, "US", "^GSPC", kind="us", min_rows=100),
    "sp600": _src("List_of_S%26P_600_companies", {"ticker": "symbol", "name": "security",
                  "hq": "headquarters"}, "US", "^GSPC", kind="us", min_rows=100),
    "ftse100": _src("FTSE_100_Index", TK, "GB", "^FTSE", ".L"),
    "ftse250": _src("FTSE_250_Index", TK, "GB", "^FTSE", ".L", min_rows=100),
    "dax": _src("DAX", TK, "DE", "^GDAXI", ".DE"),
    "mdax": _src("MDAX", {"ticker": "symbol", "name": "name"}, "DE", "^GDAXI", ".DE"),
    "sdax": _src("SDAX", {"name": "name"}, "DE", "^GDAXI", ".DE", kind="resolve", min_rows=30),
    "cac40": _src("CAC_40", TK, "FR", "^FCHI", ".PA"),
    "sbf120": _src("SBF_120", {}, "FR", "^FCHI", ".PA", lang="fr", kind="fr_list"),
    "aex": _src("AEX_index", TK, "NL", "^AEX", ".AS"),
    "amx": _src("AMX_index", TK, "NL", "^AEX", ".AS"),
    "ibex35": _src("IBEX_35", TK, "ES", "^IBEX", ".MC"),
    "ftsemib": _src("FTSE_MIB", {"ticker": "ticker", "name": "company", "isin": "isin"},
                    "IT", "FTSEMIB.MI", ".MI"),
    "omxs30": _src("OMX_Stockholm_30", TK, "SE", "^OMX", ".ST"),
    "omxc25": _src("OMX_Copenhagen_25", TK, "DK", "^OMXC25", ".CO"),
    "omxh25": _src("OMX_Helsinki_25", TK, "FI", "^OMXH25", ".HE"),
    "obx": _src("OBX_Index", TK, "NO", "OBX.OL", ".OL"),
    "bel20": _src("BEL_20", TK, "BE", "^BFX", ".BR"),
    "atx": _src("Austrian_Traded_Index", {"name": "company"}, "AT", "^ATX", ".VI",
                kind="resolve", min_rows=15),
    "psi": _src("PSI-20", TK, "PT", "PSI20.LS", ".LS"),
    # extra lists beyond the required ones, to widen the universe toward ~3-5k
    "stoxx600": _src("STOXX_Europe_600", {"ticker": "ticker", "name": "company",
                     "country": "country"}, "", "", kind="stoxx", min_rows=200),
    "ftse_smallcap": _src("FTSE_SmallCap_Index", TK, "GB", "^FTSE", ".L", min_rows=50),
    "oslo": _src("List_of_companies_listed_on_the_Oslo_Stock_Exchange", TK, "NO", "OBX.OL",
                 ".OL", min_rows=100),
    # all US common stocks from the free Nasdaq Trader symbol directory (no Wikipedia page)
    "nasdaqtrader": _src("", {}, "US", "^GSPC", kind="nasdaqtrader", min_rows=1000),
}
NASDAQTRADER = {  # file -> (symbol column, test-issue column, ETF column)
    "nasdaqlisted": ("Symbol", "Test Issue", "ETF"),
    "otherlisted": ("ACT Symbol", "Test Issue", "ETF"),
}
NASDAQTRADER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/{name}.txt"
DEFAULT_MAX_PER_SOURCE = {"nasdaqtrader": 2600}   # applied unless --limit is given
# security names that are not plain common stock
NOT_COMMON = re.compile(r"\b(?:warrants?|rights?|units?|preferred|preference|depositary|notes?|"
                        r"acquisition (?:corp|co)\w*|debentures?|subordinated|etn|"
                        r"trust certificates?|fund|beneficial interest)\b", re.I)


# --------------------------------------------------------------------------- paths
def _pretrain_dir(pretrain_dir=None) -> Path:
    if pretrain_dir is None:
        pretrain_dir = os.environ.get("EF_PRETRAIN_DIR")
    if pretrain_dir is None:
        pretrain_dir = Path(__file__).resolve().parents[2] / "data" / "pretrain"
    return Path(pretrain_dir)


# --------------------------------------------------------------------------- html / tables
def _fetch_html(url: str, path: Path, refresh: bool) -> str:
    if path.exists() and not refresh:
        return path.read_text()
    last = None
    for k in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                html = r.read().decode("utf-8")
            path.write_text(html)
            return html
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(3 * (k + 1))
    raise RuntimeError(f"{url}: {last!r}")


def _clean(x) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\s*\[.*?\]", "", str(x))).strip()


def _pick_table(html: str, cols: dict, min_rows: int) -> pd.DataFrame:
    """First table having every wanted column (matched by lowercase prefix); renamed to our keys."""
    for t in pd.read_html(io.StringIO(html), flavor="lxml"):
        if isinstance(t.columns, pd.MultiIndex) or len(t) < min_rows:
            continue
        names = {c: re.sub(r"\[.*?\]", "", str(c)).strip().lower() for c in t.columns}
        found = {}
        for key, prefix in cols.items():
            hit = next((c for c, n in names.items() if n.startswith(prefix)), None)
            if hit is None:
                break
            found[key] = hit
        else:
            out = t[list(found.values())].copy()
            out.columns = list(found)
            return out.reset_index(drop=True)
    raise ValueError(f"no table with columns {list(cols.values())} and >= {min_rows} rows")


def _fr_list(html: str) -> pd.DataFrame:
    """SBF 120 on fr.wikipedia is a bulleted list of company names under 'Liste des entreprises'."""
    from lxml import html as lh

    root = lh.fromstring(html)
    h = root.xpath('//h3[@id="Liste_des_entreprises"]')
    if not h:
        raise ValueError("SBF 120 list heading not found")
    section = h[0].xpath("ancestor::section[1]")  # the list is an <ol> inside this section
    if not section:
        raise ValueError("SBF 120 list section not found")
    names = [_clean(li.text_content()) for li in section[0].xpath(".//li")]
    return pd.DataFrame({"name": [n for n in names if n]})


# --------------------------------------------------------------------------- symbols
def _symbol(raw, suffix: str, us: bool = False) -> str | None:
    """Wikipedia ticker text -> yfinance symbol."""
    s = _clean(raw)
    if not s or s.lower() == "nan":
        return None
    s = s.split(": ")[-1].strip().rstrip(".").replace(" ", "-")
    if us:
        return s.replace(".", "-")
    if "." in s and s.rsplit(".", 1)[1] in SUFFIXES:
        return s
    return s.replace(".", "-") + suffix


def _query_variants(name: str) -> list[str]:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    short = re.sub(r"\b(AG|SE|KGaA|Co\.?|Group|Holding|International|S\.A\.|N\.V\.)\b", " ",
                   ascii_name)
    out = [ascii_name, re.sub(r"[&.]", " ", short).strip()]
    return [q for i, q in enumerate(out) if q and q not in out[:i]]


def _resolve_names(names: list[str], suffix: str, cache_path: Path) -> list[str | None]:
    """Name -> yfinance symbol on the wanted exchange via yfinance Search (cached)."""
    import yfinance as yf

    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    out = []
    for n in names:
        key = f"{n}|{suffix}"
        if key not in cache:
            sym = None
            for q in _query_variants(n):  # yfinance Search finds nothing for accented names
                for k in range(3):
                    try:
                        quotes = yf.Search(q, max_results=12, news_count=0).quotes
                        sym = next((x["symbol"] for x in quotes if x.get("quoteType") == "EQUITY"
                                    and x.get("symbol", "").endswith(suffix)), "")
                        break
                    except Exception as e:  # noqa: BLE001
                        print(f"[universe] search {q!r}: {e!r}")
                        time.sleep(5 * (k + 1))
                if sym != "":  # found, or the lookup itself failed (None): stop trying variants
                    break
            if sym is None:
                out.append(None)  # lookup failed (not cached, retried on the next run)
                continue
            cache[key] = sym
            cache_path.write_text(json.dumps(cache, indent=1))
        out.append(cache[key] or None)
    return out


def _is_swiss(name, hq=None, isin=None, country=None) -> bool:
    return bool(SWISS_NAMES.search(_clean(name))
                or (hq is not None and "switzerland" in str(hq).lower())
                or (isin is not None and str(isin).upper().startswith("CH"))
                or (country is not None and str(country) == "Switzerland"))


def _nasdaqtrader(raw: Path, refresh: bool) -> pd.DataFrame:
    """Common stocks from nasdaqlisted.txt + otherlisted.txt (pipe separated, footer dropped)."""
    html_dir = raw / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, (sym, test, etf) in NASDAQTRADER.items():
        txt = _fetch_html(NASDAQTRADER_URL.format(name=name), html_dir / f"{name}.txt", refresh)
        t = pd.read_csv(io.StringIO(txt), sep="|", dtype=str, keep_default_na=False)
        t = t[~t[sym].str.startswith("File Creation Time")]
        t = t[(t[etf] == "N") & (t[test] == "N") & ~t["Security Name"].str.contains(NOT_COMMON)]
        t = t[~t[sym].str.contains(r"\$", regex=True)]
        frames.append(pd.DataFrame({"ticker": t[sym].str.strip().str.replace(".", "-", regex=False),
                                    "name": t["Security Name"]}))
    return pd.concat(frames, ignore_index=True)


def _load_source(key: str, raw: Path, refresh: bool, cap: int | None) -> pd.DataFrame:
    spec = SOURCES[key]
    html_dir = raw / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    if spec["kind"] == "nasdaqtrader":
        df = _nasdaqtrader(raw, refresh)
        df["name"] = df["name"].map(_clean)
        n_swiss = int(sum(_is_swiss(n) for n in df["name"]))
        df = df[[not _is_swiss(n) for n in df["name"]]]
        df = df[~df["ticker"].str.endswith(".SW")].drop_duplicates("ticker")
        df = df.assign(source=key, country=spec["country"], market_index=spec["index"])
        df.attrs["n_swiss_dropped"] = n_swiss
        return df[["ticker", "name", "source", "country", "market_index"]].reset_index(drop=True)
    html = _fetch_html(WIKI.format(lang=spec["lang"], page=spec["page"]),
                       html_dir / f"{key}.html", refresh)
    kind = spec["kind"]
    df = _fr_list(html) if kind == "fr_list" else _pick_table(html, spec["cols"], spec["min_rows"])
    df["name"] = df["name"].map(_clean)
    for c in ("hq", "isin", "country"):
        if c not in df.columns:
            df[c] = None
    swiss = [_is_swiss(n, h, i, c) for n, h, i, c in zip(df["name"], df["hq"], df["isin"],
                                                        df["country"])]
    df = df[[not s for s in swiss]].reset_index(drop=True)
    n_swiss = int(sum(swiss))
    if kind == "stoxx":  # one list, per-row country -> suffix and market index
        df = df[df["country"].isin(STOXX_COUNTRY)].reset_index(drop=True)
        meta = df["country"].map(STOXX_COUNTRY)
        df["ticker"] = [_symbol(t, m[1]) for t, m in zip(df["ticker"], meta)]
        df["country"] = [m[0] for m in meta]
        df["market_index"] = [m[2] for m in meta]
    else:
        if cap is not None:
            df = df.head(cap).reset_index(drop=True)
        if kind in ("resolve", "fr_list"):
            df["ticker"] = _resolve_names(df["name"].tolist(), spec["suffix"],
                                          raw / "resolve_cache.json")
        else:
            df["ticker"] = [_symbol(t, spec["suffix"], kind == "us") for t in df["ticker"]]
        df["country"] = spec["country"]
        df["market_index"] = spec["index"]
    df = df.dropna(subset=["ticker"])
    df = df[~df["ticker"].str.endswith(".SW")]
    if cap is not None:
        df = df.head(cap)
    df["source"] = key
    df.attrs["n_swiss_dropped"] = n_swiss
    return df[["ticker", "name", "source", "country", "market_index"]].reset_index(drop=True)


# --------------------------------------------------------------------------- universe
def _sample_new(df: pd.DataFrame, seen: set, n: int) -> pd.DataFrame:
    """Seeded random sample (seed 0) of n tickers that are not in an earlier source."""
    df = df[~df["ticker"].isin(seen)].sort_values("ticker")
    return df if len(df) <= n else df.sample(n=n, random_state=0).sort_values("ticker")


def build_universe(root: Path, raw: Path, refresh: bool, keys: list[str], limit: int | None,
                   max_per_source: dict | None = None):
    uni_path, meta_path = root / "universe.csv", raw / "universe_meta.json"
    max_per_source = {k: n for k, n in (max_per_source or {}).items() if k in keys}
    want = {"sources": keys, "limit": limit, "max_per_source": max_per_source}
    if not refresh and uni_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if {k: meta.get(k) for k in want} == want:
            print("[universe] cached")
            return pd.read_csv(uni_path), meta
    if refresh:
        (raw / "resolve_cache.json").unlink(missing_ok=True)
    uncapped = [k for k in keys if k not in max_per_source]  # --limit is shared by these only
    cap = math.ceil(limit / len(uncapped)) if limit and uncapped else None
    frames, status, sampled = [], {}, {}
    for key in keys:
        url = WIKI.format(lang=SOURCES[key]["lang"], page=SOURCES[key]["page"])
        if key == "nasdaqtrader":
            url = ", ".join(NASDAQTRADER_URL.format(name=n) for n in NASDAQTRADER)
        try:
            df = _load_source(key, raw, refresh, None if key in max_per_source else cap)
            n_swiss = int(df.attrs.get("n_swiss_dropped", 0))
            if key in max_per_source:
                seen = set(pd.concat(frames)["ticker"]) if frames else set()
                n_avail = int((~df["ticker"].isin(seen)).sum())
                df = _sample_new(df, seen, max_per_source[key])
                sampled[key] = {"max": max_per_source[key], "n_new_available": n_avail,
                                "n_kept": int(len(df)), "seed": 0}
            frames.append(df)
            status[key] = {"url": url, "status": "ok", "n_listed": int(len(df)),
                           "n_swiss_dropped": n_swiss}
            print(f"[universe] {key}: {len(df)} tickers")
        except Exception as e:  # noqa: BLE001
            status[key] = {"url": url, "status": "failed", "error": repr(e)[:300]}
            print(f"[universe] {key} FAILED: {e!r}")
    if not frames:
        raise RuntimeError("no constituent list could be loaded")
    uni = pd.concat(frames, ignore_index=True)
    uni = uni.drop_duplicates("ticker", keep="first").reset_index(drop=True)
    uni.to_csv(uni_path, index=False)
    meta = {**want, "source_status": status, "sampled_sources": sampled,
            "dropped_swiss_by_info": []}
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[universe] {len(uni)} unique tickers from {len(frames)}/{len(keys)} sources")
    return uni, meta


# --------------------------------------------------------------------------- prices
def _batch_part(parts: Path, prefix: str, batch: list[str]) -> Path:
    h = hashlib.md5(",".join(batch).encode()).hexdigest()[:12]
    return parts / f"{prefix}_{h}.parquet"


def _download_batch(batch: list[str], label: str, strict: bool = True) -> pd.DataFrame | None:
    """Download one batch; None when it looks rate limited / offline (retried on the next run).

    strict: an empty result for a big batch is treated as a rate limit and retried with backoff;
    the second pass over known-failed tickers passes strict=False (empty is then a real answer).
    """
    for wait in (0, 30, 90) if strict else (0,):
        time.sleep(wait)
        try:
            df = _download(batch)
        except Exception as e:  # noqa: BLE001
            print(f"[{label}] batch error: {e!r}")
            continue
        if len(df) or len(batch) < 10 or not strict:
            return df
        print(f"[{label}] empty result for {len(batch)} tickers, backing off")
    return None


def _fetch_parts(tickers: list[str], prefix: str, parts: Path, refresh: bool,
                 batch_size: int = BATCH) -> tuple[list[str], int]:
    """Download in batches into parts/<prefix>_*.parquet; returns (no-data tickers, pending)."""
    parts.mkdir(parents=True, exist_ok=True)
    nodata_path = parts / f"{prefix}_nodata.json"
    if refresh:  # start over: stale parts would otherwise shadow the new download
        for p in parts.glob(f"{prefix}_*"):
            p.unlink()
    nodata = set() if not nodata_path.exists() else set(json.loads(nodata_path.read_text()))
    have = set()
    for p in parts.glob(f"{prefix}_*.parquet"):
        have |= set(pd.read_parquet(p, columns=["ticker"])["ticker"])
    # incremental: only tickers that no earlier run has downloaded (or found empty) are fetched
    todo = sorted(set(tickers) - have - nodata)
    print(f"[{prefix}] fetching {len(todo)} tickers ({len(set(tickers)) - len(todo)} cached)")
    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    pending, attempted = 0, []
    for n, batch in enumerate(batches, 1):
        df = _download_batch(batch, prefix)
        if df is None:
            pending += 1
            continue
        df.to_parquet(_batch_part(parts, prefix, batch), index=False)
        attempted += batch
        print(f"[{prefix}] batch {n}/{len(batches)}: {df.ticker.nunique()}/{len(batch)} ok")
    for p in parts.glob(f"{prefix}_*.parquet"):
        have |= set(pd.read_parquet(p, columns=["ticker"])["ticker"])
    missing = sorted(set(attempted) - have - nodata)
    for i in range(0, len(missing), 20):  # second pass: smaller batches for tickers that failed
        batch = missing[i:i + 20]
        df = _download_batch(batch, prefix, strict=False)
        if df is None:
            pending += 1
            continue
        if len(df):
            df.to_parquet(_batch_part(parts, prefix, batch + ["retry"]), index=False)
            have |= set(df["ticker"])
        print(f"[{prefix}] retry batch: {df.ticker.nunique()}/{len(batch)} ok")
    if not pending:
        nodata |= set(attempted) - have
        nodata_path.write_text(json.dumps(sorted(nodata)))
    return sorted(nodata & set(tickers)), pending


def _concat_parts(parts: Path, prefix: str, keep: set[str]) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in sorted(parts.glob(f"{prefix}_*.parquet"))]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=PRICE_COLS)
    out = pd.concat(frames, ignore_index=True)
    out = out[out["ticker"].isin(keep)]
    for c in PRICE_COLS[2:]:
        out[c] = out[c].astype("float64")
    out["date"] = pd.to_datetime(out["date"])
    out = out.drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"])
    return out[PRICE_COLS].reset_index(drop=True)


def resolve_indices(uni: pd.DataFrame, raw: Path, refresh: bool) -> tuple[dict, int]:
    """Download the market indices (with fallbacks); returns ({primary: used symbol}, pending)."""
    primary = sorted(set(uni["market_index"].dropna()))
    cands = sorted(set(primary) | {INDEX_FALLBACK[p] for p in primary if p in INDEX_FALLBACK})
    parts = raw / "parts"
    _, pending = _fetch_parts(cands, "index", parts, refresh)
    idx = _concat_parts(parts, "index", set(cands))
    rows = idx.groupby("ticker").size()
    used = {}
    for p in primary:
        alt = INDEX_FALLBACK.get(p)
        if rows.get(p, 0) < MIN_INDEX_ROWS and alt and rows.get(alt, 0) > rows.get(p, 0):
            used[p] = alt
        else:
            used[p] = p
    return used, pending


# --------------------------------------------------------------------------- info
def _fetch_info(t: str, path: Path) -> tuple[str, str | None]:
    import yfinance as yf

    last = None
    for k in range(3):
        try:
            info = yf.Ticker(t).info
            info = dict(info) if isinstance(info, dict) else {}
            info["_snapshot_date"] = str(date.today())
            if len(info) > 2:
                path.write_text(json.dumps(info, default=str))
                return t, None
            info["_empty"] = True  # nothing known about this symbol: cache the stub, do not retry
            path.write_text(json.dumps(info, default=str))
            return t, "empty info"
        except Exception as e:  # noqa: BLE001
            last = repr(e)
            limited = "ate" in last and "imit" in last or "429" in last or "Too Many" in last
            time.sleep((30 if limited else 2) * (k + 1) + random.random())
    return t, last


def build_info(tickers: list[str], raw: Path, refresh: bool) -> dict:
    d = raw / "info"
    d.mkdir(parents=True, exist_ok=True)
    errors: dict[str, str] = {}
    todo = [t for t in tickers if refresh or not (d / f"{t}.json").exists()]
    print(f"[info] fetching {len(todo)} tickers ({len(tickers) - len(todo)} cached)")
    if todo:
        with ThreadPoolExecutor(max_workers=INFO_WORKERS) as ex:
            for k, (t, err) in enumerate(ex.map(lambda t: _fetch_info(t, d / f"{t}.json"), todo), 1):
                if err:
                    errors[t] = err
                if k % 50 == 0:
                    print(f"[info] {k}/{len(todo)}")
    return errors


def _read_info(tickers: list[str], raw: Path) -> tuple[pd.DataFrame, set]:
    """info.parquet rows from the cached jsons; also the tickers whose country is Switzerland."""
    rows, swiss = [], set()
    for t in tickers:
        f = raw / "info" / f"{t}.json"
        if not f.exists():
            continue
        info = json.loads(f.read_text())
        if info.get("country") == "Switzerland":
            swiss.add(t)
        row = {"ticker": t}
        for k, keys in INFO_FIELDS.items():
            row[k] = next((info[x] for x in keys if info.get(x) not in (None, "")), None)
        rows.append(row)
    df = pd.DataFrame(rows, columns=["ticker", *INFO_FIELDS])
    for c in ("shares_outstanding", "market_cap"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, swiss


# --------------------------------------------------------------------------- manifest
def write_manifest(uni, prices, index, info, root, meta, info_errors, nodata, used_index) -> None:
    counts = prices.groupby("ticker").size()
    failed = {}
    for t in uni["ticker"]:
        reasons = []
        if t not in counts.index:
            reasons.append("no price data")
        elif counts[t] < 500:
            reasons.append(f"only {counts[t]} price rows")
        if t in info_errors:
            reasons.append(f"info: {info_errors[t]}")
        elif t in set(info["ticker"]) and info.loc[info.ticker == t, "sector"].isna().all():
            reasons.append("no sector")
        if reasons:
            failed[t] = "; ".join(reasons)
    sources = [s["url"] for s in meta["source_status"].values() if s["status"] == "ok"]
    irows = index.groupby("ticker").size()
    manifest = {
        "run_date": str(date.today()),
        "n_universe": int(len(uni)),
        "n_with_prices": int(len(counts)),
        "n_with_>=500_rows": int((counts >= 500).sum()),
        "price_date_range": [str(prices.date.min())[:10], str(prices.date.max())[:10]],
        "index_tickers": sorted(index.ticker.unique()),
        "index_used": used_index,
        "index_failed": sorted(set(used_index.values()) - set(irows.index)),
        "index_date_ranges": {t: [str(g.date.min())[:10], str(g.date.max())[:10]]
                              for t, g in index.groupby("ticker")},
        "n_with_sector": int(info.sector.notna().sum()),
        "sector_counts": {k: int(v) for k, v in info.sector.value_counts().items()},
        "n_by_source": {k: int(v) for k, v in uni.source.value_counts().items()},
        "n_by_country": {k: int(v) for k, v in uni.country.value_counts().items()},
        "dropped_swiss_by_info": meta.get("dropped_swiss_by_info", []),
        "failed_tickers": failed,
        "source_status": meta["source_status"],
        "sources": sources,
        "sampled_sources": meta.get("sampled_sources", {}),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] written; {len(failed)} tickers with issues, {len(nodata)} without prices")


# --------------------------------------------------------------------------- main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="re-download even if cached")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the universe at about N tickers, spread evenly over the sources")
    ap.add_argument("--max-per-source", action="append", default=[], metavar="SOURCE=N",
                    help="keep a seeded random sample of N tickers of SOURCE that are not in an "
                         "earlier source (repeatable; default nasdaqtrader=2600 without --limit)")
    ap.add_argument("--pretrain-dir", default=None)
    ap.add_argument("--sources", default=",".join(SOURCES), help="comma separated source keys")
    ap.add_argument("--universe-only", action="store_true", help="stop after universe.csv")
    ap.add_argument("--no-info", action="store_true", help="skip the slow Ticker.info snapshots")
    a = ap.parse_args(argv)
    keys = [k.strip() for k in a.sources.split(",") if k.strip()]
    bad = [k for k in keys if k not in SOURCES]
    if bad:
        ap.error(f"unknown sources {bad}; choose from {list(SOURCES)}")
    # threaded yfinance downloads leak sockets; macOS default soft limit is 256
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = 8192 if hard == resource.RLIM_INFINITY else min(8192, hard)
    if soft < want:
        resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    maxes = {} if a.limit else dict(DEFAULT_MAX_PER_SOURCE)
    for item in a.max_per_source:
        k, _, n = item.partition("=")
        if k not in SOURCES or not n.isdigit():
            ap.error(f"--max-per-source expects SOURCE=N with a known source, got {item!r}")
        maxes[k] = int(n)
    root = _pretrain_dir(a.pretrain_dir)
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    uni, meta = build_universe(root, raw, a.refresh, keys, a.limit, maxes)
    if a.universe_only:
        print(uni.groupby("source", sort=False).size().to_string())
        return
    parts = raw / "parts"
    used_index, pend_idx = resolve_indices(uni, raw, a.refresh)
    nodata, pend = _fetch_parts(uni["ticker"].tolist(), "prices", parts, a.refresh)
    if pend or pend_idx:
        raise SystemExit(f"{pend + pend_idx} batches still pending (rate limit?); rerun to resume")
    prices = _concat_parts(parts, "prices", set(uni["ticker"]))
    if prices.empty:
        raise RuntimeError("price download returned no data")
    errors = {}
    have = sorted(prices.ticker.unique())
    if not a.no_info:
        errors = build_info(have, raw, a.refresh)
    info, swiss = _read_info(have, raw)
    if swiss:  # Swiss-domiciled per yfinance country: out of the universe, prices and info
        print(f"[universe] dropping {len(swiss)} Swiss-domiciled tickers: {sorted(swiss)}")
        meta["dropped_swiss_by_info"] = sorted(swiss)
        (raw / "universe_meta.json").write_text(json.dumps(meta, indent=2))
        uni = uni[~uni.ticker.isin(swiss)].reset_index(drop=True)
        prices = prices[~prices.ticker.isin(swiss)].reset_index(drop=True)
        info = info[~info.ticker.isin(swiss)].reset_index(drop=True)
    uni["market_index"] = uni["market_index"].map(lambda m: used_index.get(m, m))
    uni.to_csv(root / "universe.csv", index=False)
    index = _concat_parts(parts, "index", set(used_index.values()))
    prices.to_parquet(raw / "prices.parquet", index=False)
    index.to_parquet(raw / "index.parquet", index=False)
    info.to_parquet(raw / "info.parquet", index=False)
    print(f"[prices] saved {len(prices)} rows, {prices.ticker.nunique()} tickers; "
          f"[index] {len(index)} rows, {index.ticker.nunique()} indices; [info] {len(info)} rows")
    write_manifest(uni, prices, index, info, root, meta, errors, nodata, used_index)


if __name__ == "__main__":
    main()
