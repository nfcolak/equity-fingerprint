"""features.panel_from_frames: build_panel is a thin wrapper; chf_only / market / rate options."""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from equity_fingerprint import features as F


def fake_frames(seed=0, T=300):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=T)
    rows = []
    for t, cur in (("AAA.SW", "CHF"), ("BBB.SW", "CHF"), ("CCC.DE", "EUR")):
        px = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, T)))
        for i, d in enumerate(dates):
            if t == "BBB.SW" and i in (20, 21):
                continue  # gap inside the span -> forward fill
            rows.append(dict(date=d, ticker=t, open=px[i] * 0.99, high=px[i] * 1.01, low=px[i] * 0.98,
                             close=px[i], adj_close=px[i], volume=float(rng.integers(0, 1000))))
    prices = pd.DataFrame(rows)
    idx = []
    for name in ("^SSHI", "^GDAXI"):
        lv = 1000 * np.exp(np.cumsum(rng.normal(0, 0.008, T)))
        idx += [dict(date=d, ticker=name, open=v, high=v, low=v, close=v, adj_close=v, volume=0.0)
                for d, v in zip(dates, lv)]
    info = pd.DataFrame(dict(ticker=["AAA.SW", "BBB.SW", "CCC.DE"], currency=["CHF", "CHF", "EUR"],
                             sector=["Industrials", "Healthcare", "Technology"],
                             shares_outstanding=[1e6, 2e6, 3e6]))
    return prices, pd.DataFrame(idx), info


class PanelFromFramesTest(unittest.TestCase):
    def setUp(self):
        self.prices, self.index, self.info = fake_frames()

    def test_build_panel_equals_frames(self):
        with tempfile.TemporaryDirectory() as d:
            raw = os.path.join(d, "raw")
            os.makedirs(raw)
            self.prices.to_parquet(os.path.join(raw, "prices.parquet"))
            self.index.to_parquet(os.path.join(raw, "index.parquet"))
            self.info.to_parquet(os.path.join(raw, "info.parquet"))
            a = F.build_panel(d)
        b = F.panel_from_frames(self.prices, self.index, self.info)
        self.assertEqual(a.tickers, b.tickers)
        for k in ("adj", "close", "open", "high", "low", "volume", "tv", "ret", "rows", "span"):
            pd.testing.assert_frame_equal(getattr(a, k), getattr(b, k))
        pd.testing.assert_series_equal(a.mkt, b.mkt)

    def test_chf_only_and_market(self):
        p = F.panel_from_frames(self.prices, self.index, self.info)
        self.assertEqual(p.tickers, ["AAA.SW", "BBB.SW"])
        self.assertEqual(p.excluded_currency, {"CCC.DE": "EUR"})
        q = F.panel_from_frames(self.prices, self.index, self.info, chf_only=False, market="^GDAXI")
        self.assertIn("CCC.DE", q.tickers)
        dax = self.index[self.index.ticker == "^GDAXI"].set_index("date")["close"]
        np.testing.assert_allclose(q.mkt.iloc[1:].to_numpy(), np.log(dax).diff().iloc[1:].to_numpy())

    def test_rate_scales_prices_not_returns(self):
        base = F.panel_from_frames(self.prices, self.index, self.info, chf_only=False)
        R = pd.DataFrame(2.0, index=base.calendar, columns=base.tickers)
        conv = F.panel_from_frames(self.prices, self.index, self.info, chf_only=False, rate=R)
        pd.testing.assert_frame_equal(conv.ret, base.ret)
        pd.testing.assert_frame_equal(conv.close, base.close * 2.0)
        pd.testing.assert_frame_equal(conv.tv, base.tv * 2.0)
        one = F.panel_from_frames(self.prices, self.index, self.info, chf_only=False,
                                  rate=pd.DataFrame(1.0, index=base.calendar, columns=base.tickers))
        for k in ("close", "open", "high", "low", "tv"):
            pd.testing.assert_frame_equal(getattr(one, k), getattr(base, k))


if __name__ == "__main__":
    unittest.main()
