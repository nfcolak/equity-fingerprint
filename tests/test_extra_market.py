import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from equity_fingerprint import data
from equity_fingerprint import extra_market as E


def fake_long() -> pd.DataFrame:
    rows = []
    for t, base in (("^VIX", 15.0), ("^STOXX50E", 4000.0)):
        for i, d in enumerate(pd.bdate_range("2020-01-01", periods=5)):
            rows.append(dict(ticker=t, date=d, close=base + i))
    df = pd.DataFrame(rows)
    # an unsorted duplicate row that must be dropped
    return pd.concat([df.iloc[::-1], df.iloc[[0]]], ignore_index=True)


class ExtraMarketTests(unittest.TestCase):
    def test_missing_file_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as cm:
                E.load_extra(tmp)
        self.assertIn("python -m equity_fingerprint.extra_market", str(cm.exception))

    def test_load_roundtrip_sorted_no_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            raw.mkdir()
            fake_long().to_parquet(raw / "extra_index.parquet", index=False)
            x = E.load_extra(tmp)
        self.assertEqual(sorted(x.columns), sorted(E.TICKERS))
        self.assertTrue(x.index.is_monotonic_increasing)
        self.assertEqual(len(x), 5)
        self.assertEqual(float(x.loc["2020-01-01", "^VIX"]), 15.0)

    def test_env_data_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            raw.mkdir()
            fake_long().to_parquet(raw / "extra_index.parquet", index=False)
            with mock.patch.dict(os.environ, {"EF_DATA_DIR": tmp}):
                self.assertEqual(len(E.load_extra()), 5)

    def test_download_uses_data_download_series_and_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(data, "download_series", return_value=fake_long()) as ds:
                E.download(tmp)
            ds.assert_called_once_with(E.TICKERS, "extra")
            self.assertTrue((Path(tmp) / "raw" / "extra_index.parquet").exists())
            self.assertEqual(len(E.load_extra(tmp)), 5)

    def test_main_runs_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(data, "download_series", return_value=fake_long()):
                E.main(["--data-dir", tmp])
            self.assertTrue((Path(tmp) / "raw" / "extra_index.parquet").exists())


if __name__ == "__main__":
    unittest.main()
