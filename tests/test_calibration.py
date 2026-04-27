from __future__ import annotations

import unittest
from pathlib import Path

from invest_digital_human.calibration import TradeNodeCalibrationStore, build_calibration, save_calibrations
from invest_digital_human.market_data import MarketCandle


class CalibrationTest(unittest.TestCase):
    def test_build_save_and_load_calibration(self) -> None:
        calibration = build_calibration("msft", _cyclical_candles(420))

        self.assertIsNotNone(calibration)
        assert calibration is not None
        self.assertTrue(calibration.usable())
        tmp_dir = Path.cwd() / ".test-calibration-cache"
        tmp_dir.mkdir(exist_ok=True)
        path = tmp_dir / "trade_node_calibration.json"
        try:
            save_calibrations(path, {calibration.source_key: calibration})
            store = TradeNodeCalibrationStore(path)

            loaded = store.get("MSFT")
        finally:
            path.unlink(missing_ok=True)
            tmp_dir.rmdir()

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.parameter_source, "calibrated")
        self.assertEqual(loaded.parameters.first_buy_drawdown_band, calibration.parameters.first_buy_drawdown_band)

    def test_insufficient_samples_do_not_create_calibration(self) -> None:
        self.assertIsNone(build_calibration("msft", _cyclical_candles(120)))


def _cyclical_candles(count: int) -> list[MarketCandle]:
    candles: list[MarketCandle] = []
    for index in range(count):
        cycle = index % 80
        if cycle < 40:
            close = 100 + cycle
        elif cycle < 58:
            close = 140 - (cycle - 40) * 1.8
        else:
            close = 108 + (cycle - 58) * 1.2
        candles.append(
            MarketCandle(
                timestamp=1_700_000_000 + index * 86_400,
                date=f"2025-01-{(index % 28) + 1:02d}",
                open=close,
                high=close + 2,
                low=close - 2,
                close=close,
                volume=1_000_000,
            )
        )
    return candles


if __name__ == "__main__":
    unittest.main()
