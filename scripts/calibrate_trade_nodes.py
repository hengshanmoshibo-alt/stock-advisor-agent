from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from invest_digital_human.calibration import build_calibration, load_calibrations, save_calibrations
from invest_digital_human.config import AppSettings
from invest_digital_human.market_data import MarketDataClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate deterministic trade-node parameters from daily candles.")
    parser.add_argument("--tickers", required=True, help="Comma-separated source keys or tickers, for example MSFT,NVDA,AMD.")
    parser.add_argument("--lookback-days", type=int, default=1600)
    parser.add_argument("--output", type=Path, default=None)
    return parser


async def _run(args: argparse.Namespace) -> int:
    settings = AppSettings.from_env()
    output = args.output or settings.trade_node_calibration_path
    provider = settings.market_data_provider
    api_key = settings.massive_api_key if provider.strip().lower() == "massive" else settings.finnhub_api_key
    market_data = MarketDataClient(
        provider=provider,
        api_key=api_key,
        timeout=settings.quote_lookup_timeout,
    )
    existing = load_calibrations(output)
    updated = dict(existing)
    tickers = [item.strip().lower() for item in args.tickers.split(",") if item.strip()]
    for source_key in tickers:
        candles = await market_data.get_daily_candles(source_key, lookback_days=args.lookback_days)
        calibration = build_calibration(source_key, candles)
        if calibration is None:
            print(f"{source_key}: skipped, insufficient backtest samples ({len(candles)} candles)")
            continue
        updated[calibration.source_key] = calibration
        first_buy = calibration.backtest_report.node_stats.get("first_buy")
        hit_rate = first_buy.hit_rate_60d if first_buy else None
        print(
            f"{source_key}: calibrated band={calibration.parameters.first_buy_drawdown_band}, "
            f"samples={first_buy.trigger_count if first_buy else 0}, hit_rate_60d={hit_rate}"
        )
    save_calibrations(output, updated)
    print(f"Calibration written to: {output}")
    return 0


def main() -> int:
    return asyncio.run(_run(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
