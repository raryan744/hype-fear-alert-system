"""
Per-ticker backtest runner that bypasses the buggy portfolio aggregation in
backtest_engine.run_all_backtests. Loops over each ticker, calls the working
Backtester.backtest_ticker, and persists results incrementally to JSON so a
crash on one ticker can't lose the others.

Output:
  backtest_results_per_ticker.json  — full result dict per ticker
  backtest_summary.txt              — human-readable summary table
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# Silence optuna's per-trial chatter
logging.getLogger("optuna").setLevel(logging.WARNING)

from alert_system import WATCHLIST
from backtest_engine import Backtester

RESULTS_FILE = Path("backtest_results_per_ticker.json")
SUMMARY_FILE = Path("backtest_summary.txt")
START_DATE   = "2022-01-01"

KEEP_FIELDS = [
    "total_return", "annualized_return", "sharpe_ratio", "win_rate",
    "max_drawdown", "total_trades", "volatility", "dominant_regime",
    "regime_distribution", "walk_forward_windows", "ml_available",
    "advanced_features", "data_points",
]


def _clean(result: dict) -> dict:
    """Strip non-JSON-serialisable fields (DataFrames, models)."""
    out = {}
    for k in KEEP_FIELDS:
        if k in result:
            v = result[k]
            if isinstance(v, (int, float, str, bool, list, dict)) or v is None:
                out[k] = v
            else:
                out[k] = str(v)
    return out


def main() -> int:
    bt = Backtester()
    results: dict = {}
    if RESULTS_FILE.exists():
        try:
            results = json.loads(RESULTS_FILE.read_text())
            logging.info(f"Resuming — already have {len(results)} tickers")
        except Exception:
            pass

    t0 = time.time()
    for i, ticker in enumerate(WATCHLIST, 1):
        if ticker in results:
            logging.info(f"[{i}/{len(WATCHLIST)}] {ticker}: skipping (already done)")
            continue

        t_start = time.time()
        logging.info(f"[{i}/{len(WATCHLIST)}] {ticker}: starting...")
        try:
            r = bt.backtest_ticker(ticker, start_date=START_DATE)
            if r is None:
                logging.warning(f"  → returned None")
                results[ticker] = {"error": "no result"}
            else:
                results[ticker] = _clean(r)
                logging.info(
                    f"  → ret={r.get('total_return', 0):.2%} "
                    f"sharpe={r.get('sharpe_ratio', 0):.2f} "
                    f"win={r.get('win_rate', 0):.1%} "
                    f"trades={r.get('total_trades', 0)} "
                    f"({time.time() - t_start:.1f}s)"
                )
        except Exception as e:
            logging.error(f"  → CRASH: {type(e).__name__}: {e}")
            results[ticker] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

        # Persist after every ticker
        RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))

    elapsed = time.time() - t0
    logging.info(f"All tickers done in {elapsed:.0f}s")

    # Build summary table
    lines = []
    lines.append("=" * 100)
    lines.append(f"HYPE-FEAR BACKTEST — {datetime.now().strftime('%Y-%m-%d %H:%M')}  "
                 f"start={START_DATE}")
    lines.append("=" * 100)
    header = (f"{'Ticker':<10} {'TotRet':>9} {'AnnRet':>9} {'Sharpe':>8} "
              f"{'WinRate':>8} {'MaxDD':>9} {'Trades':>7} {'Vol':>7} {'Regime':>10}")
    lines.append(header)
    lines.append("-" * 100)

    sharpes, rets, wins, dds, n_trades = [], [], [], [], []
    for ticker in WATCHLIST:
        r = results.get(ticker, {})
        if "error" in r:
            lines.append(f"{ticker:<10}  ERROR: {r['error']}")
            continue
        try:
            row = (
                f"{ticker:<10} "
                f"{r.get('total_return', 0):>8.2%} "
                f"{r.get('annualized_return', 0):>8.2%} "
                f"{r.get('sharpe_ratio', 0):>8.2f} "
                f"{r.get('win_rate', 0):>7.1%} "
                f"{r.get('max_drawdown', 0):>8.2%} "
                f"{int(r.get('total_trades', 0)):>7d} "
                f"{r.get('volatility', 0):>6.1%} "
                f"{str(r.get('dominant_regime', 'N/A'))[:10]:>10}"
            )
            lines.append(row)
            sharpes.append(r.get("sharpe_ratio", 0))
            rets.append(r.get("annualized_return", 0))
            wins.append(r.get("win_rate", 0))
            dds.append(r.get("max_drawdown", 0))
            n_trades.append(int(r.get("total_trades", 0)))
        except Exception as e:
            lines.append(f"{ticker:<10}  formatting error: {e}")

    lines.append("-" * 100)
    if sharpes:
        positive = [s for s in sharpes if s > 0]
        avg = lambda xs: sum(xs) / len(xs) if xs else 0
        lines.append("")
        lines.append("PORTFOLIO SUMMARY")
        lines.append("-" * 40)
        lines.append(f"  Tickers tested        : {len(sharpes)}")
        lines.append(f"  Avg Sharpe            : {avg(sharpes):+.3f}")
        lines.append(f"  Median Sharpe         : {sorted(sharpes)[len(sharpes)//2]:+.3f}")
        lines.append(f"  % Positive Sharpe     : {len(positive) / len(sharpes):.0%}")
        lines.append(f"  Avg Annualised Return : {avg(rets):+.2%}")
        lines.append(f"  Avg Win Rate          : {avg(wins):.1%}")
        lines.append(f"  Worst Max Drawdown    : {min(dds):.2%}")
        lines.append(f"  Total Trades          : {sum(n_trades):,}")

    summary = "\n".join(lines)
    SUMMARY_FILE.write_text(summary)
    print()
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
