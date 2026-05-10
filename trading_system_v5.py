"""
Trading System v5 — Orchestration Layer

Combines backtesting, live signal generation, portfolio management,
and regime-adaptive execution into a single entry point.

Modes:
  backtest  — run walk-forward backtest on all profiles and generate report
  live      — run real-time alert loop (delegates to alert_system.py)
  signal    — print current signals for all assets and exit
  optimize  — re-optimise per-asset thresholds and save results

Usage:
  python trading_system_v5.py --mode backtest
  python trading_system_v5.py --mode live
  python trading_system_v5.py --mode signal --tickers TSLA NVDA BTC-USD
"""

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from backtest_engine import Backtester
from alert_system import AlertEngine, WATCHLIST, compute_composite_score, send_telegram
from grok_sentiment import get_live_sentiment_batch
from market_regime_detector import MarketRegimeClassifier


# ---------------------------------------------------------------------------
# Portfolio-level metrics
# ---------------------------------------------------------------------------

def portfolio_summary(results: Dict) -> Dict:
    """Aggregate per-ticker backtest results into portfolio-level stats."""
    sharpes   = [r["sharpe_ratio"]      for r in results.values() if "sharpe_ratio"      in r]
    returns_a = [r["annualized_return"] for r in results.values() if "annualized_return" in r]
    win_rates = [r["win_rate"]          for r in results.values() if "win_rate"          in r]
    drawdowns = [r["max_drawdown"]      for r in results.values() if "max_drawdown"      in r]

    positive_sharpe = [s for s in sharpes if s > 0]

    return {
        "avg_sharpe":        np.mean(sharpes)         if sharpes    else 0,
        "median_sharpe":     np.median(sharpes)        if sharpes    else 0,
        "pct_positive_sharpe": len(positive_sharpe) / len(sharpes) if sharpes else 0,
        "avg_annualised_ret": np.mean(returns_a)      if returns_a  else 0,
        "avg_win_rate":       np.mean(win_rates)       if win_rates  else 0,
        "worst_drawdown":     min(drawdowns)           if drawdowns  else 0,
        "n_tickers":          len(results),
    }


def print_portfolio_summary(summary: Dict) -> None:
    print("\n" + "=" * 60)
    print("PORTFOLIO SUMMARY")
    print("=" * 60)
    print(f"  Avg Sharpe Ratio       : {summary['avg_sharpe']:+.3f}")
    print(f"  Median Sharpe          : {summary['median_sharpe']:+.3f}")
    print(f"  % Positive Sharpe      : {summary['pct_positive_sharpe']:.0%}")
    print(f"  Avg Annualised Return  : {summary['avg_annualised_ret']:+.1%}")
    print(f"  Avg Win Rate           : {summary['avg_win_rate']:.1%}")
    print(f"  Worst Max Drawdown     : {summary['worst_drawdown']:.1%}")
    print(f"  Tickers Tested         : {summary['n_tickers']}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Backtest mode
# ---------------------------------------------------------------------------

def run_backtest(tickers: Optional[List[str]] = None,
                 start_date: str = "2020-01-01") -> None:
    logger.info("Starting walk-forward backtest...")
    backtester = Backtester()

    if tickers:
        results = {t: backtester.backtest_ticker(t, start_date=start_date) for t in tickers}
        results = {k: v for k, v in results.items() if v is not None}
    else:
        results = backtester.run_all_backtests()

    report = backtester.generate_report(results)
    print(report)

    # Save report
    fname = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
    with open(fname, "w") as f:
        f.write(report)
    logger.info(f"Report saved to {fname}")

    # Portfolio summary
    summary = portfolio_summary(results)
    print_portfolio_summary(summary)

    # Save JSON results for downstream use
    json_results = {}
    for ticker, r in results.items():
        json_results[ticker] = {k: v for k, v in r.items()
                                if isinstance(v, (int, float, str, bool, list, dict))}
    with open("backtest_results_latest.json", "w") as f:
        json.dump(json_results, f, indent=2)
    logger.info("JSON results saved to backtest_results_latest.json")


# ---------------------------------------------------------------------------
# Signal mode
# ---------------------------------------------------------------------------

def run_signal_scan(tickers: List[str]) -> None:
    """Print current live signals for the given tickers and exit."""
    logger.info(f"Generating live signals for: {tickers}")

    profiles_path = "profiles.json"
    try:
        with open(profiles_path) as f:
            profiles = json.load(f)
    except Exception:
        profiles = {}

    grok_results = get_live_sentiment_batch(tickers, delay_secs=0.5)
    regime_clf   = MarketRegimeClassifier()

    print("\n" + "=" * 70)
    print(f"LIVE SIGNALS — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)
    print(f"{'Ticker':<10} {'Composite':>10} {'Grok':>8} {'Tech':>8} {'Regime':<10} {'Conf':>6}")
    print("-" * 70)

    for ticker in tickers:
        profile = profiles.get(ticker, {})
        grok    = grok_results.get(ticker, {"score": 0.0, "confidence": 0.0, "summary": ""})
        scores  = compute_composite_score(ticker, profile, grok)

        direction = ""
        if scores["composite"] > 0.35:
            direction = "↑ HYPE"
        elif scores["composite"] < -0.35:
            direction = "↓ FEAR"

        print(
            f"{ticker:<10} {scores['composite']:>+10.3f} "
            f"{scores['grok_score']:>+8.3f} {scores['tech_signal']:>+8.3f} "
            f"{scores['regime']:<10} {scores['confidence']:>5.0%}  {direction}"
        )

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Live mode
# ---------------------------------------------------------------------------

def run_live() -> None:
    """Start the real-time alert loop."""
    from alert_system import main as alert_main
    logger.info("Starting live alert system...")
    alert_main()


# ---------------------------------------------------------------------------
# Optimize mode
# ---------------------------------------------------------------------------

def run_optimize(tickers: Optional[List[str]] = None) -> None:
    """
    Run backtests and suggest optimised per-asset thresholds
    based on realised win-rate / trade-frequency trade-off.
    """
    logger.info("Running threshold optimisation...")
    backtester = Backtester()
    tickers    = tickers or list(Backtester().engine.profiles.keys())

    suggestions = {}
    for ticker in tickers:
        result = backtester.backtest_ticker(ticker)
        if result is None:
            continue

        sharpe   = result.get("sharpe_ratio", 0)
        win_rate = result.get("win_rate", 0)
        n_trades = result.get("total_trades", 0)

        # Heuristic: if win_rate low, tighten threshold; if too few trades, loosen it
        base_thresh = 0.22
        if win_rate < 0.50 and n_trades > 200:
            base_thresh += 0.05   # tighten
        elif n_trades < 50:
            base_thresh -= 0.04   # loosen

        suggestions[ticker] = {
            "suggested_bull_threshold": round(base_thresh, 3),
            "suggested_bear_threshold": round(base_thresh + 0.15, 3),
            "current_sharpe": round(sharpe, 3),
            "current_win_rate": round(win_rate, 3),
            "current_trades": n_trades,
        }

        logger.info(
            f"{ticker}: Sharpe={sharpe:.2f}, WinRate={win_rate:.1%}, "
            f"Trades={n_trades} → thresh={base_thresh:.3f}"
        )

    with open("threshold_suggestions.json", "w") as f:
        json.dump(suggestions, f, indent=2)
    logger.info("Threshold suggestions saved to threshold_suggestions.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hype-Fear Trading System v5")
    parser.add_argument("--mode",    choices=["backtest", "live", "signal", "optimize"],
                        default="signal")
    parser.add_argument("--tickers", nargs="+", help="Subset of tickers (default: all)")
    parser.add_argument("--start",   default="2020-01-01", help="Backtest start date")
    args = parser.parse_args()

    tickers = args.tickers or WATCHLIST

    if args.mode == "backtest":
        run_backtest(tickers=args.tickers, start_date=args.start)
    elif args.mode == "live":
        run_live()
    elif args.mode == "signal":
        run_signal_scan(tickers)
    elif args.mode == "optimize":
        run_optimize(tickers=args.tickers)


if __name__ == "__main__":
    main()
