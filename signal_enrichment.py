"""
Enrich a triggered signal with:
  - Current price (yfinance, cached)
  - Expected % / $ move over a configurable timeframe, derived from realised
    volatility (ATR-based — honest, not promising returns the system can't deliver)
  - Suggested option contract (Polygon options chain, nearest weekly 7-14 DTE,
    1-strike OTM in the signal direction)
  - Per-asset historical edge from backtest_results_per_ticker.json (Sharpe,
    Win rate, AnnRet) so the alert shows the asset's own track record

All sources fail gracefully — the alert always sends, optional fields are just
omitted when their data source is unavailable.
"""

import json
import logging
import math
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default forward-looking window for the "expected move" estimate.
DEFAULT_HORIZON_DAYS = int(os.getenv("SIGNAL_HORIZON_DAYS", 5))

_BACKTEST_FILE = Path(__file__).parent / "backtest_results_per_ticker.json"


# ---------------------------------------------------------------------------
# Backtest edge lookup (loaded once at import)
# ---------------------------------------------------------------------------

def _load_backtest_table() -> Dict[str, Dict]:
    if not _BACKTEST_FILE.exists():
        return {}
    try:
        return json.loads(_BACKTEST_FILE.read_text())
    except Exception as e:
        logger.warning(f"Could not load backtest table: {e}")
        return {}

_BACKTEST = _load_backtest_table()


def get_asset_edge(ticker: str) -> Dict:
    """Return historical track record for this asset from the backtest."""
    r = _BACKTEST.get(ticker, {})
    if not r or "error" in r:
        return {}
    return {
        "ann_return":  r.get("annualized_return"),
        "sharpe":      r.get("sharpe_ratio"),
        "win_rate":    r.get("win_rate"),
        "max_dd":      r.get("max_drawdown"),
        "trades":      r.get("total_trades"),
    }


# ---------------------------------------------------------------------------
# Price + expected move (ATR-based, honest about what we can predict)
# ---------------------------------------------------------------------------

def expected_move(price_history: pd.DataFrame, composite: float,
                  horizon_days: int = DEFAULT_HORIZON_DAYS) -> Dict:
    """
    Compute expected move using ATR scaled by horizon and signal strength.

    This is a volatility-based estimate, NOT a price prediction. It answers:
    "given this asset's recent realised volatility and how strong the signal is,
    what's a reasonable magnitude for the next N days?" — not "where will price be."

    Returns:
        {
            "horizon_days": int,
            "expected_pct": float,   # signed, in decimal form (+0.03 = +3%)
            "expected_dollars": float,
            "atr_pct": float,        # daily ATR as % of price (for context)
            "current_price": float,
        }
    """
    if price_history is None or price_history.empty or len(price_history) < 14:
        return {}

    try:
        high  = price_history["High"]
        low   = price_history["Low"]
        close = price_history["Close"]
        prev_close = close.shift(1)

        tr = pd.concat([high - low,
                        (high - prev_close).abs(),
                        (low  - prev_close).abs()], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean().iloc[-1]
        price = float(close.iloc[-1])
        atr_pct_daily = float(atr14) / price if price else 0.0

        # Volatility scales with sqrt(time). Signal strength (|composite|) sizes
        # the expected magnitude as a fraction of the horizon-scaled ATR move.
        horizon_atr = atr_pct_daily * math.sqrt(horizon_days)
        magnitude   = horizon_atr * max(0.5, min(abs(composite), 1.0))
        signed_pct  = magnitude * (1 if composite > 0 else -1)
        dollar_move = signed_pct * price

        return {
            "horizon_days":    horizon_days,
            "expected_pct":    signed_pct,
            "expected_dollars": dollar_move,
            "atr_pct":         atr_pct_daily,
            "current_price":   price,
        }
    except Exception as e:
        logger.warning(f"expected_move failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Option contract suggestion (Polygon)
# ---------------------------------------------------------------------------

_polygon_client = None

def _get_polygon():
    global _polygon_client
    if _polygon_client is not None:
        return _polygon_client
    try:
        from polygon import RESTClient
        key = os.getenv("POLYGON_API_KEY")
        if not key:
            return None
        _polygon_client = RESTClient(key)
        return _polygon_client
    except ImportError:
        return None


def suggest_option(ticker: str, current_price: float, composite: float,
                   target_dte_days: int = 10) -> Dict:
    """
    Suggest a near-the-money option contract matching the signal direction.

      HYPE → 1-strike-OTM call expiring 7-14 days out
      FEAR → 1-strike-OTM put  expiring 7-14 days out

    Crypto and many ETFs don't have options — returns {} cleanly in that case.
    """
    client = _get_polygon()
    if client is None or not current_price or "-USD" in ticker:
        return {}

    contract_type = "call" if composite > 0 else "put"
    today = datetime.utcnow().date()
    min_exp = (today + timedelta(days=max(target_dte_days - 5, 5))).isoformat()
    max_exp = (today + timedelta(days=target_dte_days + 8)).isoformat()

    try:
        contracts = list(client.list_options_contracts(
            underlying_ticker=ticker,
            contract_type=contract_type,
            expiration_date_gte=min_exp,
            expiration_date_lte=max_exp,
            limit=200,
        ))
        if not contracts:
            return {}

        # 1-strike OTM in the signal direction.
        # For calls we want strike > price, for puts strike < price.
        if contract_type == "call":
            candidates = [c for c in contracts if c.strike_price > current_price]
            candidates.sort(key=lambda c: (c.strike_price - current_price,
                                            c.expiration_date))
        else:
            candidates = [c for c in contracts if c.strike_price < current_price]
            candidates.sort(key=lambda c: (current_price - c.strike_price,
                                            c.expiration_date))

        if not candidates:
            return {}
        c = candidates[0]

        # Try to fetch last quote (free tier may not include this in real-time)
        last_price = None
        try:
            quote = client.get_last_trade(c.ticker)
            last_price = float(getattr(quote, "price", None) or 0) or None
        except Exception:
            pass

        breakeven = (c.strike_price + (last_price or 0)) if contract_type == "call" \
                    else (c.strike_price - (last_price or 0))

        return {
            "contract_symbol": c.ticker,
            "type":            contract_type,
            "strike":          c.strike_price,
            "expiration":      c.expiration_date,
            "last_price":      last_price,
            "breakeven":       breakeven if last_price else None,
            "underlying":      ticker,
        }
    except Exception as e:
        logger.warning(f"suggest_option failed for {ticker}: {e}")
        return {}


# ---------------------------------------------------------------------------
# All-in-one enrichment
# ---------------------------------------------------------------------------

def enrich(ticker: str, composite: float,
           price_history: pd.DataFrame) -> Dict:
    """
    Compute all enrichment fields for a triggered alert. Returns a dict with
    everything; missing pieces are simply absent.
    """
    out: Dict = {}
    move = expected_move(price_history, composite)
    if move:
        out.update(move)
        out["option"] = suggest_option(ticker, move["current_price"], composite)
    out["edge"] = get_asset_edge(ticker)
    return out


# ---------------------------------------------------------------------------
# Telegram-ready formatting
# ---------------------------------------------------------------------------

def format_enrichment_block(enriched: Dict) -> str:
    """Format enrichment fields as a Telegram-Markdown string."""
    lines = []
    price = enriched.get("current_price")
    if price:
        lines.append(f"Price: `${price:,.2f}`")

    horizon = enriched.get("horizon_days")
    pct     = enriched.get("expected_pct")
    dollars = enriched.get("expected_dollars")
    if horizon and pct is not None:
        lines.append(
            f"Expected {horizon}d move: `{pct:+.2%}` (≈ `${dollars:+.2f}`)"
        )

    edge = enriched.get("edge") or {}
    if edge and edge.get("sharpe") is not None:
        lines.append(
            "Historical edge: "
            f"Sharpe `{edge['sharpe']:.2f}` · "
            f"Win `{edge.get('win_rate', 0):.0%}` · "
            f"AnnRet `{edge.get('ann_return', 0):+.1%}`"
        )

    opt = enriched.get("option") or {}
    if opt and opt.get("contract_symbol"):
        strike     = opt.get("strike")
        exp        = opt.get("expiration")
        ctype      = opt.get("type", "").upper()
        last_price = opt.get("last_price")
        breakeven  = opt.get("breakeven")

        line = f"Option (info only): `{opt['contract_symbol']}` "
        line += f"({ctype} ${strike} exp {exp})"
        lines.append(line)
        if last_price:
            be_str = f" · breakeven `${breakeven:,.2f}`" if breakeven else ""
            lines.append(f"  Last: `${last_price:.2f}`{be_str}")

    return "\n".join(lines) if lines else ""
