"""
Hype-Fear Alert System v4 — Real-Time Monitor

Runs a continuous loop that:
  1. Scores each asset via Grok sentiment + technical signals
  2. Applies per-asset sensitivity profiles
  3. Fires Telegram alerts when hype/fear thresholds are crossed
  4. Respects cooldown windows to avoid alert spam

Required env vars:
  XAI_API_KEY          — xAI / Grok API key
  TELEGRAM_BOT_TOKEN   — Telegram bot token
  TELEGRAM_CHAT_ID     — Target chat/channel ID
  POLYGON_API_KEY      — (optional) Polygon for options flow
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from grok_sentiment import get_live_sentiment, format_sentiment_message

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCAN_INTERVAL_SECS  = 300          # 5 min between full scans
ALERT_COOLDOWN_SECS = 3600         # 1 hr cooldown per asset per direction
HYPE_THRESHOLD      = 0.45         # composite score above this → hype alert
FEAR_THRESHOLD      = -0.45        # composite score below this → fear alert
CONFIDENCE_MIN      = 0.40         # minimum Grok confidence to fire

WATCHLIST = [
    "TSLA", "NVDA", "AAPL", "AMZN", "META", "GOOGL", "AMD", "MSFT",
    "SPY", "QQQ", "GME",
    "BTC-USD", "ETH-USD", "SOL-USD",
    "TLT", "VOO", "SCHD", "TQQQ",
]

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def send_telegram(message: str) -> bool:
    token   = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.warning("Telegram not configured — printing to stdout instead")
        print(f"\n[ALERT]\n{message}\n")
        return False
    url  = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, data=data, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Technical signal (fast, no ML — for real-time use)
# ---------------------------------------------------------------------------

def compute_fast_tech_signal(ticker: str) -> Tuple[float, str]:
    """
    Returns (signal, regime) using the last 60 days of price data.
    signal: -1 to +1
    regime: 'bull' | 'bear' | 'ranging' | 'crisis'
    """
    try:
        df = yf.download(ticker, period="90d", interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 20:
            return 0.0, "unknown"

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close  = df["Close"]
        volume = df.get("Volume", pd.Series(1, index=df.index))

        # Technical indicators
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean() if len(df) >= 50 else sma20
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd  = ema12 - ema26
        macd_sig = macd.ewm(span=9).mean()

        delta = close.diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi   = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))

        bb_mid   = close.rolling(20).mean()
        bb_std   = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std

        c    = close.iloc[-1]
        sig  = 0.0

        # RSI
        r = rsi.iloc[-1]
        if r < 30:   sig += 0.40
        elif r > 70: sig -= 0.40

        # MACD
        if macd.iloc[-1] > macd_sig.iloc[-1]:  sig += 0.25
        else:                                    sig -= 0.25

        # Bollinger
        if c < bb_lower.iloc[-1]:  sig += 0.20
        elif c > bb_upper.iloc[-1]: sig -= 0.20

        # Trend
        if sma20.iloc[-1] > sma50.iloc[-1]: sig += 0.15
        else:                                sig -= 0.15

        # Volume confirmation
        vol_ratio = volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]
        if vol_ratio > 1.8: sig *= 1.25

        sig = float(np.clip(sig, -1, 1))

        # Regime
        ret20  = (c / close.iloc[-21] - 1) if len(close) > 21 else 0
        vol20  = close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100
        ret10  = (c / close.iloc[-11] - 1) if len(close) > 11 else 0
        ma50v  = sma50.iloc[-1]
        ma200v = close.rolling(min(200, len(close))).mean().iloc[-1]

        if vol20 > 35 or ret10 < -0.10:
            regime = "crisis"
        elif sma20.iloc[-1] > ma200v and ret20 > 0.02:
            regime = "bull"
        elif sma20.iloc[-1] < ma200v and ret20 < -0.02:
            regime = "bear"
        else:
            regime = "ranging"

        return sig, regime

    except Exception as e:
        logger.error(f"Tech signal error for {ticker}: {e}")
        return 0.0, "unknown"


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

def compute_composite_score(ticker: str, profile: Dict,
                             grok_result: Dict) -> Dict:
    """Combine Grok sentiment + technical signal into a single composite score."""
    grok_score  = grok_result.get("score", 0.0)
    grok_conf   = grok_result.get("confidence", 0.0)
    grok_summary = grok_result.get("summary", "")

    tech_signal, regime = compute_fast_tech_signal(ticker)

    # Weight sentiments by profile sensitivity + regime
    sent_sensitivity = profile.get("hype_sensitivity", 1.0)
    tech_sensitivity = profile.get("technical_sensitivity", 1.0)

    # Regime-aware weights
    regime_tech_w = {"bull": 0.50, "bear": 0.30, "ranging": 0.45, "crisis": 0.20, "unknown": 0.40}
    regime_sent_w = {"bull": 0.35, "bear": 0.50, "ranging": 0.30, "crisis": 0.60, "unknown": 0.40}

    tw = regime_tech_w.get(regime, 0.40) * tech_sensitivity
    sw = regime_sent_w.get(regime, 0.40) * sent_sensitivity
    total_w = tw + sw or 1.0

    composite = (tech_signal * tw + grok_score * sw) / total_w
    composite = float(np.clip(composite, -1, 1))

    # Confidence: blend Grok confidence with technical conviction
    tech_conviction = min(abs(tech_signal), 1.0)
    blended_conf    = (grok_conf * 0.6 + tech_conviction * 0.4)

    return {
        "ticker":        ticker,
        "composite":     composite,
        "confidence":    blended_conf,
        "grok_score":    grok_score,
        "grok_conf":     grok_conf,
        "tech_signal":   tech_signal,
        "regime":        regime,
        "summary":       grok_summary,
        "timestamp":     datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Alert engine
# ---------------------------------------------------------------------------

class AlertEngine:

    def __init__(self, profiles_path: str = "profiles.json"):
        self.profiles       = self._load_profiles(profiles_path)
        self.last_alert_ts  = {}   # (ticker, direction) → timestamp
        self.alert_history  = []

    def _load_profiles(self, path: str) -> Dict:
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Could not load profiles: {e}")
            return {}

    def _in_cooldown(self, ticker: str, direction: str) -> bool:
        key = (ticker, direction)
        last = self.last_alert_ts.get(key)
        if last is None:
            return False
        return (datetime.utcnow() - last).total_seconds() < ALERT_COOLDOWN_SECS

    def _record_alert(self, ticker: str, direction: str) -> None:
        self.last_alert_ts[(ticker, direction)] = datetime.utcnow()

    def evaluate(self, ticker: str, scores: Dict) -> Optional[str]:
        """Return an alert message string if threshold crossed, else None."""
        composite  = scores["composite"]
        confidence = scores["confidence"]
        regime     = scores["regime"]

        if confidence < CONFIDENCE_MIN:
            return None

        if composite >= HYPE_THRESHOLD and not self._in_cooldown(ticker, "hype"):
            direction = "HYPE 🚀"
            emoji     = "📈"
            alert_type = "hype"
        elif composite <= FEAR_THRESHOLD and not self._in_cooldown(ticker, "fear"):
            direction = "FEAR 🔻"
            emoji     = "📉"
            alert_type = "fear"
        else:
            return None

        self._record_alert(ticker, alert_type)

        bars = int(abs(composite) * 10)
        bar  = ("🟢" if composite > 0 else "🔴") * bars + "⬜" * (10 - bars)

        msg = (
            f"{emoji} *{ticker} {direction}*\n"
            f"{bar}\n"
            f"Composite: `{composite:+.3f}`  Confidence: `{confidence:.0%}`\n"
            f"Tech: `{scores['tech_signal']:+.2f}`  Grok: `{scores['grok_score']:+.2f}`\n"
            f"Regime: `{regime.upper()}`\n"
            f"_{scores['summary']}_\n"
            f"`{scores['timestamp'][:19]} UTC`"
        )
        return msg

    def run_scan(self, watchlist: List[str]) -> List[str]:
        """Scan all assets and return list of alert messages fired."""
        from grok_sentiment import get_live_sentiment_batch
        logger.info(f"Scanning {len(watchlist)} assets...")

        grok_results = get_live_sentiment_batch(watchlist, delay_secs=0.3)

        alerts_fired = []
        for ticker in watchlist:
            profile = self.profiles.get(ticker, {})
            grok    = grok_results.get(ticker, {"score": 0.0, "confidence": 0.0, "summary": ""})
            scores  = compute_composite_score(ticker, profile, grok)

            logger.info(
                f"{ticker:10s} composite={scores['composite']:+.3f} "
                f"conf={scores['confidence']:.0%} regime={scores['regime']}"
            )

            alert_msg = self.evaluate(ticker, scores)
            if alert_msg:
                logger.info(f"ALERT: {ticker}")
                send_telegram(alert_msg)
                alerts_fired.append(alert_msg)
                self.alert_history.append({"ticker": ticker, "scores": scores, "msg": alert_msg})

        return alerts_fired


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    logger.info("Hype-Fear Alert System v4 starting...")
    engine = AlertEngine()

    scan_count = 0
    while True:
        scan_count += 1
        logger.info(f"=== Scan #{scan_count} — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} ===")

        try:
            alerts = engine.run_scan(WATCHLIST)
            logger.info(f"Scan complete — {len(alerts)} alert(s) fired")
        except Exception as e:
            logger.error(f"Scan error: {e}")

        logger.info(f"Sleeping {SCAN_INTERVAL_SECS}s until next scan...")
        time.sleep(SCAN_INTERVAL_SECS)


if __name__ == "__main__":
    main()
