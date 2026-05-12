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

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from grok_sentiment import get_live_sentiment, format_sentiment_message
from obsidian_logger import log_sentiment_scan, log_alert, log_session_summary, ensure_daily_note

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCAN_INTERVAL_SECS  = int(os.getenv("SCAN_INTERVAL_SECS", 300))   # 5 min between full scans
ALERT_COOLDOWN_SECS = int(os.getenv("ALERT_COOLDOWN_SECS", 3600)) # 1 hr cooldown per asset per direction
HYPE_THRESHOLD      = float(os.getenv("HYPE_THRESHOLD", 0.45))
FEAR_THRESHOLD      = float(os.getenv("FEAR_THRESHOLD", -0.45))
CONFIDENCE_MIN      = float(os.getenv("CONFIDENCE_MIN", 0.40))
PRICE_CACHE_TTL     = int(os.getenv("PRICE_CACHE_TTL", 240))      # 4 min — < scan interval
STATE_DIR           = Path(os.getenv("STATE_DIR", "./state"))
STATE_DIR.mkdir(parents=True, exist_ok=True)
COOLDOWN_FILE = STATE_DIR / "cooldown_state.json"
HISTORY_FILE  = STATE_DIR / "alert_history.json"

# Graceful shutdown flag — set by SIGTERM/SIGINT handler
_shutdown = False

def _handle_signal(signum, frame):
    global _shutdown
    logger.info(f"Received signal {signum} — shutting down gracefully after current scan")
    _shutdown = True

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT,  _handle_signal)

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
# Price cache — avoid re-downloading 90d of OHLCV every scan
# ---------------------------------------------------------------------------

_price_cache: Dict[str, Tuple[float, pd.DataFrame]] = {}

def _cached_download(ticker: str, period: str = "90d") -> pd.DataFrame:
    now = time.time()
    cached = _price_cache.get(ticker)
    if cached and (now - cached[0]) < PRICE_CACHE_TTL:
        return cached[1]
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    _price_cache[ticker] = (now, df)
    return df


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
        df = _cached_download(ticker)
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

        # Regime — uses asset-relative vol so high-beta names don't permanently
        # register as 'crisis'. Crisis requires both elevated absolute vol AND
        # vol above this asset's own 90d high water mark.
        ret20  = (c / close.iloc[-21] - 1) if len(close) > 21 else 0
        vol_series = close.pct_change().rolling(20).std() * np.sqrt(252) * 100
        vol20  = vol_series.iloc[-1]
        vol_recent_high = vol_series.iloc[-min(90, len(vol_series)):].max() or vol20
        vol_ratio = vol20 / vol_recent_high if vol_recent_high else 1.0
        ret10  = (c / close.iloc[-11] - 1) if len(close) > 11 else 0
        ma200v = close.rolling(min(200, len(close))).mean().iloc[-1]

        if (vol20 > 25 and vol_ratio > 0.90) or ret10 < -0.10:
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
        self.last_alert_ts  = self._load_cooldown()   # (ticker, direction) → datetime
        self.alert_history  = self._load_history()

    def _load_profiles(self, path: str) -> Dict:
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Could not load profiles: {e}")
            return {}

    def _load_cooldown(self) -> Dict[Tuple[str, str], datetime]:
        if not COOLDOWN_FILE.exists():
            return {}
        try:
            raw = json.loads(COOLDOWN_FILE.read_text())
            return {tuple(k.split("|")): datetime.fromisoformat(v) for k, v in raw.items()}
        except Exception as e:
            logger.warning(f"Could not load cooldown state: {e}")
            return {}

    def _save_cooldown(self) -> None:
        try:
            serial = {f"{t}|{d}": ts.isoformat() for (t, d), ts in self.last_alert_ts.items()}
            COOLDOWN_FILE.write_text(json.dumps(serial, indent=2))
        except Exception as e:
            logger.warning(f"Could not save cooldown state: {e}")

    def _load_history(self) -> List[Dict]:
        if not HISTORY_FILE.exists():
            return []
        try:
            data = json.loads(HISTORY_FILE.read_text())
            # Trim to last 500 entries to keep file small
            return data[-500:]
        except Exception as e:
            logger.warning(f"Could not load alert history: {e}")
            return []

    def _save_history(self) -> None:
        try:
            HISTORY_FILE.write_text(json.dumps(self.alert_history[-500:], indent=2, default=str))
        except Exception as e:
            logger.warning(f"Could not save alert history: {e}")

    def _in_cooldown(self, ticker: str, direction: str) -> bool:
        key = (ticker, direction)
        last = self.last_alert_ts.get(key)
        if last is None:
            return False
        return (datetime.utcnow() - last).total_seconds() < ALERT_COOLDOWN_SECS

    def _record_alert(self, ticker: str, direction: str) -> None:
        self.last_alert_ts[(ticker, direction)] = datetime.utcnow()
        self._save_cooldown()

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

        all_scores   = {}
        alerts_fired = []

        for ticker in watchlist:
            profile = self.profiles.get(ticker, {})
            grok    = grok_results.get(ticker, {"score": 0.0, "confidence": 0.0, "summary": ""})
            scores  = compute_composite_score(ticker, profile, grok)
            all_scores[ticker] = scores

            logger.info(
                f"{ticker:10s} composite={scores['composite']:+.3f} "
                f"conf={scores['confidence']:.0%} regime={scores['regime']}"
            )

            alert_msg = self.evaluate(ticker, scores)
            if alert_msg:
                direction = "HYPE" if scores["composite"] > 0 else "FEAR"
                logger.info(f"ALERT: {ticker}")
                send_telegram(alert_msg)
                log_alert(ticker, direction, scores, alert_msg)
                alerts_fired.append(alert_msg)
                self.alert_history.append({
                    "ticker": ticker,
                    "scores": scores,
                    "msg": alert_msg,
                    "fired_at": datetime.utcnow().isoformat(),
                })

        if alerts_fired:
            self._save_history()

        # Log full scan to Obsidian daily note (best-effort)
        try:
            log_sentiment_scan(all_scores)
        except Exception as e:
            logger.warning(f"Obsidian log failed: {e}")

        return alerts_fired


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_single_scan(engine: Optional[AlertEngine] = None) -> int:
    """Execute one scan and return the number of alerts fired. For cron/one-shot use."""
    engine = engine or AlertEngine()
    try:
        ensure_daily_note()
    except Exception as e:
        logger.warning(f"ensure_daily_note failed: {e}")
    alerts = engine.run_scan(WATCHLIST)
    logger.info(f"Scan complete — {len(alerts)} alert(s) fired")
    return len(alerts)


def _interruptible_sleep(seconds: int) -> None:
    """Sleep in 1s chunks so SIGTERM is honored quickly."""
    for _ in range(seconds):
        if _shutdown:
            return
        time.sleep(1)


def run_loop() -> None:
    logger.info("Hype-Fear Alert System starting in loop mode...")
    try:
        ensure_daily_note()
    except Exception as e:
        logger.warning(f"ensure_daily_note failed: {e}")
    engine     = AlertEngine()
    scan_count = 0
    all_alerts: List[str] = []

    while not _shutdown:
        scan_count += 1
        logger.info(f"=== Scan #{scan_count} — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} ===")

        try:
            alerts = engine.run_scan(WATCHLIST)
            all_alerts.extend(alerts)
            logger.info(f"Scan complete — {len(alerts)} alert(s) fired")
        except Exception as e:
            logger.error(f"Scan error: {e}", exc_info=True)

        # Write session summary every 12 scans (~1 hour)
        if scan_count % 12 == 0 and engine.alert_history:
            top = sorted(
                engine.alert_history[-12:],
                key=lambda x: abs(x["scores"].get("composite", 0)),
                reverse=True,
            )
            try:
                log_session_summary(scan_count, len(all_alerts),
                                    [x["ticker"] for x in top[:3]])
            except Exception as e:
                logger.warning(f"Session summary log failed: {e}")

        if _shutdown:
            break
        logger.info(f"Sleeping {SCAN_INTERVAL_SECS}s until next scan...")
        _interruptible_sleep(SCAN_INTERVAL_SECS)

    logger.info("Shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="Hype-Fear Alert System")
    parser.add_argument("--once", action="store_true",
                        help="Run a single scan and exit (for cron deploys)")
    args = parser.parse_args()

    if args.once:
        run_single_scan()
        return
    run_loop()


if __name__ == "__main__":
    main()
