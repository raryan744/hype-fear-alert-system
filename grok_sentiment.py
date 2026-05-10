"""
Grok-powered real-time sentiment analysis for the Hype-Fear Alert System.

Uses the xAI API (OpenAI-compatible) to score market sentiment from
live news headlines and X/social content for each tracked asset.

Set env var:  XAI_API_KEY=your_key
"""

import os
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from functools import lru_cache

import yfinance as yf

try:
    from openai import OpenAI
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False

logger = logging.getLogger(__name__)

XAI_BASE_URL = "https://api.x.ai/v1"
GROK_MODEL   = "grok-3-mini"   # fast + cheap; swap to "grok-3" for deeper analysis

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def _get_client() -> Optional["OpenAI"]:
    if not OPENAI_SDK_AVAILABLE:
        logger.warning("openai SDK not installed — run: pip install openai")
        return None
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        logger.warning("XAI_API_KEY not set — Grok sentiment disabled")
        return None
    return OpenAI(api_key=api_key, base_url=XAI_BASE_URL)


# ---------------------------------------------------------------------------
# News fetching (yfinance — free, no key required)
# ---------------------------------------------------------------------------

def fetch_headlines(ticker: str, max_items: int = 15) -> List[str]:
    """Return recent news headlines for a ticker via yfinance."""
    try:
        t = yf.Ticker(ticker)
        news = t.news or []
        headlines = []
        for item in news[:max_items]:
            # yfinance ≥0.2.50 nests content; older versions had flat title key
            content = item.get("content", item)
            title = content.get("title", "") or item.get("title", "")
            if title:
                headlines.append(title)
        return headlines
    except Exception as e:
        logger.warning(f"Could not fetch headlines for {ticker}: {e}")
        return []


# ---------------------------------------------------------------------------
# Grok scoring
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a financial sentiment analyst. Given a list of news headlines about a stock or crypto asset, \
output a single JSON object with these fields:
  "score"     : float from -1.0 (extreme fear/bearish) to +1.0 (extreme hype/bullish)
  "confidence": float from 0.0 to 1.0 indicating how clear the signal is
  "summary"   : one sentence explaining the dominant sentiment driver

Only return valid JSON. No markdown, no explanation outside the JSON."""


def score_headlines_with_grok(ticker: str, headlines: List[str]) -> Dict:
    """Send headlines to Grok and return a sentiment score dict."""
    default = {"score": 0.0, "confidence": 0.0, "summary": "No signal", "source": "default"}

    if not headlines:
        return default

    client = _get_client()
    if client is None:
        return default

    headlines_text = "\n".join(f"- {h}" for h in headlines)
    user_msg = f"Asset: {ticker}\n\nHeadlines:\n{headlines_text}"

    try:
        resp = client.chat.completions.create(
            model=GROK_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()

        import json
        result = json.loads(raw)
        result["source"] = "grok"
        result["ticker"] = ticker
        result["timestamp"] = datetime.utcnow().isoformat()
        return result

    except Exception as e:
        logger.error(f"Grok API error for {ticker}: {e}")
        return default


# ---------------------------------------------------------------------------
# High-level interface
# ---------------------------------------------------------------------------

def get_live_sentiment(ticker: str) -> Dict:
    """
    Full pipeline: fetch headlines → score with Grok → return sentiment dict.

    Returns:
        {
            "ticker": str,
            "score": float,          # -1 to +1
            "confidence": float,     # 0 to 1
            "summary": str,
            "headlines": List[str],
            "source": "grok" | "default",
            "timestamp": str,
        }
    """
    headlines = fetch_headlines(ticker)
    result    = score_headlines_with_grok(ticker, headlines)
    result["headlines"] = headlines
    return result


def get_live_sentiment_batch(tickers: List[str], delay_secs: float = 0.5) -> Dict[str, Dict]:
    """
    Score multiple tickers with a small delay to respect rate limits.
    Returns dict keyed by ticker.
    """
    results = {}
    for ticker in tickers:
        results[ticker] = get_live_sentiment(ticker)
        if delay_secs > 0:
            time.sleep(delay_secs)
    return results


def format_sentiment_message(ticker: str, sentiment: Dict, profile: Dict) -> str:
    """Format a Telegram-ready sentiment line for a ticker."""
    score      = sentiment.get("score", 0)
    confidence = sentiment.get("confidence", 0)
    summary    = sentiment.get("summary", "")

    bar_len = 10
    filled  = int(abs(score) * bar_len)
    bar     = ("🟢" if score > 0 else "🔴") * filled + "⬜" * (bar_len - filled)

    direction = "HYPE 📈" if score > 0.3 else ("FEAR 📉" if score < -0.3 else "NEUTRAL ➡️")

    return (
        f"*{ticker}* — {direction}\n"
        f"{bar}  score={score:+.2f}  conf={confidence:.0%}\n"
        f"_{summary}_"
    )


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    test_tickers = ["TSLA", "BTC-USD", "NVDA"]
    print(f"Testing Grok sentiment for: {test_tickers}\n")

    results = get_live_sentiment_batch(test_tickers)
    for ticker, data in results.items():
        print(f"\n{'='*50}")
        print(f"Ticker   : {ticker}")
        print(f"Score    : {data['score']:+.3f}")
        print(f"Confidence: {data['confidence']:.1%}")
        print(f"Summary  : {data['summary']}")
        print(f"Source   : {data['source']}")
        print(f"Headlines: {len(data.get('headlines', []))} fetched")
