"""
External signal collectors for Google Trends and funding rates.
Provides additional data sources for the hype-fear alert system.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import os

try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False
    logging.warning("pytrends not available. Install with: pip install pytrends")

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logging.warning("httpx not available for funding rate fetching")

logger = logging.getLogger(__name__)


class GoogleTrendsCollector:
    """Collects Google Trends data for assets."""

    def __init__(self, hl='en-US', tz=360, timeout=(10, 25)):
        if not PYTRENDS_AVAILABLE:
            raise ImportError("pytrends is required for Google Trends collection")

        self.pytrends = TrendReq(hl=hl, tz=tz, timeout=timeout)
        self._keyword_cache = {}

        # Define search keywords for each asset
        self.asset_keywords = {
            'TSLA': ['Tesla', 'Tesla stock', 'TSLA'],
            'NVDA': ['NVIDIA', 'NVIDIA stock', 'NVDA'],
            'AAPL': ['Apple', 'Apple stock', 'AAPL'],
            'AMZN': ['Amazon', 'Amazon stock', 'AMZN'],
            'META': ['Meta', 'Facebook stock', 'META'],
            'GOOGL': ['Google', 'Alphabet stock', 'GOOGL'],
            'AMD': ['AMD', 'AMD stock'],
            'MSFT': ['Microsoft', 'Microsoft stock', 'MSFT'],
            'SPY': ['S&P 500', 'SPY ETF'],
            'QQQ': ['Nasdaq 100', 'QQQ ETF'],
            'GME': ['GameStop', 'GameStop stock', 'GME'],
            'BTC-USD': ['Bitcoin', 'BTC', 'Bitcoin price'],
            'ETH-USD': ['Ethereum', 'ETH', 'Ethereum price'],
            'SOL-USD': ['Solana', 'SOL', 'Solana price']
        }

    def get_trends_data(self, asset: str, timeframe: str = 'today 3-m') -> Optional[Dict]:
        """
        Get Google Trends interest over time for an asset.

        Args:
            asset: Asset ticker/symbol
            timeframe: Timeframe for trends data (default: last 3 months)

        Returns:
            Dict with trends data and metadata, or None if failed
        """
        if asset not in self.asset_keywords:
            logger.warning(f"No keywords defined for asset: {asset}")
            return None

        keywords = self.asset_keywords[asset]

        try:
            # Build payload with keywords
            self.pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo='', gprop='')

            # Get interest over time
            interest_data = self.pytrends.interest_over_time()

            if interest_data.empty:
                logger.warning(f"No trends data found for {asset}")
                return None

            # Calculate average interest and recent momentum
            avg_interest = interest_data[keywords[0]].mean()
            recent_avg = interest_data[keywords[0]].tail(7).mean()  # Last week
            prev_avg = interest_data[keywords[0]].head(-7).tail(7).mean()  # Previous week

            momentum = ((recent_avg - prev_avg) / prev_avg) if prev_avg > 0 else 0

            return {
                'asset': asset,
                'avg_interest': float(avg_interest),
                'recent_momentum': float(momentum),
                'max_interest': float(interest_data[keywords[0]].max()),
                'timestamp': datetime.now().isoformat(),
                'keywords_used': keywords,
                'timeframe': timeframe
            }

        except Exception as e:
            logger.error(f"Failed to fetch Google Trends for {asset}: {e}")
            return None

    def get_multiple_assets(self, assets: List[str], timeframe: str = 'today 3-m') -> Dict[str, Dict]:
        """
        Get trends data for multiple assets with rate limiting.

        Args:
            assets: List of asset tickers
            timeframe: Timeframe for trends data

        Returns:
            Dict mapping assets to their trends data
        """
        results = {}

        for asset in assets:
            data = self.get_trends_data(asset, timeframe)
            if data:
                results[asset] = data

            # Rate limiting - Google Trends can be slow
            time.sleep(1)

        return results


class FundingRateCollector:
    """Collects crypto funding rates from exchanges."""

    def __init__(self):
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required for funding rate collection")

        self.client = httpx.Client(timeout=10.0)
        self._rate_cache = {}
        self._cache_timeout = 300  # 5 minutes

        # Exchange endpoints for funding rates
        self.endpoints = {
            'binance': {
                'BTCUSDT': 'https://fapi.binance.com/fapi/v1/premiumIndex',
                'ETHUSDT': 'https://fapi.binance.com/fapi/v1/premiumIndex',
                'SOLUSDT': 'https://fapi.binance.com/fapi/v1/premiumIndex'
            },
            'bybit': {
                'BTCUSDT': 'https://api.bybit.com/v5/market/tickers?category=linear&symbol=BTCUSDT',
                'ETHUSDT': 'https://api.bybit.com/v5/market/tickers?category=linear&symbol=ETHUSDT',
                'SOLUSDT': 'https://api.bybit.com/v5/market/tickers?category=linear&symbol=SOLUSDT'
            }
        }

    def _get_cached_rate(self, exchange: str, symbol: str) -> Optional[float]:
        """Get cached funding rate if still valid."""
        cache_key = f"{exchange}_{symbol}"
        if cache_key in self._rate_cache:
            cached_time, rate = self._rate_cache[cache_key]
            if time.time() - cached_time < self._cache_timeout:
                return rate
        return None

    def _cache_rate(self, exchange: str, symbol: str, rate: float):
        """Cache funding rate with timestamp."""
        cache_key = f"{exchange}_{symbol}"
        self._rate_cache[cache_key] = (time.time(), rate)

    def get_binance_funding_rate(self, symbol: str) -> Optional[float]:
        """Get funding rate from Binance."""
        cached = self._get_cached_rate('binance', symbol)
        if cached is not None:
            return cached

        try:
            url = f"{self.endpoints['binance'][symbol]}?symbol={symbol}"
            response = self.client.get(url)
            response.raise_for_status()

            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                rate = float(data[0].get('lastFundingRate', 0))
            else:
                rate = float(data.get('lastFundingRate', 0))

            self._cache_rate('binance', symbol, rate)
            return rate

        except Exception as e:
            logger.error(f"Failed to fetch Binance funding rate for {symbol}: {e}")
            return None

    def get_bybit_funding_rate(self, symbol: str) -> Optional[float]:
        """Get funding rate from Bybit."""
        cached = self._get_cached_rate('bybit', symbol)
        if cached is not None:
            return cached

        try:
            url = self.endpoints['bybit'][symbol]
            response = self.client.get(url)
            response.raise_for_status()

            data = response.json()
            if data.get('retCode') == 0:
                tickers = data.get('result', {}).get('list', [])
                if tickers:
                    rate = float(tickers[0].get('fundingRate', 0))
                    self._cache_rate('bybit', symbol, rate)
                    return rate

            return None

        except Exception as e:
            logger.error(f"Failed to fetch Bybit funding rate for {symbol}: {e}")
            return None

    def get_funding_rate(self, asset: str) -> Optional[Dict]:
        """
        Get funding rate data for a crypto asset.

        Args:
            asset: Asset symbol (BTC-USD, ETH-USD, SOL-USD)

        Returns:
            Dict with funding rate data from multiple exchanges
        """
        # Map asset to exchange symbols
        symbol_map = {
            'BTC-USD': 'BTCUSDT',
            'ETH-USD': 'ETHUSDT',
            'SOL-USD': 'SOLUSDT'
        }

        if asset not in symbol_map:
            logger.warning(f"No funding rate mapping for asset: {asset}")
            return None

        symbol = symbol_map[asset]

        # Get rates from multiple exchanges
        binance_rate = self.get_binance_funding_rate(symbol)
        bybit_rate = self.get_bybit_funding_rate(symbol)

        rates = []
        if binance_rate is not None:
            rates.append({'exchange': 'binance', 'rate': binance_rate})
        if bybit_rate is not None:
            rates.append({'exchange': 'bybit', 'rate': bybit_rate})

        if not rates:
            return None

        # Calculate average rate
        avg_rate = sum(r['rate'] for r in rates) / len(rates)

        # Determine sentiment (positive = bullish, negative = bearish)
        sentiment = 'bullish' if avg_rate > 0 else 'bearish'

        # Extreme funding rates can be contrarian signals
        extreme_threshold = 0.01  # 1%
        is_extreme = abs(avg_rate) > extreme_threshold

        return {
            'asset': asset,
            'symbol': symbol,
            'avg_funding_rate': avg_rate,
            'sentiment': sentiment,
            'is_extreme': is_extreme,
            'exchange_rates': rates,
            'timestamp': datetime.now().isoformat()
        }

    def get_multiple_assets(self, assets: List[str]) -> Dict[str, Dict]:
        """
        Get funding rates for multiple crypto assets.

        Args:
            assets: List of crypto asset symbols

        Returns:
            Dict mapping assets to their funding rate data
        """
        results = {}

        for asset in assets:
            data = self.get_funding_rate(asset)
            if data:
                results[asset] = data

            # Small delay between requests
            time.sleep(0.5)

        return results


class ExternalSignalsManager:
    """Manages collection of external signals (Google Trends, funding rates)."""

    def __init__(self):
        self.trends_collector = None
        self.funding_collector = None

        # Initialize collectors if dependencies available
        try:
            self.trends_collector = GoogleTrendsCollector()
        except ImportError:
            logger.warning("Google Trends collector not available")

        try:
            self.funding_collector = FundingRateCollector()
        except ImportError:
            logger.warning("Funding rate collector not available")

    def collect_all_signals(self, assets: List[str]) -> Dict[str, Dict]:
        """
        Collect all available external signals for given assets.

        Args:
            assets: List of asset symbols

        Returns:
            Dict with signal data organized by asset and signal type
        """
        results = {}

        # Separate crypto and non-crypto assets
        crypto_assets = [a for a in assets if a in ['BTC-USD', 'ETH-USD', 'SOL-USD']]
        other_assets = [a for a in assets if a not in crypto_assets]

        # Collect Google Trends for all assets
        if self.trends_collector:
            logger.info(f"Collecting Google Trends for {len(assets)} assets")
            trends_data = self.trends_collector.get_multiple_assets(assets)
            for asset, data in trends_data.items():
                if asset not in results:
                    results[asset] = {}
                results[asset]['google_trends'] = data

        # Collect funding rates for crypto assets
        if self.funding_collector and crypto_assets:
            logger.info(f"Collecting funding rates for {len(crypto_assets)} crypto assets")
            funding_data = self.funding_collector.get_multiple_assets(crypto_assets)
            for asset, data in funding_data.items():
                if asset not in results:
                    results[asset] = {}
                results[asset]['funding_rate'] = data

        return results


# Convenience functions for easy integration
def get_google_trends_signals(assets: List[str]) -> Dict[str, Dict]:
    """Get Google Trends signals for assets."""
    if not PYTRENDS_AVAILABLE:
        return {}
    collector = GoogleTrendsCollector()
    return collector.get_multiple_assets(assets)


def get_funding_rate_signals(assets: List[str]) -> Dict[str, Dict]:
    """Get funding rate signals for crypto assets."""
    if not HTTPX_AVAILABLE:
        return {}
    collector = FundingRateCollector()
    crypto_assets = [a for a in assets if a in ['BTC-USD', 'ETH-USD', 'SOL-USD']]
    return collector.get_multiple_assets(crypto_assets)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    assets = ['TSLA', 'BTC-USD', 'ETH-USD']

    manager = ExternalSignalsManager()
    signals = manager.collect_all_signals(assets)

    print(json.dumps(signals, indent=2))