"""
Market Regime Detection — Rules-Based Implementation

Replaces the broken K-means classifier with a deterministic, interpretable approach:

  Bull   : 50d MA > 200d MA AND 20d return > +2%
  Bear   : 50d MA < 200d MA AND 20d return < -2%
  Crisis : 20d annualised vol > 35% OR 10d return < -10%
  Ranging: everything else (sideways / low-conviction)

Regime-specific signal weights and position sizing are defined in
RegimeAdaptiveStrategy and applied in AdaptiveSignalProcessor.
"""

import logging
from typing import Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------

class MarketRegimeClassifier:

    def fit(self, data: pd.DataFrame, n_regimes: int = 4) -> None:
        """No-op: rules-based classifier needs no training."""
        pass

    def predict_regime(self, data: pd.DataFrame) -> str:
        if data.empty or len(data) < 50:
            return 'ranging'
        return _classify(data.iloc[-1], data)

    def predict_regime_series(self, data: pd.DataFrame) -> pd.Series:
        """Return a per-row regime series for the full history."""
        features = _build_features(data)
        regimes = []
        for i in range(len(features)):
            row = features.iloc[i]
            regimes.append(_classify_row(row))
        return pd.Series(regimes, index=features.index)


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build regime features. Vol is normalised to each asset's own history so
    that naturally high-beta names (TSLA, crypto) don't get labelled 'crisis'
    just for being themselves.
    """
    out = pd.DataFrame(index=df.index)
    close = df['Close']

    out['ma50']      = close.rolling(50).mean()
    out['ma200']     = close.rolling(200).mean()
    out['ret20']     = close.pct_change(20)
    out['ret10']     = close.pct_change(10)
    out['vol20_ann'] = close.pct_change().rolling(20).std() * np.sqrt(252) * 100

    # Asset-relative vol: how high is current vol vs trailing year for THIS ticker?
    out['vol_p90_252d'] = out['vol20_ann'].rolling(252, min_periods=60).quantile(0.90)
    out['vol_ratio']    = out['vol20_ann'] / out['vol_p90_252d']

    return out.dropna()


def _classify_row(row: pd.Series) -> str:
    vol       = row['vol20_ann']
    ret10     = row['ret10']
    ret20     = row['ret20']
    ma50      = row['ma50']
    ma200     = row['ma200']
    vol_ratio = row.get('vol_ratio', 1.0)

    # Crisis: vol is BOTH high in absolute terms (>25%) AND in the top
    # decile of this asset's trailing-year history, OR a 10d drawdown >10%.
    # Old rule used a fixed 35% which fired permanently for TSLA, crypto, GME.
    if (vol > 25 and vol_ratio > 1.0) or ret10 < -0.10:
        return 'crisis'
    if ma50 > ma200 and ret20 > 0.02:
        return 'bull'
    if ma50 < ma200 and ret20 < -0.02:
        return 'bear'
    return 'ranging'


def _classify(last_row: pd.Series, data: pd.DataFrame) -> str:
    features = _build_features(data)
    if features.empty:
        return 'ranging'
    return _classify_row(features.iloc[-1])


# ---------------------------------------------------------------------------
# Regime-adaptive strategy
# ---------------------------------------------------------------------------

class RegimeAdaptiveStrategy:

    def __init__(self):
        self.regime_classifier = MarketRegimeClassifier()
        self.regime_strategies = {
            'bull': {
                'signal_weights': {'tech_weight': 0.50, 'sentiment_weight': 0.20,
                                   'options_weight': 0.10, 'macro_weight': 0.20},
                'position_size_multiplier': 1.4,
                'stop_loss_multiplier': 1.2,
                'trend_following': True,
                'momentum_bias': 0.8,
            },
            'bear': {
                'signal_weights': {'tech_weight': 0.30, 'sentiment_weight': 0.40,
                                   'options_weight': 0.20, 'macro_weight': 0.10},
                'position_size_multiplier': 0.65,
                'stop_loss_multiplier': 0.75,
                'trend_following': False,
                'momentum_bias': -0.4,
            },
            'ranging': {
                'signal_weights': {'tech_weight': 0.45, 'sentiment_weight': 0.15,
                                   'options_weight': 0.25, 'macro_weight': 0.15},
                'position_size_multiplier': 0.85,
                'stop_loss_multiplier': 0.65,
                'trend_following': False,
                'momentum_bias': 0.0,
            },
            'crisis': {
                'signal_weights': {'tech_weight': 0.20, 'sentiment_weight': 0.50,
                                   'options_weight': 0.20, 'macro_weight': 0.10},
                'position_size_multiplier': 0.25,
                'stop_loss_multiplier': 0.45,
                'trend_following': False,
                'momentum_bias': -1.0,
            },
            'unknown': {
                'signal_weights': {'tech_weight': 0.40, 'sentiment_weight': 0.30,
                                   'options_weight': 0.20, 'macro_weight': 0.10},
                'position_size_multiplier': 1.0,
                'stop_loss_multiplier': 1.0,
                'trend_following': False,
                'momentum_bias': 0.0,
            },
        }

    def train_regime_classifier(self, historical_data: pd.DataFrame) -> None:
        pass  # rules-based, no training needed

    def get_regime_adaptive_parameters(self, current_data: pd.DataFrame,
                                       base_profile: Dict) -> Dict[str, Any]:
        regime = self.regime_classifier.predict_regime(current_data)
        strategy = self.regime_strategies.get(regime, self.regime_strategies['unknown'])

        weights = self._apply_profile_sensitivities(
            strategy['signal_weights'].copy(), base_profile)

        base_pos  = base_profile.get('base_position_size', 1.0)
        base_stop = base_profile.get('base_stop_loss', 0.05)

        return {
            'regime': regime,
            'strategy_name': f"{regime.title()} Market Strategy",
            'signal_weights': weights,
            'position_size': base_pos * strategy['position_size_multiplier'],
            'stop_loss': base_stop * strategy['stop_loss_multiplier'],
            'momentum_bias': strategy['momentum_bias'],
            'trend_following': strategy['trend_following'],
        }

    def _apply_profile_sensitivities(self, weights: Dict, profile: Dict) -> Dict:
        adapted = weights.copy()
        adapted['sentiment_weight'] *= profile.get('sentiment_sensitivity', 1.0)
        adapted['tech_weight']      *= profile.get('technical_sensitivity', 1.0)
        adapted['options_weight']   *= profile.get('options_sensitivity', 1.0)
        total = sum(adapted.values())
        if total > 0:
            adapted = {k: v / total for k, v in adapted.items()}
        return adapted


# ---------------------------------------------------------------------------
# Adaptive signal processor
# ---------------------------------------------------------------------------

class AdaptiveSignalProcessor:

    def __init__(self):
        self.regime_strategy = RegimeAdaptiveStrategy()

    def process_adaptive_signals(self, data: pd.DataFrame, profile: Dict) -> pd.DataFrame:
        df = data.copy()
        params = self.regime_strategy.get_regime_adaptive_parameters(df, profile)

        df['current_regime']  = params['regime']
        df['regime_strategy'] = params['strategy_name']

        w = params['signal_weights']
        df['adaptive_composite_signal'] = (
            df.get('tech_signal', pd.Series(0, index=df.index)) * w['tech_weight'] +
            (df.get('headline_sentiment', pd.Series(0, index=df.index)) * profile.get('headline_sensitivity', 1.0) +
             df.get('social_sentiment',   pd.Series(0, index=df.index)) * profile.get('hype_sensitivity', 1.0))
            * w['sentiment_weight'] +
            df.get('options_signal', pd.Series(0, index=df.index)) * w['options_weight'] +
            df.get('macro_signal',   pd.Series(0, index=df.index)) * w['macro_weight']
        )

        bias = params['momentum_bias']
        if bias != 0 and 'momentum_signal' in df.columns:
            df['adaptive_composite_signal'] += df['momentum_signal'] * bias * 0.2

        if params['trend_following'] and 'trend_filter' in df.columns:
            df['adaptive_composite_signal'] *= df['trend_filter']

        df['adaptive_signal'] = np.where(
            df['adaptive_composite_signal'] > 0.5, 1,
            np.where(df['adaptive_composite_signal'] < -0.5, -1, 0)
        )
        df['adaptive_position_size'] = params['position_size']
        df['adaptive_stop_loss']     = params['stop_loss']

        logger.info(f"Regime: {params['regime']} → {params['strategy_name']}")
        return df


class AdvancedFeatureEngineer:
    """Advanced technical and market microstructure features."""

    def create_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df = self._add_momentum_features(df)
        df = self._add_volatility_features(df)
        df = self._add_volume_features(df)
        df = self._create_composite_signals(df)
        return df

    def _add_momentum_features(self, df):
        for p in [5, 10, 20, 50]:
            df[f'roc_{p}'] = (df['Close'] - df['Close'].shift(p)) / df['Close'].shift(p)
        df['acceleration_10']    = df['roc_10'] - df['roc_10'].shift(5)
        df['momentum_divergence'] = df['roc_20'] - df['roc_50']
        if 'RSI' in df.columns:
            df['rsi_slope']        = df['RSI'] - df['RSI'].shift(5)
            df['rsi_acceleration'] = df['rsi_slope'] - df['rsi_slope'].shift(5)
        return df

    def _add_volatility_features(self, df):
        ret = df['Close'].pct_change()
        for p in [5, 10, 20, 50]:
            df[f'realized_vol_{p}'] = ret.rolling(p).std() * np.sqrt(252)
        df['vol_of_vol']     = df['realized_vol_20'].rolling(20).std()
        df['volatility_skew'] = (
            (df['realized_vol_5'] - df['realized_vol_20']) /
            df['realized_vol_20'].replace(0, 1)
        )
        if 'High' in df.columns and 'Low' in df.columns:
            df['parkinson_vol'] = (
                np.log(df['High'] / df['Low']) ** 2 / (4 * np.log(2))
            ).rolling(20).mean() * np.sqrt(252)
        return df

    def _add_volume_features(self, df):
        if 'Volume' not in df.columns:
            return df
        df['volume_sma_20']      = df['Volume'].rolling(20).mean()
        df['volume_ratio']       = df['Volume'] / df['volume_sma_20']
        df['volume_trend_short'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()
        return df

    def _create_composite_signals(self, df):
        df['trend_signal'] = np.where(
            (df.get('roc_20', 0) > 0.02) & (df.get('momentum_divergence', 0) > 0), 1,
            np.where((df.get('roc_20', 0) < -0.02) & (df.get('momentum_divergence', 0) < 0), -1, 0)
        )
        df['momentum_signal'] = np.where(
            df.get('acceleration_10', 0) > 0.008, 1,
            np.where(df.get('acceleration_10', 0) < -0.008, -1, 0)
        )
        if 'vol_of_vol' in df.columns:
            q30 = df['vol_of_vol'].rolling(50).quantile(0.3)
            q70 = df['vol_of_vol'].rolling(50).quantile(0.7)
            df['volatility_signal'] = np.where(
                df['vol_of_vol'] < q30, 1,
                np.where(df['vol_of_vol'] > q70, -1, 0)
            )
        if 'volume_ratio' in df.columns:
            df['volume_signal'] = np.where(
                df['volume_ratio'] > 1.5, 1,
                np.where(df['volume_ratio'] < 0.7, -1, 0)
            )
        return df


# ---------------------------------------------------------------------------
# Global instances + convenience functions (backward-compatible API)
# ---------------------------------------------------------------------------

regime_detector    = MarketRegimeClassifier()
adaptive_processor = AdaptiveSignalProcessor()
feature_engineer   = AdvancedFeatureEngineer()


def apply_regime_adaptation(data: pd.DataFrame, profile: Dict) -> pd.DataFrame:
    df = data.copy()
    try:
        df = feature_engineer.create_advanced_features(df)

        # Attach per-row regime labels
        regime_series = regime_detector.predict_regime_series(df)
        df['current_regime'] = regime_series.reindex(df.index).ffill().fillna('ranging')

        df = adaptive_processor.process_adaptive_signals(df, profile)
        logger.info("Applied rules-based regime-aware signal processing")
    except Exception as e:
        logger.error(f"Error in regime adaptation: {e}")
        df['adaptive_signal']        = df.get('signal', 0)
        df['adaptive_position_size'] = 1.0
        df['adaptive_stop_loss']     = 0.05
        df['current_regime']         = 'ranging'
    return df


def train_regime_system(historical_data: pd.DataFrame) -> None:
    """No-op for rules-based system — kept for API compatibility."""
    logger.info("Rules-based regime system requires no training.")
