"""
Market Regime Detection and Adaptive Strategy System

PHASE 2A: Market Regime Awareness
- Bull/Bear/Ranging/Crisis market classification
- Regime-specific signal weighting
- Adaptive parameter optimization
- Volatility-based position sizing

This module enables the system to adapt strategies based on current market conditions.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MarketRegimeClassifier:
    """Classifies market regimes using unsupervised learning."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = None
        self.regime_labels = {
            0: 'bull',
            1: 'bear',
            2: 'ranging',
            3: 'crisis'
        }
        self.fitted = False

    def fit(self, data: pd.DataFrame, n_regimes: int = 4) -> None:
        """Fit the regime classifier on historical data."""
        try:
            # Extract regime features
            features = self._extract_regime_features(data)

            # Scale features
            scaled_features = self.scaler.fit_transform(features)

            # Fit K-means clustering
            self.kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
            self.kmeans.fit(scaled_features)

            self.fitted = True
            logger.info(f"Regime classifier fitted with {n_regimes} regimes")

        except Exception as e:
            logger.error(f"Error fitting regime classifier: {e}")

    def predict_regime(self, data: pd.DataFrame) -> str:
        """Predict current market regime."""
        if not self.fitted:
            return 'unknown'

        try:
            # Extract features for current data point
            features = self._extract_regime_features(data)
            if features.empty:
                return 'unknown'

            # Scale and predict
            scaled_features = self.scaler.transform(features)
            regime_idx = self.kmeans.predict(scaled_features)[0]

            return self.regime_labels.get(regime_idx, 'unknown')

        except Exception as e:
            logger.error(f"Error predicting regime: {e}")
            return 'unknown'

    def _extract_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for regime classification."""
        df = data.copy()

        # Trend features
        df['trend_20'] = (df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20)
        df['trend_50'] = (df['Close'] - df['Close'].shift(50)) / df['Close'].shift(50)
        df['trend_100'] = (df['Close'] - df['Close'].shift(100)) / df['Close'].shift(100)

        # Volatility features
        df['volatility_20'] = df['Close'].pct_change().rolling(20).std()
        df['volatility_50'] = df['Close'].pct_change().rolling(50).std()

        # Momentum features
        df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1

        # Volume features
        if 'Volume' in df.columns:
            df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            df['volume_trend'] = df['Volume'].rolling(10).mean() / df['Volume'].rolling(50).mean()

        # Technical indicators
        if 'RSI' in df.columns:
            df['rsi_trend'] = df['RSI'] - df['RSI'].shift(10)

        if 'MACD' in df.columns:
            df['macd_trend'] = df['MACD'] - df['MACD'].shift(10)

        # Select features for clustering
        feature_cols = [
            'trend_20', 'trend_50', 'trend_100',
            'volatility_20', 'volatility_50',
            'momentum_10', 'momentum_20'
        ]

        # Add optional features if available
        optional_features = ['volume_ratio', 'volume_trend', 'rsi_trend', 'macd_trend']
        for feat in optional_features:
            if feat in df.columns:
                feature_cols.append(feat)

        # Return features (drop NaN values)
        features = df[feature_cols].dropna()

        return features


class RegimeAdaptiveStrategy:
    """Adapts trading strategy based on market regime."""

    def __init__(self):
        self.regime_classifier = MarketRegimeClassifier()
        self.regime_strategies = self._define_regime_strategies()

    def _define_regime_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Define optimal strategies for each market regime."""

        return {
            'bull': {
                'name': 'Bull Market Strategy',
                'description': 'Aggressive momentum following',
                'signal_weights': {
                    'tech_weight': 0.5,      # High technical weight
                    'sentiment_weight': 0.2, # Moderate sentiment
                    'options_weight': 0.1,   # Low options weight
                    'macro_weight': 0.2      # Moderate macro
                },
                'position_size_multiplier': 1.5,  # Larger positions
                'stop_loss_multiplier': 1.2,      # Wider stops
                'trend_following': True,
                'momentum_bias': 1.0
            },

            'bear': {
                'name': 'Bear Market Strategy',
                'description': 'Defensive short bias with hedging',
                'signal_weights': {
                    'tech_weight': 0.3,      # Moderate technical
                    'sentiment_weight': 0.4, # High sentiment weight
                    'options_weight': 0.2,   # Higher options weight
                    'macro_weight': 0.1      # Low macro
                },
                'position_size_multiplier': 0.7,  # Smaller positions
                'stop_loss_multiplier': 0.8,      # Tighter stops
                'trend_following': False,
                'momentum_bias': -0.5
            },

            'ranging': {
                'name': 'Range Trading Strategy',
                'description': 'Mean reversion with tight stops',
                'signal_weights': {
                    'tech_weight': 0.4,      # Balanced technical
                    'sentiment_weight': 0.1, # Low sentiment
                    'options_weight': 0.3,   # High options weight
                    'macro_weight': 0.2      # Moderate macro
                },
                'position_size_multiplier': 0.8,  # Moderate positions
                'stop_loss_multiplier': 0.6,      # Very tight stops
                'trend_following': False,
                'momentum_bias': 0.0
            },

            'crisis': {
                'name': 'Crisis Management Strategy',
                'description': 'Ultra-defensive with heavy risk management',
                'signal_weights': {
                    'tech_weight': 0.2,      # Low technical
                    'sentiment_weight': 0.5, # Very high sentiment
                    'options_weight': 0.2,   # Moderate options
                    'macro_weight': 0.1      # Low macro
                },
                'position_size_multiplier': 0.3,  # Very small positions
                'stop_loss_multiplier': 0.5,      # Ultra-tight stops
                'trend_following': False,
                'momentum_bias': -1.0
            },

            'unknown': {
                'name': 'Conservative Default Strategy',
                'description': 'Balanced approach when regime unclear',
                'signal_weights': {
                    'tech_weight': 0.4,
                    'sentiment_weight': 0.3,
                    'options_weight': 0.2,
                    'macro_weight': 0.1
                },
                'position_size_multiplier': 1.0,
                'stop_loss_multiplier': 1.0,
                'trend_following': False,
                'momentum_bias': 0.0
            }
        }

    def train_regime_classifier(self, historical_data: pd.DataFrame) -> None:
        """Train the regime classifier on historical data."""
        logger.info("Training market regime classifier...")
        self.regime_classifier.fit(historical_data)
        logger.info("Regime classifier training completed")

    def get_regime_adaptive_parameters(self, current_data: pd.DataFrame,
                                     base_profile: Dict) -> Dict[str, Any]:
        """Get regime-adapted trading parameters."""

        # Detect current regime
        current_regime = self.regime_classifier.predict_regime(current_data)

        # Get regime-specific strategy
        regime_strategy = self.regime_strategies.get(current_regime,
                                                    self.regime_strategies['unknown'])

        # Adapt signal weights based on regime
        adapted_weights = regime_strategy['signal_weights'].copy()

        # Apply profile sensitivities
        adapted_weights = self._apply_profile_sensitivities(adapted_weights, base_profile)

        # Calculate position sizing
        base_position_size = base_profile.get('base_position_size', 1.0)
        position_size = base_position_size * regime_strategy['position_size_multiplier']

        # Calculate stop loss
        base_stop_loss = base_profile.get('base_stop_loss', 0.05)
        stop_loss = base_stop_loss * regime_strategy['stop_loss_multiplier']

        # Momentum bias adjustment
        momentum_bias = regime_strategy['momentum_bias']

        return {
            'regime': current_regime,
            'strategy_name': regime_strategy['name'],
            'signal_weights': adapted_weights,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'momentum_bias': momentum_bias,
            'trend_following': regime_strategy['trend_following']
        }

    def _apply_profile_sensitivities(self, weights: Dict[str, float],
                                   profile: Dict) -> Dict[str, float]:
        """Apply ticker-specific sensitivities to weights."""
        adapted = weights.copy()

        # Apply sentiment sensitivity
        sentiment_sensitivity = profile.get('sentiment_sensitivity', 1.0)
        adapted['sentiment_weight'] *= sentiment_sensitivity

        # Apply technical sensitivity
        technical_sensitivity = profile.get('technical_sensitivity', 1.0)
        adapted['tech_weight'] *= technical_sensitivity

        # Apply options sensitivity
        options_sensitivity = profile.get('options_sensitivity', 1.0)
        adapted['options_weight'] *= options_sensitivity

        # Normalize weights to sum to 1
        total_weight = sum(adapted.values())
        if total_weight > 0:
            adapted = {k: v / total_weight for k, v in adapted.items()}

        return adapted


class AdaptiveSignalProcessor:
    """Processes signals with regime-aware adaptations."""

    def __init__(self):
        self.regime_strategy = RegimeAdaptiveStrategy()

    def process_adaptive_signals(self, data: pd.DataFrame, profile: Dict) -> pd.DataFrame:
        """Process signals with regime adaptation."""
        df = data.copy()

        # Get regime-adaptive parameters
        adaptive_params = self.regime_strategy.get_regime_adaptive_parameters(df, profile)

        # Store regime information
        df['current_regime'] = adaptive_params['regime']
        df['regime_strategy'] = adaptive_params['strategy_name']

        # Apply adaptive signal weights
        signal_weights = adaptive_params['signal_weights']

        # Calculate composite signal with adaptive weights
        df['adaptive_composite_signal'] = (
            df.get('tech_signal', 0) * signal_weights['tech_weight'] +
            (df['headline_sentiment'] * profile.get('headline_sensitivity', 1.0) +
             df['social_sentiment'] * profile.get('hype_sensitivity', 1.0)) * signal_weights['sentiment_weight'] +
            df.get('options_signal', 0) * signal_weights['options_weight'] +
            df.get('macro_signal', 0) * signal_weights['macro_weight']
        )

        # Apply momentum bias
        momentum_bias = adaptive_params['momentum_bias']
        if momentum_bias != 0:
            momentum_signal = df.get('momentum_signal', 0)
            df['adaptive_composite_signal'] += momentum_signal * momentum_bias * 0.2

        # Apply trend following filter if enabled
        if adaptive_params['trend_following']:
            trend_filter = df.get('trend_filter', 1)
            df['adaptive_composite_signal'] *= trend_filter

        # Generate final signals
        df['adaptive_signal'] = np.where(
            df['adaptive_composite_signal'] > 0.6, 1,
            np.where(df['adaptive_composite_signal'] < -0.6, -1, 0)
        )

        # Apply adaptive position sizing
        df['adaptive_position_size'] = adaptive_params['position_size']

        # Apply adaptive stop loss
        df['adaptive_stop_loss'] = adaptive_params['stop_loss']

        logger.info(f"Applied {adaptive_params['strategy_name']} for {adaptive_params['regime']} regime")

        return df


class AdvancedFeatureEngineer:
    """Creates advanced technical and market microstructure features."""

    def __init__(self):
        pass

    def create_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create sophisticated features for better signal generation."""
        df = data.copy()

        # Higher-order technical features
        df = self._add_momentum_features(df)
        df = self._add_volatility_features(df)
        df = self._add_volume_features(df)
        df = self._add_intermarket_features(df)
        df = self._add_microstructure_features(df)

        # Composite signals
        df = self._create_composite_signals(df)

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced momentum features."""
        # Rate of change features
        for period in [5, 10, 20, 50]:
            df[f'roc_{period}'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)

        # Acceleration (second derivative)
        df['acceleration_10'] = df['roc_10'] - df['roc_10'].shift(5)

        # Momentum divergence
        df['momentum_divergence'] = df['roc_20'] - df['roc_50']

        # Relative strength index variations
        if 'RSI' in df.columns:
            df['rsi_slope'] = df['RSI'] - df['RSI'].shift(5)
            df['rsi_acceleration'] = df['rsi_slope'] - df['rsi_slope'].shift(5)

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced volatility features."""
        returns = df['Close'].pct_change()

        # Realized volatility at different horizons
        for period in [5, 10, 20, 50]:
            df[f'realized_vol_{period}'] = returns.rolling(period).std() * np.sqrt(252)

        # Volatility of volatility
        df['vol_of_vol'] = df['realized_vol_20'].rolling(20).std()

        # Volatility skew
        df['volatility_skew'] = (
            df['realized_vol_5'] - df['realized_vol_20']
        ) / df['realized_vol_20'].replace(0, 1)

        # Parkinson volatility (if high/low available)
        if 'High' in df.columns and 'Low' in df.columns:
            df['parkinson_vol'] = (
                np.log(df['High']/df['Low'])**2 / (4 * np.log(2))
            ).rolling(20).mean() * np.sqrt(252)

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced volume features."""
        if 'Volume' in df.columns:
            # Volume rate of change
            df['volume_roc_5'] = (df['Volume'] - df['Volume'].shift(5)) / df['Volume'].shift(5)

            # Volume moving averages
            df['volume_sma_20'] = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma_20']

            # Volume trends
            df['volume_trend_short'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()
            df['volume_trend_long'] = df['Volume'].rolling(20).mean() / df['Volume'].rolling(50).mean()

            # Volume-price trend
            price_trend = df['Close'].rolling(20).mean() / df['Close'].rolling(50).mean()
            df['volume_price_trend'] = df['volume_trend_short'] * price_trend

        return df

    def _add_intermarket_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add inter-market relationship features."""
        # These would be enhanced with actual inter-market data
        # For now, create proxy features based on available data

        # Trend strength relative to volatility
        if 'trend_20' in df.columns and 'volatility_20' in df.columns:
            df['trend_strength'] = df['trend_20'] / df['volatility_20'].replace(0, 0.01)

        # Momentum vs volatility
        if 'momentum_20' in df.columns:
            df['momentum_vol_ratio'] = df['momentum_20'] / df['volatility_20'].replace(0, 0.01)

        return df

    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        # Bid-ask spread proxy (using high-low range)
        if 'High' in df.columns and 'Low' in df.columns and 'Close' in df.columns:
            df['spread_proxy'] = (df['High'] - df['Low']) / df['Close']

        # Price impact proxy
        if 'Volume' in df.columns:
            df['price_impact'] = df['Close'].pct_change() / df['Volume'].replace(0, 1)

        # Order flow proxy (simplified)
        df['order_flow'] = df['Close'].pct_change() * df.get('Volume', 1)

        return df

    def _create_composite_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite signals from advanced features."""
        # Trend signal
        df['trend_signal'] = np.where(
            (df.get('roc_20', 0) > 0.02) & (df.get('momentum_20', 0) > 0), 1,
            np.where((df.get('roc_20', 0) < -0.02) & (df.get('momentum_20', 0) < 0), -1, 0)
        )

        # Momentum signal
        df['momentum_signal'] = np.where(
            df.get('acceleration_10', 0) > 0.01, 1,
            np.where(df.get('acceleration_10', 0) < -0.01, -1, 0)
        )

        # Volatility signal
        df['volatility_signal'] = np.where(
            df.get('vol_of_vol', 0) < df.get('vol_of_vol', 0).rolling(50).quantile(0.3), 1,
            np.where(df.get('vol_of_vol', 0) > df.get('vol_of_vol', 0).rolling(50).quantile(0.7), -1, 0)
        )

        # Volume signal
        df['volume_signal'] = np.where(
            df.get('volume_ratio', 1) > 1.5, 1,
            np.where(df.get('volume_ratio', 1) < 0.7, -1, 0)
        )

        return df


# Global instances
regime_detector = MarketRegimeClassifier()
adaptive_processor = AdaptiveSignalProcessor()
feature_engineer = AdvancedFeatureEngineer()


def apply_regime_adaptation(data: pd.DataFrame, profile: Dict) -> pd.DataFrame:
    """Apply full regime-aware processing to data."""
    df = data.copy()

    try:
        # Add advanced features
        df = feature_engineer.create_advanced_features(df)

        # Apply regime-adaptive processing
        df = adaptive_processor.process_adaptive_signals(df, profile)

        logger.info("Applied regime-aware signal processing")

    except Exception as e:
        logger.error(f"Error in regime adaptation: {e}")
        # Return original data if processing fails
        df['adaptive_signal'] = df.get('signal', 0)
        df['adaptive_position_size'] = 1.0
        df['adaptive_stop_loss'] = 0.05
        df['current_regime'] = 'unknown'

    return df


def train_regime_system(historical_data: pd.DataFrame) -> None:
    """Train the regime detection system."""
    logger.info("Training regime detection system...")
    regime_detector.fit(historical_data)
    adaptive_processor.regime_strategy.train_regime_classifier(historical_data)
    logger.info("Regime system training completed")


if __name__ == "__main__":
    # Test the regime detection system
    print("Testing market regime detection system...")

    # Create sample data
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    np.random.seed(42)

    # Simulate different market regimes
    n_points = len(dates)
    prices = [100]

    for i in range(1, n_points):
        # Add regime-dependent returns
        if i < n_points // 4:  # Bull market
            ret = np.random.normal(0.001, 0.02)
        elif i < n_points // 2:  # Bear market
            ret = np.random.normal(-0.001, 0.025)
        elif i < 3 * n_points // 4:  # Ranging
            ret = np.random.normal(0.000, 0.015)
        else:  # Crisis
            ret = np.random.normal(-0.002, 0.04)

        prices.append(prices[-1] * (1 + ret))

    test_data = pd.DataFrame({
        'Close': prices,
        'Volume': np.random.exponential(1000000, n_points)
    }, index=dates)

    # Add basic technical indicators
    test_data['SMA_20'] = test_data['Close'].rolling(20).mean()
    test_data['RSI'] = 50 + np.random.normal(0, 10, n_points)  # Mock RSI
    test_data['volatility_20'] = test_data['Close'].pct_change().rolling(20).std()

    print("Training regime classifier...")
    train_regime_system(test_data)

    # Test regime prediction
    recent_data = test_data.tail(50)
    regime = regime_detector.predict_regime(recent_data)
    print(f"Current regime prediction: {regime}")

    print("Regime detection system test completed!")