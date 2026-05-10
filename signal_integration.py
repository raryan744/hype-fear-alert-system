"""
Integration module for combining external signals (Google Trends, funding rates)
with the existing hype-fear alert system.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

try:
    from external_signals import ExternalSignalsManager
    EXTERNAL_SIGNALS_AVAILABLE = True
except ImportError:
    EXTERNAL_SIGNALS_AVAILABLE = False
    logging.warning("external_signals module not available")

logger = logging.getLogger(__name__)


class SignalIntegrator:
    """Integrates external signals into the alert system."""

    def __init__(self, profiles_path: str = 'profiles.json'):
        self.profiles_path = profiles_path
        self.profiles = self._load_profiles()
        self.external_manager = None

        if EXTERNAL_SIGNALS_AVAILABLE:
            self.external_manager = ExternalSignalsManager()
        else:
            logger.warning("External signals not available - running with limited functionality")

    def _load_profiles(self) -> Dict:
        """Load asset profiles with sensitivity parameters."""
        try:
            with open(self.profiles_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load profiles: {e}")
            return {}

    def calculate_external_signal_score(self, asset: str, external_data: Dict) -> float:
        """
        Calculate weighted signal score from external data sources.

        Args:
            asset: Asset symbol
            external_data: External signal data for the asset

        Returns:
            Weighted signal score (normalized -1 to 1)
        """
        if asset not in self.profiles:
            logger.warning(f"No profile found for asset: {asset}")
            return 0.0

        profile = self.profiles[asset]
        total_score = 0.0
        total_weight = 0.0

        # Google Trends signal
        if 'google_trends' in external_data:
            trends_data = external_data['google_trends']
            sensitivity = profile.get('google_trends_sensitivity', 1.0)

            # Normalize trends interest (0-100 scale) to -1 to 1
            avg_interest = trends_data.get('avg_interest', 50)
            normalized_trends = (avg_interest - 50) / 50  # Center at 0

            # Apply momentum multiplier (recent changes are more significant)
            momentum = trends_data.get('recent_momentum', 0)
            momentum_multiplier = 1 + abs(momentum) * 0.5

            trends_score = normalized_trends * momentum_multiplier
            total_score += trends_score * sensitivity
            total_weight += sensitivity

        # Funding rate signal (crypto only)
        if 'funding_rate' in external_data:
            funding_data = external_data['funding_rate']
            sensitivity = profile.get('funding_rate_sensitivity', 1.0)

            avg_rate = funding_data.get('avg_funding_rate', 0)
            is_extreme = funding_data.get('is_extreme', False)

            # Normalize funding rate (typically -0.01 to 0.01 range)
            # Extreme rates (>1%) can be contrarian signals
            if is_extreme:
                # Invert extreme signals (very high funding = potential reversal)
                normalized_funding = -avg_rate * 100  # Convert to percentage and invert
            else:
                normalized_funding = avg_rate * 100  # Convert to percentage

            # Clamp to reasonable range
            normalized_funding = max(-1.0, min(1.0, normalized_funding))

            total_score += normalized_funding * sensitivity
            total_weight += sensitivity

        # Return weighted average
        if total_weight > 0:
            return total_score / total_weight
        return 0.0

    def get_combined_signal(self, asset: str, existing_signals: Optional[Dict] = None) -> Dict:
        """
        Get combined signal including external data sources.

        Args:
            asset: Asset symbol
            existing_signals: Existing signal data from the main system

        Returns:
            Combined signal dictionary
        """
        combined = {
            'asset': asset,
            'timestamp': datetime.now().isoformat(),
            'external_signals': {},
            'combined_score': 0.0
        }

        # Get external signals
        if self.external_manager:
            try:
                external_data = self.external_manager.collect_all_signals([asset])
                if asset in external_data:
                    combined['external_signals'] = external_data[asset]

                    # Calculate external signal score
                    external_score = self.calculate_external_signal_score(asset, external_data[asset])
                    combined['external_score'] = external_score

            except Exception as e:
                logger.error(f"Failed to collect external signals for {asset}: {e}")

        # Combine with existing signals if provided
        if existing_signals:
            # This would integrate with your existing signal calculation logic
            # For now, we'll just include both
            combined['existing_signals'] = existing_signals

            # Simple combination - you may want more sophisticated logic
            existing_score = existing_signals.get('total_score', 0)
            external_score = combined.get('external_score', 0)

            # Weighted combination (adjust weights as needed)
            combined['combined_score'] = (existing_score * 0.7) + (external_score * 0.3)

        return combined

    def get_alert_recommendation(self, combined_signal: Dict) -> Dict:
        """
        Generate alert recommendation based on combined signals.

        Args:
            combined_signal: Combined signal data

        Returns:
            Alert recommendation with level and reasoning
        """
        asset = combined_signal['asset']
        combined_score = combined_signal.get('combined_score', 0)
        external_score = combined_signal.get('external_score', 0)

        # Define alert thresholds (adjust based on your system's calibration)
        hype_threshold = 0.6
        fear_threshold = -0.6

        recommendation = {
            'asset': asset,
            'alert_level': 'neutral',
            'confidence': 'low',
            'reasoning': [],
            'external_contribution': external_score
        }

        # Analyze combined score
        if combined_score > hype_threshold:
            recommendation['alert_level'] = 'hype_alert'
            recommendation['reasoning'].append(f"Combined signal score {combined_score:.2f} exceeds hype threshold")
        elif combined_score < fear_threshold:
            recommendation['alert_level'] = 'fear_alert'
            recommendation['reasoning'].append(f"Combined signal score {combined_score:.2f} below fear threshold")

        # Analyze external signals specifically
        if abs(external_score) > 0.4:
            recommendation['confidence'] = 'high'
            if external_score > 0.4:
                recommendation['reasoning'].append(f"Strong external signals indicate hype ({external_score:.2f})")
            else:
                recommendation['reasoning'].append(f"Strong external signals indicate fear ({external_score:.2f})")

        # Check for extreme funding rates (contrarian signals)
        funding_data = combined_signal.get('external_signals', {}).get('funding_rate', {})
        if funding_data.get('is_extreme', False):
            rate = funding_data.get('avg_funding_rate', 0)
            if rate > 0.01:  # Very high funding rate
                recommendation['reasoning'].append("Extreme positive funding rate may signal potential reversal")
            elif rate < -0.01:  # Very negative funding rate
                recommendation['reasoning'].append("Extreme negative funding rate may signal capitulation")

        return recommendation

    def process_assets(self, assets: List[str], existing_signals: Optional[Dict] = None) -> Dict[str, Dict]:
        """
        Process multiple assets and return combined signals and recommendations.

        Args:
            assets: List of asset symbols
            existing_signals: Dict of existing signals keyed by asset

        Returns:
            Dict with combined signals and recommendations for each asset
        """
        results = {}

        for asset in assets:
            existing = existing_signals.get(asset) if existing_signals else None
            combined = self.get_combined_signal(asset, existing)
            recommendation = self.get_alert_recommendation(combined)

            results[asset] = {
                'combined_signal': combined,
                'recommendation': recommendation
            }

        return results


# Example integration functions for existing system
def integrate_external_signals(existing_system_function):
    """
    Decorator to integrate external signals into existing system functions.

    Example usage:
    @integrate_external_signals
    def calculate_signals(assets):
        # Your existing logic
        return signals
    """
    def wrapper(*args, **kwargs):
        # Call original function
        result = existing_system_function(*args, **kwargs)

        # Integrate external signals
        integrator = SignalIntegrator()

        # Assuming result is a dict with asset keys
        if isinstance(result, dict):
            for asset in result.keys():
                if asset in integrator.profiles:
                    combined = integrator.get_combined_signal(asset, result[asset])
                    result[asset]['external_signals'] = combined['external_signals']
                    result[asset]['external_score'] = combined.get('external_score', 0)

        return result

    return wrapper


# Convenience functions
def get_external_signals_report(assets: List[str]) -> str:
    """Generate a human-readable report of external signals."""
    if not EXTERNAL_SIGNALS_AVAILABLE:
        return "External signals module not available"

    integrator = SignalIntegrator()
    results = integrator.process_assets(assets)

    report_lines = ["External Signals Report", "=" * 50]

    for asset, data in results.items():
        combined = data['combined_signal']
        rec = data['recommendation']

        report_lines.append(f"\n{asset}:")
        report_lines.append(f"  Alert Level: {rec['alert_level'].upper()}")
        report_lines.append(f"  Confidence: {rec['confidence']}")
        report_lines.append(f"  External Score: {combined.get('external_score', 0):.3f}")

        if rec['reasoning']:
            report_lines.append("  Reasoning:")
            for reason in rec['reasoning']:
                report_lines.append(f"    - {reason}")

        # Show signal details
        external_signals = combined.get('external_signals', {})
        if 'google_trends' in external_signals:
            trends = external_signals['google_trends']
            report_lines.append(f"  Google Trends: {trends.get('avg_interest', 0):.1f} interest, {trends.get('recent_momentum', 0):.1%} momentum")

        if 'funding_rate' in external_signals:
            funding = external_signals['funding_rate']
            report_lines.append(f"  Funding Rate: {funding.get('avg_funding_rate', 0):.4f} ({funding.get('sentiment', 'neutral')})")

    return "\n".join(report_lines)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    assets = ['TSLA', 'BTC-USD', 'ETH-USD']

    print("Testing external signals integration...")
    print(get_external_signals_report(assets))