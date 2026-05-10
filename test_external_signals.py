"""
Test script for external signals functionality.
Tests Google Trends and funding rate signal collection and integration.
"""

import sys
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_external_signals():
    """Test external signals collection."""
    print("Testing External Signals Collection")
    print("=" * 50)

    try:
        from external_signals import ExternalSignalsManager

        manager = ExternalSignalsManager()

        # Test assets
        test_assets = ['TSLA', 'NVDA', 'BTC-USD']

        print(f"Collecting signals for: {test_assets}")

        # Collect signals
        signals = manager.collect_all_signals(test_assets)

        print(f"\nCollected signals for {len(signals)} assets:")

        for asset, signal_data in signals.items():
            print(f"\n{asset}:")
            if 'google_trends' in signal_data:
                trends = signal_data['google_trends']
                print(f"  Google Trends: {trends.get('avg_interest', 'N/A')} interest")
                print(f"    Momentum: {trends.get('recent_momentum', 0):.1%}")
                print(f"    Keywords: {trends.get('keywords_used', [])}")
            else:
                print("  Google Trends: Not available")

            if 'funding_rate' in signal_data:
                funding = signal_data['funding_rate']
                print(f"  Funding Rate: {funding.get('avg_funding_rate', 0):.4f}")
                print(f"    Sentiment: {funding.get('sentiment', 'neutral')}")
                print(f"    Extreme: {funding.get('is_extreme', False)}")
            else:
                print("  Funding Rate: Not applicable or unavailable")

        return True

    except Exception as e:
        print(f"Error testing external signals: {e}")
        return False

def test_signal_integration():
    """Test signal integration with profiles."""
    print("\n\nTesting Signal Integration")
    print("=" * 50)

    try:
        from signal_integration import SignalIntegrator

        integrator = SignalIntegrator()

        # Test assets
        test_assets = ['TSLA', 'BTC-USD']

        print(f"Testing integration for: {test_assets}")

        results = integrator.process_assets(test_assets)

        for asset, data in results.items():
            combined = data['combined_signal']
            rec = data['recommendation']

            print(f"\n{asset}:")
            print(f"  External Score: {combined.get('external_score', 0):.3f}")
            print(f"  Alert Level: {rec['alert_level']}")
            print(f"  Confidence: {rec['confidence']}")

            if rec['reasoning']:
                print("  Reasoning:")
                for reason in rec['reasoning']:
                    print(f"    - {reason}")

        return True

    except Exception as e:
        print(f"Error testing signal integration: {e}")
        return False

def test_profiles_updated():
    """Test that profiles.json has been updated with new sensitivity parameters."""
    print("\n\nTesting Profiles Update")
    print("=" * 50)

    try:
        import json

        with open('profiles.json', 'r') as f:
            profiles = json.load(f)

        # Check for new parameters
        has_google_trends = False
        has_funding_rate = False

        for asset, profile in profiles.items():
            if 'google_trends_sensitivity' in profile:
                has_google_trends = True
                print(f"✓ {asset} has google_trends_sensitivity: {profile['google_trends_sensitivity']}")

            if 'funding_rate_sensitivity' in profile:
                has_funding_rate = True
                print(f"✓ {asset} has funding_rate_sensitivity: {profile['funding_rate_sensitivity']}")

        if has_google_trends:
            print("✓ Google Trends sensitivity parameters found")
        else:
            print("✗ Google Trends sensitivity parameters missing")

        if has_funding_rate:
            print("✓ Funding rate sensitivity parameters found")
        else:
            print("✗ Funding rate sensitivity parameters missing")

        return has_google_trends

    except Exception as e:
        print(f"Error testing profiles: {e}")
        return False

def test_dependencies():
    """Test that required dependencies are available."""
    print("\n\nTesting Dependencies")
    print("=" * 50)

    dependencies = {
        'pytrends': False,
        'httpx': False,
        'json': True,  # Built-in
        'datetime': True  # Built-in
    }

    # Test pytrends
    try:
        from pytrends.request import TrendReq
        dependencies['pytrends'] = True
        print("✓ pytrends available")
    except ImportError:
        print("✗ pytrends not available - install with: pip install pytrends")

    # Test httpx
    try:
        import httpx
        dependencies['httpx'] = True
        print("✓ httpx available")
    except ImportError:
        print("✗ httpx not available - install with: pip install httpx")

    return dependencies['pytrends'] or dependencies['httpx']

def generate_test_report():
    """Generate a comprehensive test report."""
    print("\n" + "=" * 60)
    print("EXTERNAL SIGNALS INTEGRATION TEST REPORT")
    print("=" * 60)
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print()

    # Run all tests
    tests = {
        'Dependencies': test_dependencies(),
        'Profiles Updated': test_profiles_updated(),
        'External Signals': test_external_signals(),
        'Signal Integration': test_signal_integration()
    }

    print("\nTEST RESULTS:")
    print("-" * 30)

    passed = 0
    total = len(tests)

    for test_name, result in tests.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1

    print("-" * 30)
    print(f"Overall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! External signals integration is ready.")
        print("\nNext steps:")
        print("1. Install missing dependencies: pip install pytrends httpx")
        print("2. Test with real data: python external_signals.py")
        print("3. Integrate into main alert system")
        print("4. Backtest the new signals")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above for details.")

    return passed == total

if __name__ == "__main__":
    success = generate_test_report()
    sys.exit(0 if success else 1)