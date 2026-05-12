# Hype-Fear Alert System

Real-time hype/fear detector that fuses Grok-scored news sentiment with technical signals and fires Telegram alerts when composite scores cross asset-specific thresholds. Built around a walk-forward XGBoost backtester, rules-based regime classifier, and per-asset sensitivity profiles.

## Architecture

```
yfinance headlines ─┐
                    ├─► Grok (xAI) ─┐
yfinance prices ────┤                ├─► composite score ─► Telegram alert
                    └─► tech signal ─┘            ▲
                                                  │
                                  asset profile + regime weights
```

| File | Role |
|---|---|
| `alert_system.py` | Live loop. Scans watchlist every 5 min, fires Telegram alerts. |
| `grok_sentiment.py` | Headline fetch + Grok scoring. |
| `backtest_engine.py` | Walk-forward backtest, XGBoost ML, Optuna tuning. |
| `market_regime_detector.py` | Rules-based bull/bear/ranging/crisis classifier. |
| `signal_integration.py` | Fuses external signals (Google Trends, funding rates) by profile sensitivity. |
| `advanced_data_sources.py` | Optional NewsAPI + transformers sentiment. |
| `external_signals.py` | Google Trends + funding-rate collectors. |
| `trading_system_v5.py` | CLI: `--mode backtest | live | signal | optimize`. |
| `obsidian_logger.py` | Writes daily notes to `~/Brain/`. |
| `profiles.json` | Per-asset sensitivity tuning. |

## Quick start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in keys
python alert_system.py
```

## Environment variables

| Var | Required | Purpose |
|---|---|---|
| `XAI_API_KEY` | yes | Grok sentiment scoring |
| `TELEGRAM_BOT_TOKEN` | yes (for alerts) | Sent via @BotFather |
| `TELEGRAM_CHAT_ID` | yes (for alerts) | Get via @userinfobot |
| `POLYGON_API_KEY` | recommended | Real options flow in backtest |
| `NEWSAPI_KEY` | optional | Extra news source for `advanced_data_sources` |
| `STATE_DIR` | optional | Where to persist cooldown/history JSON (default: `./state`) |

## Modes

```bash
python trading_system_v5.py --mode signal                       # one-off scan
python trading_system_v5.py --mode backtest --start 2020-01-01  # full WF backtest
python trading_system_v5.py --mode optimize                     # threshold tuning
python alert_system.py                                          # live loop
python alert_system.py --once                                   # single scan (for cron deploys)
```

## Deployment

See `render.yaml` for Render Background Worker / Cron Job config. Both options share the same image.

## Testing

```bash
python test_external_signals.py
```
