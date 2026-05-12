"""
Hype-Fear local dashboard.

Reads the SQLite DB that the Render worker pushes daily to data/hype_fear.db,
plus the per-ticker backtest results, and renders an at-a-glance view of
what's accumulating.

Usage:
    pip install streamlit
    streamlit run local_dashboard.py

Auto-refresh every 30s. Click "Pull latest from GitHub" to fetch the newest
daily snapshot before that.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd
import streamlit as st

REPO_DIR     = Path(__file__).parent
DATA_DB_PATH = REPO_DIR / "data" / "hype_fear.db"
LOCAL_DB_PATH = REPO_DIR / "state" / "hype_fear.db"
BACKTEST     = REPO_DIR / "backtest_results_per_ticker.json"
REFRESH_SECS = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_db_path() -> Tuple[Path, str]:
    """Prefer local state/ DB (if running on the worker host), else data/ from GitHub sync."""
    if LOCAL_DB_PATH.exists() and LOCAL_DB_PATH.stat().st_size > 0:
        return LOCAL_DB_PATH, "local (state/)"
    if DATA_DB_PATH.exists() and DATA_DB_PATH.stat().st_size > 0:
        return DATA_DB_PATH, "GitHub snapshot (data/)"
    return DATA_DB_PATH, "not found"


@st.cache_data(ttl=REFRESH_SECS)
def query(db_path: str, sql: str, params: tuple = ()) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


@st.cache_data(ttl=300)
def load_backtest() -> dict:
    if not BACKTEST.exists():
        return {}
    return json.loads(BACKTEST.read_text())


def fmt_ts(ts: str) -> str:
    if not ts:
        return ""
    try:
        return datetime.fromisoformat(ts.replace("Z", "")).strftime("%m-%d %H:%M")
    except Exception:
        return ts[:16]


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Hype-Fear Live",
    page_icon="📊",
    layout="wide",
)

st.title("Hype-Fear Alert System")

db_path, db_source = resolve_db_path()

# Controls row
c1, c2, c3, c4 = st.columns([2, 2, 2, 6])

if c1.button("🔄 Pull from GitHub", use_container_width=True):
    with st.spinner("git pull..."):
        result = subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=str(REPO_DIR), capture_output=True, text=True,
        )
        if result.returncode == 0:
            st.success(result.stdout.strip() or "Up to date")
            # Bust the cache so the next render sees the new file
            st.cache_data.clear()
        else:
            st.error(result.stderr.strip() or "git pull failed")
        time.sleep(0.8)
        st.rerun()

auto_refresh = c2.toggle("Auto-refresh", value=True, help=f"Every {REFRESH_SECS}s")

c3.metric("DB source", db_source)

if db_source == "not found":
    st.warning(
        "Database not found locally. Click **Pull from GitHub** to fetch the daily snapshot. "
        "The Render worker pushes a fresh snapshot at 03:00 UTC daily."
    )
    st.stop()

c4.caption(f"Reading from: `{db_path}` · refreshed {datetime.now().strftime('%H:%M:%S')}")


# ---------------------------------------------------------------------------
# Top counters
# ---------------------------------------------------------------------------

summary_sql = {
    "scan_runs":   "SELECT COUNT(*) AS c FROM scan_runs",
    "scans":       "SELECT COUNT(*) AS c FROM scans",
    "alerts":      "SELECT COUNT(*) AS c FROM alerts",
    "alerts_24h":  "SELECT COUNT(*) AS c FROM alerts WHERE fired_at > datetime('now','-1 day')",
    "sec_events":  "SELECT COUNT(*) AS c FROM sec_events",
    "sec_24h":     "SELECT COUNT(*) AS c FROM sec_events WHERE filed_at > datetime('now','-1 day')",
    "outcomes":    "SELECT COUNT(*) AS c FROM outcomes",
}

counters = {}
for key, sql in summary_sql.items():
    try:
        counters[key] = int(query(str(db_path), sql).iloc[0, 0])
    except Exception:
        counters[key] = None

win_rate_sql = "SELECT AVG(correct_5d) FROM outcomes WHERE correct_5d IS NOT NULL"
try:
    wr = query(str(db_path), win_rate_sql).iloc[0, 0]
    counters["win_rate_5d"] = float(wr) if wr is not None else None
except Exception:
    counters["win_rate_5d"] = None

st.subheader("Counters")
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Scan runs",      f"{counters.get('scan_runs') or 0:,}")
m2.metric("Scans (rows)",   f"{counters.get('scans') or 0:,}")
m3.metric("Alerts total",   f"{counters.get('alerts') or 0:,}")
m4.metric("Alerts 24h",     f"{counters.get('alerts_24h') or 0}")
m5.metric("SEC events",     f"{counters.get('sec_events') or 0:,}",
          delta=f"+{counters.get('sec_24h') or 0} (24h)" if counters.get('sec_24h') else None)
wr = counters.get("win_rate_5d")
m6.metric("5d win rate",
          f"{wr * 100:.1f}%" if wr is not None else "—",
          delta=f"{counters.get('outcomes') or 0} scored" if counters.get('outcomes') else "no data yet")


# ---------------------------------------------------------------------------
# Latest signals per ticker
# ---------------------------------------------------------------------------

st.subheader("Latest signals (one row per ticker)")
latest_sql = """
SELECT s.ticker, s.ts, s.composite, s.confidence, s.grok_score, s.tech_signal,
       s.regime, s.price, s.grok_summary
FROM scans s
INNER JOIN (
    SELECT ticker, MAX(ts) AS max_ts FROM scans GROUP BY ticker
) latest ON s.ticker = latest.ticker AND s.ts = latest.max_ts
ORDER BY ABS(s.composite) DESC
"""
try:
    latest = query(str(db_path), latest_sql)
    if not latest.empty:
        latest["ts"] = latest["ts"].apply(fmt_ts)
        latest["composite"] = latest["composite"].round(3)
        latest["confidence"] = (latest["confidence"] * 100).round(0).astype(int).astype(str) + "%"
        latest["grok_score"] = latest["grok_score"].round(2)
        latest["tech_signal"] = latest["tech_signal"].round(2)
        latest["price"] = latest["price"].round(2)

        def color_composite(v):
            try:
                v = float(v)
                if v > 0.3:  return "background-color: #133813"  # green
                if v > 0.0:  return "background-color: #1c2b1c"
                if v < -0.3: return "background-color: #4a1a1a"  # red
                if v < 0.0:  return "background-color: #2b1c1c"
            except Exception:
                pass
            return ""

        styled = latest.style.map(color_composite, subset=["composite"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info("No scans in DB yet.")
except Exception as e:
    st.error(f"latest-signals query failed: {e}")


# ---------------------------------------------------------------------------
# Recent alerts
# ---------------------------------------------------------------------------

st.subheader("Recent alerts (last 30)")
alerts_sql = """
SELECT a.fired_at, a.ticker, a.direction, a.composite, a.confidence,
       a.regime, a.price, a.expected_pct, a.option_symbol, a.option_strike,
       a.option_expiry, o.return_5d, o.correct_5d
FROM alerts a
LEFT JOIN outcomes o ON o.alert_id = a.alert_id
ORDER BY a.fired_at DESC
LIMIT 30
"""
try:
    alerts = query(str(db_path), alerts_sql)
    if not alerts.empty:
        alerts["fired_at"] = alerts["fired_at"].apply(fmt_ts)
        alerts["composite"] = alerts["composite"].round(3)
        alerts["confidence"] = (alerts["confidence"] * 100).round(0).astype(int).astype(str) + "%"
        alerts["price"] = alerts["price"].round(2)
        alerts["expected_pct"] = (alerts["expected_pct"] * 100).round(2).astype(str) + "%"
        alerts["return_5d"] = alerts["return_5d"].apply(
            lambda v: f"{v*100:+.2f}%" if pd.notna(v) else "—"
        )
        alerts["correct_5d"] = alerts["correct_5d"].apply(
            lambda v: "✓" if v == 1 else ("✗" if v == 0 else "—")
        )
        st.dataframe(alerts, use_container_width=True, hide_index=True)
    else:
        st.info("No alerts yet. Worker fires them when composite crosses ±0.45–±0.65 per asset.")
except Exception as e:
    st.error(f"alerts query failed: {e}")


# ---------------------------------------------------------------------------
# EDGAR feed
# ---------------------------------------------------------------------------

st.subheader("EDGAR SEC events (last 30)")
sec_sql = """
SELECT filed_at, ticker, form_type, transaction_code, insider_name, insider_role,
       dollar_value, cluster_size_5d, is_first_time, is_post_decline,
       strength_score, alerted, filing_url
FROM sec_events
ORDER BY filed_at DESC
LIMIT 30
"""
try:
    sec = query(str(db_path), sec_sql)
    if not sec.empty:
        sec["filed_at"] = sec["filed_at"].apply(fmt_ts)
        sec["dollar_value"] = sec["dollar_value"].apply(
            lambda v: f"${v:,.0f}" if pd.notna(v) and v else "—"
        )
        sec["strength_score"] = sec["strength_score"].round(2)
        sec["is_first_time"] = sec["is_first_time"].apply(lambda v: "yes" if v else "")
        sec["is_post_decline"] = sec["is_post_decline"].apply(lambda v: "yes" if v else "")
        sec["alerted"] = sec["alerted"].apply(lambda v: "📋" if v else "")
        # Show filing_url as a markdown link by truncating
        sec["filing_url"] = sec["filing_url"].apply(
            lambda u: "→ link" if pd.notna(u) and u else ""
        )
        st.dataframe(sec, use_container_width=True, hide_index=True)
    else:
        st.info(
            "No SEC events yet. Listener picks them up at most 60s after they hit EDGAR. "
            "Most of the time the feed will only contain filings from non-watchlist companies — "
            "the system silently skips those."
        )
except Exception as e:
    st.error(f"sec_events query failed: {e}")


# ---------------------------------------------------------------------------
# Backtest reference panel
# ---------------------------------------------------------------------------

with st.expander("📊 Backtest reference (4yr / 2022–2026)", expanded=False):
    bt = load_backtest()
    if bt:
        rows = []
        for ticker, r in bt.items():
            if "error" in r:
                continue
            rows.append({
                "ticker":   ticker,
                "ann_ret":  r.get("annualized_return"),
                "sharpe":   r.get("sharpe_ratio"),
                "win_rate": r.get("win_rate"),
                "max_dd":   r.get("max_drawdown"),
                "trades":   r.get("total_trades"),
                "regime":   r.get("dominant_regime"),
            })
        df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
        df["ann_ret"]  = (df["ann_ret"] * 100).round(2).astype(str) + "%"
        df["sharpe"]   = df["sharpe"].round(2)
        df["win_rate"] = (df["win_rate"] * 100).round(1).astype(str) + "%"
        df["max_dd"]   = (df["max_dd"] * 100).round(1).astype(str) + "%"
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("backtest_results_per_ticker.json not found.")


# ---------------------------------------------------------------------------
# Scan-activity sparkline
# ---------------------------------------------------------------------------

st.subheader("Scan & alert activity (last 7 days)")
activity_sql = """
SELECT date(ts) AS day,
       COUNT(*) AS scan_rows,
       (SELECT COUNT(*) FROM alerts WHERE date(fired_at) = date(scans.ts)) AS alerts
FROM scans
WHERE ts > datetime('now','-7 days')
GROUP BY date(ts)
ORDER BY day
"""
try:
    activity = query(str(db_path), activity_sql)
    if not activity.empty:
        activity = activity.set_index("day")
        col_a, col_b = st.columns(2)
        col_a.line_chart(activity[["scan_rows"]], height=240)
        col_b.line_chart(activity[["alerts"]], height=240)
    else:
        st.info("No activity in last 7 days.")
except Exception as e:
    st.error(f"activity query failed: {e}")


# ---------------------------------------------------------------------------
# Auto-refresh tail
# ---------------------------------------------------------------------------

if auto_refresh:
    time.sleep(REFRESH_SECS)
    st.rerun()
