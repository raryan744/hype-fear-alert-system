"""
SQLite signal logger for the Hype-Fear Alert System.

Schema
------
scans       — one row per (scan, ticker). Every signal we ever computed.
alerts      — one row per fired alert, with enrichment snapshot at signal time.
outcomes    — retroactive scoring: realized 1d/5d/10d return + win/loss flag.
scan_runs   — one row per top-level scan (timestamp, ticker count, alert count).

This is the dataset Phase C (calibration model) will train on.

DB lives at STATE_DIR/hype_fear.db (so it's on the Render persistent disk in
prod). It is intentionally placed alongside cooldown_state.json + alert_history.json
so a single disk mount covers all persistent state.
"""

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Resolve the same STATE_DIR that alert_system uses
DB_PATH = Path(os.getenv("STATE_DIR", "./state")) / "hype_fear.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()


SCHEMA = """
CREATE TABLE IF NOT EXISTS scan_runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    n_tickers     INTEGER,
    n_alerts      INTEGER,
    duration_secs REAL
);

CREATE TABLE IF NOT EXISTS scans (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER REFERENCES scan_runs(run_id),
    ts            TEXT NOT NULL,
    ticker        TEXT NOT NULL,
    composite     REAL,
    confidence    REAL,
    grok_score    REAL,
    grok_conf     REAL,
    tech_signal   REAL,
    regime        TEXT,
    price         REAL,
    grok_summary  TEXT
);

CREATE INDEX IF NOT EXISTS idx_scans_ts     ON scans(ts);
CREATE INDEX IF NOT EXISTS idx_scans_ticker ON scans(ticker);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    fired_at      TEXT NOT NULL,
    ticker        TEXT NOT NULL,
    direction     TEXT NOT NULL,            -- 'hype' | 'fear'
    composite     REAL,
    confidence    REAL,
    grok_score    REAL,
    tech_signal   REAL,
    regime        TEXT,
    price         REAL,
    expected_pct  REAL,
    expected_dollars REAL,
    horizon_days  INTEGER,
    option_symbol TEXT,
    option_strike REAL,
    option_expiry TEXT,
    option_type   TEXT,
    enrichment_json TEXT,                   -- full enrichment dict for forensics
    message       TEXT
);

CREATE INDEX IF NOT EXISTS idx_alerts_fired  ON alerts(fired_at);
CREATE INDEX IF NOT EXISTS idx_alerts_ticker ON alerts(ticker);

CREATE TABLE IF NOT EXISTS outcomes (
    alert_id      INTEGER PRIMARY KEY REFERENCES alerts(alert_id),
    scored_at     TEXT NOT NULL,
    price_at_signal REAL,
    price_1d      REAL,
    price_5d      REAL,
    price_10d     REAL,
    return_1d     REAL,
    return_5d     REAL,
    return_10d    REAL,
    correct_5d    INTEGER,                  -- 1 if return aligns with signal direction
    notes         TEXT
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db() -> None:
    with _lock, _connect() as conn:
        conn.executescript(SCHEMA)


# Initialise schema at import so the file exists immediately
try:
    init_db()
except Exception as e:
    logger.warning(f"signal_logger init failed: {e}")


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def start_run(n_tickers: int) -> int:
    """Begin a new scan run, return its run_id."""
    with _lock, _connect() as conn:
        cur = conn.execute(
            "INSERT INTO scan_runs (started_at, n_tickers) VALUES (?, ?)",
            (datetime.utcnow().isoformat(), n_tickers),
        )
        return cur.lastrowid


def finish_run(run_id: int, n_alerts: int, duration_secs: float) -> None:
    with _lock, _connect() as conn:
        conn.execute(
            "UPDATE scan_runs SET n_alerts=?, duration_secs=? WHERE run_id=?",
            (n_alerts, duration_secs, run_id),
        )


def log_scan_row(run_id: int, ticker: str, scores: Dict, price: Optional[float] = None) -> None:
    """Insert one row per ticker scanned."""
    with _lock, _connect() as conn:
        conn.execute(
            """
            INSERT INTO scans (run_id, ts, ticker, composite, confidence,
                               grok_score, grok_conf, tech_signal, regime,
                               price, grok_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                scores.get("timestamp") or datetime.utcnow().isoformat(),
                ticker,
                scores.get("composite"),
                scores.get("confidence"),
                scores.get("grok_score"),
                scores.get("grok_conf"),
                scores.get("tech_signal"),
                scores.get("regime"),
                price,
                scores.get("summary"),
            ),
        )


def log_alert_row(ticker: str, direction: str, scores: Dict, message: str) -> int:
    """Insert one row per fired alert with full enrichment snapshot."""
    enrichment = scores.get("enrichment") or {}
    opt        = (enrichment.get("option") or {})
    with _lock, _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO alerts (
                fired_at, ticker, direction, composite, confidence,
                grok_score, tech_signal, regime, price,
                expected_pct, expected_dollars, horizon_days,
                option_symbol, option_strike, option_expiry, option_type,
                enrichment_json, message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(),
                ticker,
                direction,
                scores.get("composite"),
                scores.get("confidence"),
                scores.get("grok_score"),
                scores.get("tech_signal"),
                scores.get("regime"),
                enrichment.get("current_price"),
                enrichment.get("expected_pct"),
                enrichment.get("expected_dollars"),
                enrichment.get("horizon_days"),
                opt.get("contract_symbol"),
                opt.get("strike"),
                opt.get("expiration"),
                opt.get("type"),
                json.dumps(enrichment, default=str),
                message,
            ),
        )
        return cur.lastrowid


# ---------------------------------------------------------------------------
# Retroactive outcome scorer — run on a schedule (daily) to fill `outcomes`
# ---------------------------------------------------------------------------

def score_pending_outcomes(max_days_back: int = 30) -> int:
    """
    Walk recent alerts that have no outcome row yet and fill in realised
    1d/5d/10d returns. Safe to call repeatedly — only scores rows where
    enough time has elapsed.

    Returns: number of outcomes newly scored.
    """
    import yfinance as yf
    import pandas as pd

    scored = 0
    cutoff = (datetime.utcnow().timestamp() - max_days_back * 86400)

    with _lock, _connect() as conn:
        rows = conn.execute(
            """
            SELECT a.alert_id, a.ticker, a.fired_at, a.price, a.direction
            FROM alerts a
            LEFT JOIN outcomes o ON o.alert_id = a.alert_id
            WHERE o.alert_id IS NULL
            ORDER BY a.fired_at DESC
            """
        ).fetchall()

    for alert_id, ticker, fired_at, signal_price, direction in rows:
        try:
            fired_dt = datetime.fromisoformat(fired_at)
        except Exception:
            continue
        if fired_dt.timestamp() < cutoff:
            continue

        # Need at least 1 trading day past the signal before we can score
        # anything. For 10d return we need 10 trading days.
        days_elapsed = (datetime.utcnow() - fired_dt).days
        if days_elapsed < 1:
            continue

        try:
            df = yf.download(ticker, start=fired_dt.date().isoformat(),
                              progress=False, auto_adjust=True)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            closes = df["Close"].dropna()
            if closes.empty:
                continue

            p_signal = float(signal_price or closes.iloc[0])
            p1  = float(closes.iloc[1])  if len(closes) >  1 else None
            p5  = float(closes.iloc[5])  if len(closes) >  5 else None
            p10 = float(closes.iloc[10]) if len(closes) > 10 else None

            r1  = (p1  / p_signal - 1) if p1  else None
            r5  = (p5  / p_signal - 1) if p5  else None
            r10 = (p10 / p_signal - 1) if p10 else None

            correct_5d = None
            if r5 is not None:
                correct_5d = 1 if ((r5 > 0 and direction == "hype")
                                    or (r5 < 0 and direction == "fear")) else 0

            with _lock, _connect() as conn:
                conn.execute(
                    """
                    INSERT INTO outcomes (alert_id, scored_at, price_at_signal,
                                          price_1d, price_5d, price_10d,
                                          return_1d, return_5d, return_10d,
                                          correct_5d)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(alert_id) DO UPDATE SET
                        scored_at  = excluded.scored_at,
                        price_1d   = excluded.price_1d,
                        price_5d   = excluded.price_5d,
                        price_10d  = excluded.price_10d,
                        return_1d  = excluded.return_1d,
                        return_5d  = excluded.return_5d,
                        return_10d = excluded.return_10d,
                        correct_5d = excluded.correct_5d
                    """,
                    (alert_id, datetime.utcnow().isoformat(), p_signal,
                     p1, p5, p10, r1, r5, r10, correct_5d),
                )
            scored += 1
        except Exception as e:
            logger.warning(f"score_outcomes failed for alert {alert_id} ({ticker}): {e}")

    return scored


# ---------------------------------------------------------------------------
# Summary helpers (for dashboard, github_push, ad-hoc queries)
# ---------------------------------------------------------------------------

def summary() -> Dict:
    with _lock, _connect() as conn:
        def c(q): return conn.execute(q).fetchone()[0]
        return {
            "db_path":       str(DB_PATH),
            "scan_runs":     c("SELECT COUNT(*) FROM scan_runs"),
            "scans":         c("SELECT COUNT(*) FROM scans"),
            "alerts":        c("SELECT COUNT(*) FROM alerts"),
            "outcomes":      c("SELECT COUNT(*) FROM outcomes"),
            "win_rate_5d":   c(
                "SELECT ROUND(AVG(correct_5d), 3) FROM outcomes "
                "WHERE correct_5d IS NOT NULL"
            ),
            "alerts_last_24h": c(
                "SELECT COUNT(*) FROM alerts "
                "WHERE fired_at > datetime('now','-1 day')"
            ),
        }
