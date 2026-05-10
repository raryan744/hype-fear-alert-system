"""
Obsidian ↔ Grok bridge.

Obsidian → Grok : reads vault context files and injects them into Grok's
                  system prompt so it understands Robert's specific assets,
                  sensitivities, and trading style.

Grok → Obsidian : appends every sentiment scan, alert, and signal to the
                  daily note at ~/Brain/Daily Notes/YYYY-MM-DD.md and keeps
                  a running signal log at ~/Brain/Claude Memory/signal-log.md
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Vault paths
# ---------------------------------------------------------------------------

VAULT = Path.home() / "Brain"
DAILY_NOTES = VAULT / "Daily Notes"
MEMORY      = VAULT / "Claude Memory"
PROJECTS    = VAULT / "Projects"

DAILY_NOTES.mkdir(parents=True, exist_ok=True)
MEMORY.mkdir(parents=True, exist_ok=True)


def _today_note() -> Path:
    return DAILY_NOTES / f"{datetime.now().strftime('%Y-%m-%d')}.md"


# ---------------------------------------------------------------------------
# Obsidian → Grok: context injection
# ---------------------------------------------------------------------------

def load_vault_context() -> str:
    """
    Read key vault files and return a compact context string to inject
    into Grok's system prompt. Keeps Grok grounded in Robert's specific
    assets, sensitivities, and style rather than generic market knowledge.
    """
    sections = []

    files = [
        MEMORY / "user-robert.md",
        MEMORY / "projects.md",
        PROJECTS / "hype-fear-alert-system.md",
    ]

    for path in files:
        if path.exists():
            content = path.read_text().strip()
            # Strip YAML frontmatter
            if content.startswith("---"):
                parts = content.split("---", 2)
                content = parts[2].strip() if len(parts) >= 3 else content
            sections.append(f"### {path.stem}\n{content}")

    if not sections:
        return ""

    return (
        "## Context from Robert's trading vault\n\n"
        + "\n\n".join(sections)
        + "\n\nUse this context when scoring sentiment — "
          "weight signals according to the asset's sensitivity profile above."
    )


# ---------------------------------------------------------------------------
# Grok → Obsidian: logging
# ---------------------------------------------------------------------------

def _append(path: Path, text: str) -> None:
    with open(path, "a") as f:
        f.write(text)


def ensure_daily_note() -> Path:
    """Create today's daily note with a header if it doesn't exist."""
    note = _today_note()
    if not note.exists():
        date_str = datetime.now().strftime("%Y-%m-%d")
        note.write_text(
            f"# {date_str}\n\n"
            "## Grok Sentiment Log\n\n"
            "| Time | Ticker | Score | Conf | Regime | Signal | Summary |\n"
            "|------|--------|-------|------|--------|--------|---------|\n"
        )
    return note


def log_sentiment_scan(results: Dict[str, Dict]) -> None:
    """
    Append a batch of sentiment results to today's daily note.
    results: { ticker: { composite, grok_score, tech_signal, confidence, regime, summary } }
    """
    note  = ensure_daily_note()
    time_ = datetime.now(timezone.utc).strftime("%H:%M")

    rows = []
    for ticker, r in results.items():
        composite = r.get("composite", 0)
        signal_label = "🟢 HYPE" if composite > 0.35 else ("🔴 FEAR" if composite < -0.35 else "⬜ HOLD")
        summary = r.get("summary", "")[:60].replace("|", "/")  # truncate for table
        rows.append(
            f"| {time_} | {ticker} | {composite:+.2f} | "
            f"{r.get('confidence', 0):.0%} | {r.get('regime', '?')} | "
            f"{signal_label} | {summary} |"
        )

    _append(note, "\n".join(rows) + "\n")


def log_alert(ticker: str, direction: str, scores: Dict, message: str) -> None:
    """Append a fired alert to today's daily note and the persistent signal log."""
    note     = ensure_daily_note()
    time_    = datetime.now(timezone.utc).strftime("%H:%M UTC")
    log_path = MEMORY / "signal-log.md"

    alert_block = (
        f"\n### 🚨 Alert — {ticker} {direction} @ {time_}\n"
        f"- Composite: `{scores.get('composite', 0):+.3f}`  "
        f"Confidence: `{scores.get('confidence', 0):.0%}`\n"
        f"- Regime: `{scores.get('regime', '?').upper()}`  "
        f"Tech: `{scores.get('tech_signal', 0):+.2f}`  "
        f"Grok: `{scores.get('grok_score', 0):+.2f}`\n"
        f"- _{scores.get('summary', '')}_\n"
    )
    _append(note, alert_block)

    # Persistent log for outcome tracking
    if not log_path.exists():
        log_path.write_text(
            "# Signal Log\n\n"
            "Tracks every alert fired. Update Outcome column after the fact.\n\n"
            "| Date | Time | Ticker | Direction | Composite | Confidence | Regime | Outcome |\n"
            "|------|------|--------|-----------|-----------|------------|--------|---------|\n"
        )

    date_ = datetime.now().strftime("%Y-%m-%d")
    log_row = (
        f"| {date_} | {time_} | {ticker} | {direction} | "
        f"{scores.get('composite', 0):+.3f} | {scores.get('confidence', 0):.0%} | "
        f"{scores.get('regime', '?')} | ⬜ pending |\n"
    )
    _append(log_path, log_row)


def log_session_summary(n_scans: int, n_alerts: int, top_movers: List[str]) -> None:
    """Append an end-of-session summary to today's daily note."""
    note  = _today_note()
    time_ = datetime.now(timezone.utc).strftime("%H:%M UTC")

    summary = (
        f"\n## Session Summary @ {time_}\n"
        f"- Scans run: {n_scans}\n"
        f"- Alerts fired: {n_alerts}\n"
        f"- Top movers: {', '.join(top_movers) if top_movers else 'none'}\n"
    )
    _append(note, summary)
