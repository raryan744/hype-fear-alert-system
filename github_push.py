"""
github_push.py — Daily push of state/hype_fear.db and a JSON summary to the
hype-fear-alert-system repo (data/ subdir).

Mirrors the kalshi/Markets github_push.py pattern.

Reads GITHUB_TOKEN from env or .env. Designed to run inside the Render worker
(daily at 03:00 UTC via an in-process timer) AND to be runnable manually.

Pushes:
  - data/hype_fear.db                          — full SQLite database
  - data/summary.json                          — counts + win-rate snapshot
  - data/daily_reports/YYYY-MM-DD.md           — human-readable daily report

Idempotent: only commits when there are actual changes.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO          = "raryan744/hype-fear-alert-system"
BRANCH        = "main"
STATE_DIR     = Path(os.getenv("STATE_DIR", "./state"))
DB_PATH       = STATE_DIR / "hype_fear.db"
COOLDOWN_FILE = STATE_DIR / "cooldown_state.json"
HISTORY_FILE  = STATE_DIR / "alert_history.json"


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def _summary_dict() -> dict:
    import signal_logger
    return {
        **signal_logger.summary(),
        "generated_at": datetime.utcnow().isoformat(),
    }


def _build_daily_report(summary: dict) -> str:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    lines = [
        f"# Hype-Fear Daily Report — {today}",
        "",
        "## Counts",
        f"- Scan runs total: **{summary.get('scan_runs', 0):,}**",
        f"- Scan rows total: **{summary.get('scans', 0):,}**",
        f"- Alerts total:    **{summary.get('alerts', 0):,}**",
        f"- Alerts last 24h: **{summary.get('alerts_last_24h', 0)}**",
        f"- Outcomes scored: **{summary.get('outcomes', 0):,}**",
        "",
        "## Performance",
        f"- 5-day win rate (all-time): **{(summary.get('win_rate_5d') or 0) * 100:.1f}%**",
        "",
        f"_Auto-generated at {summary.get('generated_at')}_",
    ]
    return "\n".join(lines)


def push() -> int:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        logger.error("GITHUB_TOKEN not set — aborting push")
        return 1

    if not DB_PATH.exists():
        logger.warning(f"No DB at {DB_PATH} — nothing to push")
        return 0

    today_iso  = datetime.utcnow().strftime("%Y-%m-%d")
    summary    = _summary_dict()
    daily_md   = _build_daily_report(summary)

    # Clone repo into temp dir, copy files in, commit + push
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        clone_url = f"https://x-access-token:{token}@github.com/{REPO}.git"
        logger.info(f"Cloning {REPO}...")
        _run(["git", "clone", "--depth", "1", "--branch", BRANCH, clone_url, str(tmp)], cwd=Path("/"))

        data_dir   = tmp / "data"
        reports_dir = data_dir / "daily_reports"
        data_dir.mkdir(exist_ok=True)
        reports_dir.mkdir(exist_ok=True)

        # Copy DB + summary + daily report
        (data_dir / "hype_fear.db").write_bytes(DB_PATH.read_bytes())
        (data_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        (reports_dir / f"{today_iso}.md").write_text(daily_md)

        # Configure committer (no global git config)
        _run(["git", "config", "user.email", "hype-fear-bot@noreply"], cwd=tmp)
        _run(["git", "config", "user.name",  "hype-fear-bot"],          cwd=tmp)

        _run(["git", "add", "data/"], cwd=tmp)
        status = _run(["git", "status", "--porcelain"], cwd=tmp).stdout.strip()
        if not status:
            logger.info("No changes to commit.")
            return 0

        commit_msg = f"data: daily snapshot {today_iso} ({summary.get('alerts_last_24h', 0)} alerts/24h)"
        _run(["git", "commit", "-m", commit_msg], cwd=tmp)
        _run(["git", "push", "origin", BRANCH], cwd=tmp)
        logger.info(f"Pushed: {commit_msg}")
        return 0


if __name__ == "__main__":
    sys.exit(push())
