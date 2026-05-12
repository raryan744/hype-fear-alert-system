"""
SEC EDGAR real-time filing listener.

Watches the EDGAR Atom feed for filings on the configured watchlist tickers
and fires INDEPENDENT Telegram alerts (separate channel from hype/fear) for
high-strength events. All filings — including low-strength ones — are
persisted to `sec_events` in the same SQLite DB so the data is available for
Phase D calibration.

Why polling (not WebSocket): SEC does not publish a WebSocket. The Atom feed
at `sec.gov/cgi-bin/browse-edgar?action=getcurrent` is the canonical
real-time interface used by every pro EDGAR client. Polling at 60s is well
below SEC's 10 req/sec ceiling and adds at most 60s of lag vs an imaginary
websocket.

Predictive constellation (per the literature):
  Form 4  : CEO/CFO open-market buys (code 'P', not 'A' awards or 'S' sales),
            cluster buys (≥3 insiders in 5d), first-time buyers, post-decline
            buys, dollar value ≥ $500k → strength_score
  13D     : always alert — activist intent, ~6-7% abnormal return on filing
  13G     : alert when stake is large (>10%)
  8-K     : only items 4.02 (restatement), 5.02 (officer departure) — most
            other items are noise. Item text matters; we surface the summary.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set
from xml.etree import ElementTree as ET

import requests

import signal_logger

logger = logging.getLogger(__name__)

# SEC requires a User-Agent identifying the requester (name + email).
# https://www.sec.gov/os/accessing-edgar-data
SEC_UA = os.getenv("SEC_USER_AGENT", "Robert Ryan hype-fear-alerts robertryan744@gmail.com")
SEC_HEADERS = {"User-Agent": SEC_UA, "Accept": "application/json"}

EDGAR_BASE = "https://www.sec.gov"
ATOM_RECENT = (
    EDGAR_BASE
    + "/cgi-bin/browse-edgar?action=getcurrent"
    + "&type={form}&output=atom&count=40"
)
COMPANY_TICKERS = EDGAR_BASE + "/files/company_tickers.json"

# Forms we watch (in priority order)
WATCH_FORMS = ["4", "13D", "13G", "8-K"]
HIGH_VALUE_8K_ITEMS = {"4.02", "5.02"}

# Poll the recent-filings feed every 60s. Each filing detail fetch is on demand.
POLL_INTERVAL_SECS = int(os.getenv("EDGAR_POLL_SECS", 60))

# Where to stash the "last seen accession" checkpoint so we don't re-process
# a backlog on restart.
STATE_DIR = Path(os.getenv("STATE_DIR", "./state"))
CHECKPOINT_FILE = STATE_DIR / "edgar_checkpoint.json"


# ---------------------------------------------------------------------------
# Ticker ↔ CIK mapping (cached for the process lifetime)
# ---------------------------------------------------------------------------

_cik_cache: Dict[str, str] = {}      # ticker → CIK string (zero-padded to 10)
_ticker_cache: Dict[str, str] = {}   # CIK → ticker
_loaded = False


def _load_cik_map(watchlist: List[str]) -> None:
    """Fetch SEC's full ticker→CIK map and cache the slice we care about."""
    global _loaded
    try:
        r = requests.get(COMPANY_TICKERS, headers=SEC_HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()  # {"0":{"cik_str":...,"ticker":"AAPL","title":...},...}
        want = {t.upper() for t in watchlist}
        for row in data.values():
            tic = row["ticker"].upper()
            if tic in want:
                cik = str(row["cik_str"]).zfill(10)
                _cik_cache[tic] = cik
                _ticker_cache[cik] = tic
        _loaded = True
        logger.info(f"EDGAR CIK map loaded for {len(_cik_cache)}/{len(want)} watchlist tickers")
    except Exception as e:
        logger.error(f"Failed to load CIK map: {e}")


# ---------------------------------------------------------------------------
# Atom feed parser
# ---------------------------------------------------------------------------

@dataclass
class FeedEntry:
    accession_no: str
    cik:          str
    form_type:    str
    filed_at:     str
    filing_url:   str
    summary:      str


_NS = {"a": "http://www.w3.org/2005/Atom"}

# Atom entry title looks like: "4 - Apple Inc (0000320193) (Issuer)"
# The trailing role is "Issuer", "Filer", or "Reporting". Form 4 generates one
# entry per role — we keep only Issuer/Filer rows so we can index by the
# company's CIK (not the insider's personal CIK).
_TITLE_RE = re.compile(r"^([A-Z0-9/-]+)\s*-\s*.*?\((\d+)\)\s*\(([A-Za-z]+)\)\s*$")
_ID_RE    = re.compile(r"accession-number=(\S+)")
_ACC_RE   = re.compile(r"AccNo[:\s]+([0-9-]+)", re.IGNORECASE)

ROLES_TO_KEEP = {"issuer", "filer"}


def _fetch_atom(form: str) -> List[FeedEntry]:
    url = ATOM_RECENT.format(form=form)
    try:
        r = requests.get(url, headers=SEC_HEADERS, timeout=20)
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception as e:
        logger.warning(f"Atom fetch failed for form {form}: {e}")
        return []

    entries: List[FeedEntry] = []
    for entry in root.findall("a:entry", _NS):
        title   = (entry.findtext("a:title",   default="", namespaces=_NS) or "").strip()
        updated = (entry.findtext("a:updated", default="", namespaces=_NS) or "").strip()
        summary = (entry.findtext("a:summary", default="", namespaces=_NS) or "").strip()
        id_text = (entry.findtext("a:id",      default="", namespaces=_NS) or "").strip()
        link_el = entry.find("a:link", _NS)
        link    = link_el.get("href") if link_el is not None else ""

        m = _TITLE_RE.match(title)
        if not m:
            continue
        observed_form = m.group(1)
        cik           = m.group(2).zfill(10)
        role          = m.group(3).lower()

        # Skip the per-insider duplicate row — keep only the issuer/filer row
        # so cik resolves to a ticker via _ticker_cache.
        if role not in ROLES_TO_KEEP:
            continue

        accession = ""
        id_match = _ID_RE.search(id_text)
        if id_match:
            accession = id_match.group(1)
        else:
            acc_match = _ACC_RE.search(summary) or _ACC_RE.search(link)
            if acc_match:
                accession = acc_match.group(1)

        entries.append(FeedEntry(
            accession_no=accession,
            cik=cik,
            form_type=observed_form,
            filed_at=updated,
            filing_url=link,
            summary=summary,
        ))
    return entries


# ---------------------------------------------------------------------------
# Form 4 detail parser
# ---------------------------------------------------------------------------

def _form4_detail_url(accession_no: str, cik: str) -> Optional[str]:
    """Build the URL of the Form 4 XML document inside a filing index."""
    if not accession_no:
        return None
    acc_no_clean = accession_no.replace("-", "")
    return (f"{EDGAR_BASE}/cgi-bin/browse-edgar?action=getcompany"
            f"&CIK={cik}&type=4&dateb=&owner=include&count=40")


def _fetch_form4_xml(filing_url: str) -> Optional[str]:
    """
    Given the filing index URL (…/{acc}-index.htm), find the XBRL data
    document via the directory's index.json and fetch its XML.
    """
    try:
        # Strip the '-index.htm' suffix to get the directory base
        dir_url = filing_url.rsplit("/", 1)[0]
        idx = requests.get(dir_url + "/index.json", headers=SEC_HEADERS, timeout=20)
        idx.raise_for_status()
        items = idx.json().get("directory", {}).get("item", [])

        # The XBRL data doc has a wf-form4_*.xml or similar name and is NOT
        # one of the obvious noise files (FilingSummary.xml, stylesheets, etc.).
        # We pick the smallest .xml that contains the word 'ownership' or
        # that has the canonical 'wf-form' prefix.
        candidates = []
        for it in items:
            name = it.get("name", "")
            if not name.endswith(".xml"):
                continue
            if "FilingSummary" in name or "primary_doc.xml" == name:
                continue
            candidates.append(name)

        # Prefer canonical Form 4 XML names
        preferred = [n for n in candidates if "form4" in n.lower() or "ownership" in n.lower()]
        chosen    = preferred[0] if preferred else (candidates[0] if candidates else None)
        if not chosen:
            # Fallback: try primary_doc.xml — newer filings use that as the XBRL
            chosen = "primary_doc.xml"

        xml_url = f"{dir_url}/{chosen}"
        r2 = requests.get(xml_url, headers=SEC_HEADERS, timeout=20)
        r2.raise_for_status()
        return r2.text
    except Exception as e:
        logger.warning(f"Could not fetch Form 4 XML from {filing_url}: {e}")
        return None


def _parse_form4(xml_text: str) -> Dict:
    """
    Extract: reporting person, role, transaction code, shares, price.

    Multiple transactions per filing are summed if same code (e.g. several
    purchase lots on the same day).
    """
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return {}

    def first_text(path: str) -> str:
        el = root.find(path)
        if el is None:
            return ""
        # Many Form 4 fields wrap value in <value>...</value>
        v = el.find("value")
        return (v.text or "").strip() if (v is not None and v.text) else (el.text or "").strip()

    name = first_text(".//reportingOwnerId/rptOwnerName")
    is_officer  = first_text(".//reportingOwnerRelationship/isOfficer") == "1"
    is_director = first_text(".//reportingOwnerRelationship/isDirector") == "1"
    is_10pct    = first_text(".//reportingOwnerRelationship/isTenPercentOwner") == "1"
    officer_title = first_text(".//reportingOwnerRelationship/officerTitle")
    role_parts = []
    if is_officer:  role_parts.append(officer_title or "Officer")
    if is_director: role_parts.append("Director")
    if is_10pct:    role_parts.append("10%-Owner")
    role = ", ".join(role_parts) or "Insider"

    # Aggregate across all non-derivative transactions in this filing
    total_shares = 0.0
    total_value  = 0.0
    codes: Set[str] = set()
    for tx in root.findall(".//nonDerivativeTransaction"):
        code_el  = tx.find(".//transactionCode")
        shares_el= tx.find(".//transactionShares/value")
        price_el = tx.find(".//transactionPricePerShare/value")
        code   = (code_el.text or "").strip() if (code_el is not None and code_el.text) else ""
        try:
            shares = float(shares_el.text) if (shares_el is not None and shares_el.text) else 0.0
            price  = float(price_el.text)  if (price_el  is not None and price_el.text)  else 0.0
        except Exception:
            shares, price = 0.0, 0.0
        if not code:
            continue
        codes.add(code)
        total_shares += shares
        total_value  += shares * price

    # Primary code: prefer P (purchase) > S (sale) > A (award) > others
    primary = next((c for c in ("P", "S", "A") if c in codes),
                   next(iter(codes), ""))
    avg_price = (total_value / total_shares) if total_shares else 0.0

    return {
        "insider_name": name,
        "insider_role": role,
        "transaction_code": primary,
        "shares": total_shares,
        "price": avg_price,
        "dollar_value": total_value,
        "is_c_suite": any(t in (officer_title or "").upper() for t in
                          ("CEO", "CFO", "COO", "PRESIDENT", "CHIEF", "CHAIRMAN")),
    }


# ---------------------------------------------------------------------------
# 8-K item parser
# ---------------------------------------------------------------------------

_ITEM_RE = re.compile(r"Item\s+(\d+\.\d+)", re.IGNORECASE)


def _parse_8k_items(summary: str) -> List[str]:
    return list({m.upper() for m in _ITEM_RE.findall(summary)})


# ---------------------------------------------------------------------------
# Strength scoring
# ---------------------------------------------------------------------------

def _post_decline(ticker: str) -> bool:
    """Has this ticker fallen ≥20% from its YTD high? Best-effort, non-fatal."""
    try:
        import yfinance as yf
        import pandas as pd
        df = yf.download(ticker, period="ytd", interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            return False
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        ytd_high = float(df["Close"].max())
        latest   = float(df["Close"].iloc[-1])
        return ytd_high > 0 and (latest / ytd_high - 1) <= -0.20
    except Exception:
        return False


def _form4_strength(parsed: Dict, ticker: str) -> Dict:
    """Score a Form 4 filing on the literature-derived dimensions."""
    if not parsed:
        return {"strength_score": 0.0, "cluster_size_5d": 0,
                "is_first_time": False, "is_post_decline": False}
    code = parsed.get("transaction_code")
    if code != "P":
        # Sales and awards get logged but never alerted by us.
        return {"strength_score": 0.0,
                "cluster_size_5d": signal_logger.cluster_size_5d(ticker),
                "is_first_time": False, "is_post_decline": False}

    cluster = signal_logger.cluster_size_5d(ticker)
    first_time = signal_logger.is_first_time_buyer(ticker, parsed.get("insider_name", ""))
    post_decline = _post_decline(ticker)
    dollar = parsed.get("dollar_value", 0.0)
    is_c_suite = parsed.get("is_c_suite", False)

    score = 0.0
    if is_c_suite:                 score += 0.35   # CEO/CFO buy is the strongest single feature
    if cluster >= 3:               score += 0.30   # cluster buy
    elif cluster >= 2:             score += 0.15
    if first_time:                 score += 0.15
    if dollar >= 500_000:          score += 0.10
    if dollar >= 1_000_000:        score += 0.05   # extra bump
    if post_decline:               score += 0.10
    score = min(score, 1.0)

    return {
        "strength_score":  score,
        "cluster_size_5d": cluster,
        "is_first_time":   bool(first_time),
        "is_post_decline": bool(post_decline),
    }


def _filing_strength(form_type: str, parsed: Dict, ticker: str,
                     item_codes: List[str]) -> Dict:
    # Normalise amendments (e.g. '4/A' → '4', '8-K/A' → '8-K')
    base = form_type.split("/")[0].upper()
    if base == "4":
        return _form4_strength(parsed, ticker)
    if base == "13D":
        return {"strength_score": 0.9}   # activist filings — always alert
    if base == "13G":
        return {"strength_score": 0.55}  # passive 5% — usually alert
    if base == "8-K":
        if any(i in HIGH_VALUE_8K_ITEMS for i in item_codes):
            return {"strength_score": 0.75}
        return {"strength_score": 0.10}
    return {"strength_score": 0.0}


# ---------------------------------------------------------------------------
# Telegram formatting
# ---------------------------------------------------------------------------

def _format_alert(event: Dict) -> str:
    form = event["form_type"]
    ticker = event["ticker"]
    url    = event.get("filing_url", "")

    if form == "4":
        name   = event.get("insider_name") or "(unknown)"
        role   = event.get("insider_role") or "Insider"
        code   = event.get("transaction_code") or "?"
        action = {"P": "BOUGHT", "S": "SOLD", "A": "AWARDED"}.get(code, code)
        shares = event.get("shares") or 0
        price  = event.get("price") or 0
        dollar = event.get("dollar_value") or 0
        cluster = event.get("cluster_size_5d") or 0
        ftime  = "yes" if event.get("is_first_time") else "no"
        decline = "yes" if event.get("is_post_decline") else "no"
        msg = (
            f"📋 *{ticker}  INSIDER {action}*\n"
            f"{name}  _{role}_\n"
            f"Shares: `{int(shares):,}` @ `${price:,.2f}`\n"
            f"Value:  `${dollar:,.0f}`\n"
            f"Cluster (5d): `{cluster}`  ·  First-time: `{ftime}`  ·  Post-decline: `{decline}`\n"
        )
    elif form == "13D":
        msg = (
            f"📋 *{ticker}  ACTIVIST 13D*\n"
            f"New >5% activist position filed.\n"
        )
    elif form == "13G":
        msg = (
            f"📋 *{ticker}  PASSIVE 13G*\n"
            f"New >5% passive position filed.\n"
        )
    elif form == "8-K":
        items = ", ".join(event.get("item_codes_list", []))
        msg = (
            f"📋 *{ticker}  8-K  (items: {items})*\n"
            f"_{(event.get('raw_summary') or '')[:240]}_\n"
        )
    else:
        msg = f"📋 *{ticker}  {form}*\n"

    if url:
        msg += f"[Filing →]({url})\n"
    msg += f"`{event.get('filed_at', '')[:19]} UTC`"
    return msg


# ---------------------------------------------------------------------------
# Checkpoint (persist across worker restarts)
# ---------------------------------------------------------------------------

def _load_checkpoint() -> Set[str]:
    if not CHECKPOINT_FILE.exists():
        return set()
    try:
        return set(json.loads(CHECKPOINT_FILE.read_text()))
    except Exception:
        return set()


def _save_checkpoint(seen: Set[str], max_size: int = 5000) -> None:
    try:
        # Trim to bounded size to avoid unbounded JSON growth
        items = list(seen)[-max_size:]
        CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_FILE.write_text(json.dumps(items))
    except Exception as e:
        logger.warning(f"Could not save EDGAR checkpoint: {e}")


# ---------------------------------------------------------------------------
# Main listener
# ---------------------------------------------------------------------------

# Telegram threshold for alerting. Below this, the event is silently logged.
ALERT_STRENGTH_MIN = float(os.getenv("EDGAR_ALERT_STRENGTH", 0.45))


def process_entry(entry: FeedEntry, watchlist_ciks: Set[str]) -> Optional[Dict]:
    """Parse and persist one feed entry. Returns the alert-ready dict or None."""
    if entry.cik not in watchlist_ciks:
        return None
    if signal_logger.already_seen(entry.accession_no):
        return None

    ticker = _ticker_cache.get(entry.cik) or ""
    base_form = entry.form_type.split("/")[0].upper()

    parsed: Dict = {}
    item_codes: List[str] = []

    if base_form == "4":
        xml = _fetch_form4_xml(entry.filing_url)
        if xml:
            parsed = _parse_form4(xml)
    elif base_form == "8-K":
        item_codes = _parse_8k_items(entry.summary)

    strength = _filing_strength(entry.form_type, parsed, ticker, item_codes)

    event = {
        "filed_at":         entry.filed_at,
        "seen_at":          datetime.now(timezone.utc).isoformat(),
        "ticker":           ticker,
        "cik":              entry.cik,
        "form_type":        entry.form_type,
        "accession_no":     entry.accession_no,
        "filing_url":       entry.filing_url,
        "item_codes":       ",".join(item_codes) if item_codes else None,
        "item_codes_list":  item_codes,
        "insider_name":     parsed.get("insider_name"),
        "insider_role":     parsed.get("insider_role"),
        "transaction_code": parsed.get("transaction_code"),
        "shares":           parsed.get("shares"),
        "price":            parsed.get("price"),
        "dollar_value":     parsed.get("dollar_value"),
        "pct_of_holdings":  None,   # would need 13F cross-reference; skip for v1
        "cluster_size_5d":  strength.get("cluster_size_5d"),
        "is_first_time":    int(bool(strength.get("is_first_time"))),
        "is_post_decline":  int(bool(strength.get("is_post_decline"))),
        "strength_score":   strength.get("strength_score", 0.0),
        "alerted":          0,
        "raw_summary":      entry.summary[:1000] if entry.summary else None,
    }

    event_id = signal_logger.log_sec_event(event)
    if event_id is None:
        return None  # duplicate

    if event["strength_score"] >= ALERT_STRENGTH_MIN:
        event["event_id"] = event_id
        return event

    return None


def poll_once(watchlist: List[str], seen_acc: Set[str]) -> List[Dict]:
    """One pass across all watch forms. Returns the alert-worthy events."""
    if not _loaded:
        _load_cik_map(watchlist)

    watchlist_ciks = set(_cik_cache.values())
    if not watchlist_ciks:
        return []

    alertable: List[Dict] = []
    for form in WATCH_FORMS:
        for entry in _fetch_atom(form):
            if entry.accession_no in seen_acc:
                continue
            seen_acc.add(entry.accession_no)
            try:
                event = process_entry(entry, watchlist_ciks)
                if event:
                    alertable.append(event)
            except Exception as e:
                logger.warning(f"process_entry failed for {entry.accession_no}: {e}")
        # Be polite — small delay between forms keeps us well under 10 req/s
        time.sleep(0.25)

    return alertable


def listener_thread(watchlist: List[str], send_telegram_func, stop_event: threading.Event) -> None:
    """Blocking loop — run inside a daemon thread from alert_system.run_loop."""
    logger.info(f"EDGAR listener starting (poll {POLL_INTERVAL_SECS}s, "
                f"alert threshold {ALERT_STRENGTH_MIN}, "
                f"watchlist {len(watchlist)} tickers)")
    seen_acc = _load_checkpoint()

    while not stop_event.is_set():
        try:
            events = poll_once(watchlist, seen_acc)
            for ev in events:
                try:
                    msg = _format_alert(ev)
                    ok = send_telegram_func(msg)
                    if ok and ev.get("event_id"):
                        signal_logger.mark_alerted(ev["event_id"])
                except Exception as e:
                    logger.warning(f"EDGAR alert send failed: {e}")
            _save_checkpoint(seen_acc)
        except Exception as e:
            logger.error(f"EDGAR poll cycle error: {e}", exc_info=True)

        # Sleep in small chunks so SIGTERM exits fast
        for _ in range(POLL_INTERVAL_SECS):
            if stop_event.is_set():
                break
            time.sleep(1)

    logger.info("EDGAR listener stopped.")
