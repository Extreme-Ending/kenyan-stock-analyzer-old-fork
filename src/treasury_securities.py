"""
Kenya Government Treasury securities — Treasury Bonds and Treasury Bills —
sourced from the Central Bank of Kenya (CBK), the authoritative issuer.

WHY THIS EXISTS
    Kenyan investors weighing NSE shares against "risk-free" government paper
    need the current government yields in the same place. This module pulls,
    straight from CBK:
      • Treasury BILLS — 91/182/364-day latest weighted-average rates
        (parsed from the CBK T-bills rates table).
      • Treasury BONDS currently on offer, and recent past offers, parsed
        directly from CBK's official prospectus PDFs (issue number, coupon,
        tenor, maturity, tax treatment, sale window, auction date, and the
        clean/dirty price for re-opened bonds).

ACCURACY (money is involved — nothing is invented)
    Every figure comes from CBK's own HTML tables and prospectus PDFs. If a
    field cannot be parsed, it is left as None and the UI shows a link to the
    official prospectus rather than a guessed value. If CBK is unreachable the
    whole module fails safe and returns empty structures — the rest of the
    dashboard is unaffected.

    This module produces DATA and neutral, factual education only. It does not
    constitute financial advice.
"""

import io
import os
import re
import json
from datetime import datetime, date

from logger import get_logger

logger = get_logger(__name__)

CBK = "https://www.centralbank.go.ke"
TBILLS_URL = f"{CBK}/bills-bonds/treasury-bills/"
BONDS_URL = f"{CBK}/bills-bonds/treasury-bonds/"
SOURCE = "Central Bank of Kenya"

_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}

_MONTHS = {m.lower(): i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
     "Nov", "Dec"], start=1)}
_MONTHS.update({m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], start=1)})
_MONTHS["july"] = 7  # guard common CBK spellings


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _today():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Africa/Nairobi")).date()
    except Exception:
        return datetime.now().date()


def _get(url, timeout=(3, 8)):
    """Fail-safe GET via the shared tight-timeout helper. Returns response or None."""
    try:
        from utils import http_get
        return http_get(url, headers=_UA, timeout=timeout)
    except Exception as e:
        logger.warning(f"Treasury fetch error {url}: {e}")
        return None


def _cells(row_html):
    import html as H
    cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, re.S | re.I)
    out = [H.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in cells]
    return [c for c in out if c != ""]


def _parse_any_date(s):
    """Tolerantly parse the many date spellings CBK uses. Returns date or None."""
    if not s:
        return None
    s = s.strip()
    # 08-Oct-2035 / 17-08-2026 / 30-July 2026 / 22-Jul-2026
    m = re.search(r"(\d{1,2})[-/\s]+([A-Za-z]{3,9}|\d{1,2})[-/,\s]+(\d{4})", s)
    if m:
        d, mon, y = m.groups()
        if mon.isdigit():
            month = int(mon)
        else:
            month = _MONTHS.get(mon.lower())
        if month:
            try:
                return date(int(y), month, int(d))
            except ValueError:
                pass
    # July 13, 2026  /  June 26, 2026
    m = re.search(r"([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})", s)
    if m:
        mon, d, y = m.groups()
        month = _MONTHS.get(mon.lower())
        if month:
            try:
                return date(int(y), month, int(d))
            except ValueError:
                pass
    return None


# ----------------------------------------------------------------------------
# Treasury Bills
# ----------------------------------------------------------------------------
def fetch_tbills():
    """
    Latest weighted-average rate for each T-bill tenor (91/182/364 day).

    CBK shows these as text cards, e.g.:
        "91-DAY  Issue Number: 2695/091  Auction Date: 13th August 2026
         Value Dated: 17th August 2026  Previous Average Interest Rate: 8.7820%"
    We anchor on those stable labels (not table structure, which is unreliable
    on this page). Returns a list 91→182→364, or [] on any failure.
    """
    r = _get(TBILLS_URL, timeout=(3, 10))
    if not r or getattr(r, "status_code", 0) != 200:
        logger.warning("T-bills page unavailable")
        return []

    import html as H
    text = " ".join(H.unescape(re.sub(r"<[^>]+>", " ", r.text)).split())

    out = []
    for tenor in (91, 182, 364):
        # segment from this tenor's label up to its "Previous Average Interest Rate"
        m = re.search(rf"(?<!\d){tenor}-DAY(.*?)Previous Average Interest Rate:\s*([\d.]+)\s*%",
                      text, re.I | re.S)
        if not m:
            continue
        seg, rate_s = m.group(1), m.group(2)
        # guard: the captured segment must be short (a single card), not a run
        # that swallowed other tenors
        if len(seg) > 400:
            continue
        try:
            rate = float(rate_s)
        except ValueError:
            continue
        if not (0 < rate < 30):  # sanity: a T-bill rate, never a price
            continue
        issue = (re.search(r"Issue Number:\s*(\S+)", seg) or [None, None])[1]
        auc = re.search(r"Auction Date:\s*([0-9]{1,2}[a-z]{0,2}\s+[A-Za-z]+\s+\d{4})", seg, re.I)
        val = re.search(r"Value\s*Date[d]?:\s*([0-9]{1,2}[a-z]{0,2}\s+[A-Za-z]+\s+\d{4})", seg, re.I)
        out.append({
            "tenor_days": tenor,
            "label": f"{tenor}-Day",
            "issue": issue,
            "rate": round(rate, 4),
            "auction_date": auc.group(1) if auc else None,
            "value_date": val.group(1) if val else None,
        })
    if out:
        summary = ", ".join(f"{o['tenor_days']}d={o['rate']}%" for o in out)
        logger.info(f"T-bills: {summary}")
    return out


# ----------------------------------------------------------------------------
# Treasury Bonds (prospectus PDFs)
# ----------------------------------------------------------------------------
def list_prospectus_links(limit=14):
    """Return absolute URLs of the most-recent bond prospectus PDFs (newest first)."""
    r = _get(BONDS_URL, timeout=(3, 10))
    if not r or getattr(r, "status_code", 0) != 200:
        logger.warning("Bonds page unavailable")
        return []
    hrefs = re.findall(
        r'href=["\']([^"\']*treasury_bonds_prospectuses[^"\']*\.pdf)["\']',
        r.text, re.I)
    out, seen = [], set()
    for h in hrefs:
        if h.startswith("/"):
            h = CBK + h
        elif not h.startswith("http"):
            h = CBK + "/" + h
        if h not in seen:
            seen.add(h)
            out.append(h)
        if len(out) >= limit:
            break
    return out


def _row(lines, label):
    for l in lines:
        if l.upper().startswith(label.upper()):
            return l
    return ""


def parse_prospectus(pdf_bytes, url=""):
    """
    Parse one CBK bond prospectus PDF into a list of per-bond dicts.

    Extracts issue number, coupon, tenor, maturity, withholding-tax %, sale
    window, auction date and (for re-opened bonds) the clean price. Missing
    fields are left None. Returns [] if the PDF can't be read or looks like a
    switch/rollover auction (which isn't a straight new-money offer).
    """
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            txt = "\n".join((p.extract_text() or "") for p in pdf.pages[:3])
    except Exception as e:
        logger.warning(f"Prospectus parse failed ({url[-40:]}): {e}")
        return []

    if "SWITCH" in txt.upper()[:400] or "switch" in url.lower():
        return []  # rollover auction — not a plain buy offer

    lines = [l.strip() for l in txt.split("\n") if l.strip()]
    issues = re.findall(r"\b((?:FXD|IFB|SDB)\d?/\d{4}/\d{2,3})\b", _row(lines, "ISSUE NUMBER"))
    if not issues:
        return []
    coupons = re.findall(r"\d+\.\d{2,4}", _row(lines, "COUPON RATE"))
    tenors = re.findall(r"(\d+\.?\d*)\s*years?", _row(lines, "TENOR"), re.I)
    maturities = re.findall(r"\d{2}-[A-Za-z]{3}-\d{4}", _row(lines, "MATURITY DATE"))
    wht = re.findall(r"N/?A|\d+\.?\d*", _row(lines, "WITHHOLDING TAX"))
    # Prices come as "... clean price (KES 99.9620) plus AI (KES ...)" and the
    # accrued interest as "(AI) of KES 3.8413 per KES 100" — one per bond, in
    # order. Dirty price = clean + accrued interest = what you actually pay.
    clean_prices = re.findall(r"price\s*\(KES\s*([\d,]+\.?\d*)\)", txt, re.I)
    accrued = re.findall(r"\(AI\)\s*of\s*KES\s*([\d,]+\.?\d*)", txt, re.I)

    sale = _row(lines, "PERIOD OF SALE")
    auction = _row(lines, "AUCTION DATE")
    # sale window: "<start> to <end>"
    sale_start = sale_end = None
    mm = re.split(r"\bto\b", sale, maxsplit=1, flags=re.I)
    if len(mm) == 2:
        sale_start = _parse_any_date(mm[0])
        sale_end = _parse_any_date(mm[1])
    auction_date = _parse_any_date(auction)

    # minimum investment
    m_min = re.search(r"Minimum\s+KES\.?\s*([\d,]+)", txt, re.I)
    min_invest = int(m_min.group(1).replace(",", "")) if m_min else None

    bonds = []
    for idx, issue in enumerate(issues):
        typ = issue[:3].upper()
        coupon = float(coupons[idx]) if idx < len(coupons) else None
        tenor = float(tenors[idx]) if idx < len(tenors) else None
        maturity = maturities[idx] if idx < len(maturities) else None
        w = wht[idx] if idx < len(wht) else None
        tax_free = (typ == "IFB") or (isinstance(w, str) and w.upper().startswith("N"))
        withholding = None if tax_free else (float(w) if (w and re.fullmatch(r"\d+\.?\d*", w)) else None)
        def _num(lst, i):
            if i < len(lst):
                try:
                    return float(lst[i].replace(",", ""))
                except ValueError:
                    return None
            return None
        clean = _num(clean_prices, idx)
        ai = _num(accrued, idx)
        dirty = round(clean + ai, 4) if (clean is not None and ai is not None) else None
        bonds.append({
            "issue": issue,
            "type": typ,
            "type_label": {"IFB": "Infrastructure Bond",
                           "FXD": "Fixed-coupon Bond",
                           "SDB": "State Development Bond"}.get(typ, typ),
            "tax_free": tax_free,
            "coupon": round(coupon, 4) if coupon is not None else None,
            "tenor_years": tenor,
            "maturity": maturity,
            "withholding_pct": withholding,
            "clean_price": clean,
            "accrued_interest": ai,
            "dirty_price": dirty,
            "sale_start": sale_start.isoformat() if sale_start else None,
            "sale_end": sale_end.isoformat() if sale_end else None,
            "auction_date": auction_date.isoformat() if auction_date else None,
            "min_invest": min_invest,
            "prospectus_url": url,
        })
    return bonds


def _status(bond, today):
    """open | upcoming | closed — by sale window / auction date vs today (Nairobi).

    The bond's date fields are ISO strings (YYYY-MM-DD) by this point, so parse
    them with date.fromisoformat — NOT _parse_any_date, which reads CBK's prose
    date formats. Getting this wrong silently marks every bond 'closed'.
    """
    def iso(s):
        try:
            return date.fromisoformat(s) if s else None
        except (ValueError, TypeError):
            return None

    end = iso(bond.get("sale_end")) or iso(bond.get("auction_date"))
    start = iso(bond.get("sale_start"))
    if end and today > end:
        return "closed"
    if start and today < start:
        return "upcoming"
    if end and start and start <= today <= end:
        return "open"
    # no window fully parsed but auction still ahead
    if end and today <= end:
        return "open"
    return "closed"


def fetch_bonds(max_prospectuses=8):
    """
    Parse the most recent bond offers into a flat, de-duplicated list with a
    status (open/upcoming/closed). Newest offers first. [] on failure.
    """
    links = list_prospectus_links()
    if not links:
        return []
    today = _today()
    seen, bonds = set(), []
    parsed = 0
    for url in links:
        if parsed >= max_prospectuses:
            break
        r = _get(url, timeout=(3, 20))
        if not r or getattr(r, "status_code", 0) != 200:
            continue
        recs = parse_prospectus(r.content, url=url)
        if not recs:
            continue
        parsed += 1
        for b in recs:
            # One row per bond: keep the most recent offer (links are newest
            # first, so the first time we see an issue is its latest re-opening).
            if b["issue"] in seen:
                continue
            seen.add(b["issue"])
            b["status"] = _status(b, today)
            bonds.append(b)
    logger.info(f"Bonds: parsed {parsed} prospectus(es) → {len(bonds)} bond line(s)")
    return bonds


# ----------------------------------------------------------------------------
# orchestrator (with a once-per-day cache so we don't re-download PDFs)
# ----------------------------------------------------------------------------
def load_treasury(cache_dir="data", logger=None, force=False):
    """
    Return everything the Government Bonds page needs:
        { as_of, tbills:[...], bonds:[...], context:{cbr, inflation_note} }
    Fails safe: any failure yields empty structures. Cached per calendar day.
    """
    lg = logger or get_logger(__name__)
    today = _today().isoformat()
    path = os.path.join(cache_dir, f"treasury_{today.replace('-', '')}.json")
    if not force and os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            lg.info("Treasury securities: using today's cache")
            return data
        except Exception:
            pass

    tbills, bonds = [], []
    try:
        tbills = fetch_tbills()
    except Exception as e:
        lg.warning(f"T-bills failed: {e}")
    try:
        bonds = fetch_bonds()
    except Exception as e:
        lg.warning(f"Bonds failed: {e}")

    # monetary-policy context (reuse Market Pulse's fail-safe CBK fetch)
    context = {"cbr": None, "inflation_note": None}
    try:
        from market_pulse import fetch_cbk
        cbk = fetch_cbk() or {}
        context["cbr"] = cbk.get("cbr_pct")
        context["inflation_note"] = cbk.get("inflation_note")
    except Exception as e:
        lg.warning(f"CBK context failed: {e}")

    data = {"as_of": today, "tbills": tbills, "bonds": bonds,
            "context": context, "source": SOURCE}

    if tbills or bonds:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            lg.debug(f"Treasury cache write failed: {e}")
    return data


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    from logger import setup_logging
    setup_logging()
    d = load_treasury(cache_dir="data", force=True)
    print(f"\nas_of {d['as_of']}  |  {len(d['tbills'])} T-bill tenors  |  {len(d['bonds'])} bond lines")
    print("\nT-BILLS:")
    for t in d["tbills"]:
        print(f"  {t['label']:8} {t['rate']:.4f}%   issue {t['issue']}   value {t['value_date']}")
    print("\nBONDS (newest offers first):")
    for b in d["bonds"]:
        c = f"{b['coupon']:.4f}%" if b["coupon"] is not None else "  ?  "
        tf = "TAX-FREE" if b["tax_free"] else (f"WHT {b['withholding_pct']}%" if b["withholding_pct"] else "")
        pr = f" clean~{b['clean_price']}" if b.get("clean_price") else ""
        print(f"  [{b['status']:8}] {b['issue']:14} {b['type_label']:20} coupon {c:9} "
              f"{str(b['tenor_years'])+'y':7} mat {b['maturity'] or '?':11} {tf}{pr}")
    print(f"\ncontext: CBR={d['context']['cbr']}  inflation={d['context']['inflation_note']}")
