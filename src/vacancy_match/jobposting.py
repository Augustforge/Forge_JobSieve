"""Read schema.org JobPosting JSON-LD, which hh, most ATS and many boards embed in vacancy pages."""
from __future__ import annotations

import html
import json
import re

_LD = re.compile(r"<script[^>]+type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_BLOCK = re.compile(r"</?(?:p|li|ul|ol|br|h[1-6]|div|tr)\b[^>]*>", re.I)
_LI = re.compile(r"<li\b[^>]*>", re.I)
_ITEM = "\u2063"  # marks a list item until its first line of text is known: <li><p>text</p></li>


def html_to_text(s: str) -> str:
    """Block tags become lines; list items keep a "- " so a short item is never mistaken for a heading."""
    s = html.unescape(_TAG.sub("", _BLOCK.sub("\n", _LI.sub("\n" + _ITEM, s))))
    out, item = [], False
    for line in s.splitlines():
        line = re.sub(r"[ \t\u00a0\u202f]+", " ", line).strip()
        if line.startswith(_ITEM):
            item, line = True, line.lstrip(_ITEM).strip()
        if line:
            out.append(f"- {line}" if item else line)
            item = False
    return "\n".join(out)


def _objects(data):
    if isinstance(data, list):
        for x in data:
            yield from _objects(x)
    elif isinstance(data, dict):
        yield data
        yield from _objects(data.get("@graph") or [])


def _salary(base) -> str:
    if not isinstance(base, dict):
        return ""
    currency, value = base.get("currency", ""), base.get("value")
    if isinstance(value, dict):
        parts = []
        if value.get("minValue") is not None:
            parts.append(f"from {value['minValue']}")
        if value.get("maxValue") is not None:
            parts.append(f"to {value['maxValue']}")
        if not parts and value.get("value") is not None:
            parts.append(str(value["value"]))
        unit = value.get("unitText", "")
        return " ".join(parts + [currency] + ([f"per {unit.lower()}"] if unit else [])).strip()
    return f"{value} {currency}".strip() if value is not None else ""


_EMPLOYMENT = {"FULL_TIME": "full-time", "PART_TIME": "part-time", "CONTRACTOR": "contract",
               "TEMPORARY": "temporary", "INTERN": "internship"}


def _work_format(obj: dict) -> str:
    """Remote flag plus employment type: part-time and contract work must not be judged as a full-time salary."""
    kinds = obj.get("employmentType") or []
    kinds = [kinds] if isinstance(kinds, str) else kinds
    parts = ["remote"] if obj.get("jobLocationType") == "TELECOMMUTE" else []
    return ", ".join(parts + [_EMPLOYMENT.get(str(k).upper(), str(k).lower()) for k in kinds if k])


def parse_jobposting(page_html: str) -> dict | None:
    for block in _LD.findall(page_html):
        try:
            data = json.loads(block.strip())
        except json.JSONDecodeError:
            continue
        for obj in _objects(data):
            types = obj.get("@type")
            if "JobPosting" not in (types if isinstance(types, list) else [types]):
                continue
            org = obj.get("hiringOrganization") or {}
            loc = obj.get("jobLocation") or {}
            loc = loc[0] if isinstance(loc, list) and loc else loc
            addr = loc.get("address", {}) if isinstance(loc, dict) else {}
            city = addr if isinstance(addr, str) else addr.get("addressLocality", "")
            return {
                "title": html.unescape(str(obj.get("title") or "")).strip(),
                "company": html.unescape(str((org.get("name") if isinstance(org, dict) else org) or "")).strip(),
                "text": html_to_text(str(obj.get("description") or "")),
                "published_at": str(obj.get("datePosted", "")),
                "location": city,
                "work_format": _work_format(obj),
                "salary_text": _salary(obj.get("baseSalary")),
            }
    return None
