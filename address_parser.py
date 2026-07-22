"""Conservative, deterministic parsing for U.S.-style property addresses.

Only text already present in ``PropertyAddress`` is rearranged.  The parser has
two deterministic stages: a strict first pass and a bounded second pass for
unlabeled alphanumeric/hyphenated suite tokens.  Internal notes are returned to
the processing layer but are never exported in the final database.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


STREET_SUFFIXES = (
    "STREET",
    "ST",
    "AVENUE",
    "AVE",
    "ROAD",
    "RD",
    "BOULEVARD",
    "BLVD",
    "DRIVE",
    "DR",
    "LANE",
    "LN",
    "HIGHWAY",
    "HWY",
    "PARKWAY",
    "PKWY",
    "COURT",
    "CT",
    "CIRCLE",
    "CIR",
    "PLACE",
    "PL",
    "WAY",
    "TERRACE",
    "TER",
    "TRAIL",
    "TRL",
)

SUITE_LABELS = (
    "SUITE",
    "STE",
    "UNIT",
    "APT",
    "APARTMENT",
    "BUILDING",
    "BLDG",
    "FLOOR",
    "FL",
    "ROOM",
    "RM",
    "#",
)

_SUFFIX_PATTERN = "|".join(sorted(STREET_SUFFIXES, key=len, reverse=True))
_WORD_LABEL_PATTERN = "|".join(
    re.escape(label)
    for label in sorted((item for item in SUITE_LABELS if item != "#"), key=len, reverse=True)
)
_STREET_SUFFIX_RE = re.compile(rf"\b(?:{_SUFFIX_PATTERN})\b", re.IGNORECASE)
_ZIP_RE = re.compile(r"(?P<zip>\d{5}(?:-\d{4})?)\s*$")
_STATE_RE = re.compile(r"(?:^|[\s,])(?P<state>[A-Za-z]{2})\s*,?\s*$")
_EXPLICIT_SUITE_RE = re.compile(
    rf"^,?\s*(?P<label>(?:(?:{_WORD_LABEL_PATTERN})\b|\#))\s*#?\s*"
    r"(?P<suite>[A-Za-z0-9][A-Za-z0-9-]*)"
    r"(?:\s*,\s*|\s+)(?P<city>.+)$",
    re.IGNORECASE,
)
_CITY_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z.'-]*$")
_HOUSE_NUMBER_RE = re.compile(r"^\d+[A-Za-z]?(?:-\d+)?$")
_NUMERIC_SUITE_RE = re.compile(r"^\d{1,4}$")
_EXPANDED_SUITE_RE = re.compile(
    r"^(?=.{1,12}$)(?=.*\d)[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$"
)


@dataclass(frozen=True)
class ParseDiagnostics:
    """Non-exported information about deterministic parsing stages."""

    original_status: str
    final_status: str
    notes: str
    review_method: str
    second_pass_corrected: bool


def clean_address_text(value: Any) -> str:
    """Return a whitespace-normalized address string without changing case."""

    if value is None:
        return ""
    try:
        if value != value:  # Handles pandas/NumPy missing values without importing pandas.
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"\s*,\s*", ", ", text)


def is_structural_street(street: str) -> bool:
    """Return whether text has a house number, street name, and known suffix."""

    tokens = clean_address_text(street).replace(",", " ").split()
    return (
        len(tokens) >= 3
        and bool(_HOUSE_NUMBER_RE.fullmatch(tokens[0]))
        and bool(_STREET_SUFFIX_RE.search(street))
    )


def is_plausible_city(city: str) -> bool:
    """Accept one or more alphabetic city-name tokens, but no digits."""

    tokens = clean_address_text(city).strip(" ,").split()
    return bool(tokens) and any(len(token.strip(".'-")) >= 2 for token in tokens) and all(
        _CITY_TOKEN_RE.fullmatch(token) for token in tokens
    )


def is_confident_component_structure(address: str, city: str) -> bool:
    """Check the minimum structure required to mark final data as Parsed."""

    return is_structural_street(address) and is_plausible_city(city)


def _extract_postal_parts(text: str) -> tuple[str, str, str, list[str]]:
    notes: list[str] = []
    zip_code = ""
    state = ""
    body = text

    zip_match = _ZIP_RE.search(body)
    if zip_match:
        zip_code = zip_match.group("zip")
        body = body[: zip_match.start()].rstrip(" ,")
    else:
        notes.append("ZIP code was not present.")

    state_match = _STATE_RE.search(body)
    if state_match:
        state = state_match.group("state").upper()
        body = body[: state_match.start()].rstrip(" ,")
    else:
        notes.append("State abbreviation was not present.")

    return body.strip(" ,"), state, zip_code, notes


def _select_street_suffix(body: str) -> re.Match[str] | None:
    # Choose the first structurally valid suffix so "ST" in "ST LOUIS" is not
    # mistaken for a second street ending.
    for suffix_match in _STREET_SUFFIX_RE.finditer(body):
        if is_structural_street(body[: suffix_match.end()].strip(" ,")):
            return suffix_match
    return None


def _base_result(
    *, address: str, suite: str, city: str, zip_code: str, status: str, notes: list[str]
) -> dict[str, str]:
    return {
        "Address": address,
        "Suite": suite,
        "City": city,
        "Zip Code": zip_code,
        "Parse Status": status,
        "Parse Notes": " ".join(item for item in notes if item),
    }


def parse_address_first_pass(address: str) -> dict[str, str]:
    """Run strict deterministic parsing without expanded suite inference."""

    cleaned = clean_address_text(address)
    if not cleaned:
        return _base_result(
            address="",
            suite="",
            city="",
            zip_code="",
            status="Review Needed",
            notes=["PropertyAddress is blank."],
        )

    body, state, zip_code, notes = _extract_postal_parts(cleaned)
    suffix_match = _select_street_suffix(body)
    if suffix_match is None:
        notes.append("A structurally valid street with a recognized suffix was not found.")
        return _base_result(
            address=body,
            suite="",
            city="",
            zip_code=zip_code,
            status="Review Needed",
            notes=notes,
        )

    street = body[: suffix_match.end()].strip(" ,")
    raw_tail = body[suffix_match.end() :]
    tail = raw_tail.strip(" ,")
    if not tail:
        notes.append("City was not found after the street address.")
        return _base_result(
            address=street,
            suite="",
            city="",
            zip_code=zip_code,
            status="Review Needed",
            notes=notes,
        )

    explicit_match = _EXPLICIT_SUITE_RE.fullmatch(raw_tail.strip())
    if explicit_match:
        city = explicit_match.group("city").strip(" ,")
        if is_plausible_city(city):
            return _base_result(
                address=street,
                suite=explicit_match.group("suite"),
                city=city,
                zip_code=zip_code,
                status="Parsed",
                notes=notes + ["Explicit suite label recognized."],
            )
        notes.append("An explicit suite was found, but the remaining city is malformed.")
        return _base_result(
            address=body,
            suite="",
            city="",
            zip_code=zip_code,
            status="Review Needed",
            notes=notes,
        )

    tokens = tail.split()
    first_token = tokens[0].strip(",")
    remaining_city = " ".join(tokens[1:]).strip(" ,")
    if _NUMERIC_SUITE_RE.fullmatch(first_token):
        if state and zip_code and is_plausible_city(remaining_city):
            return _base_result(
                address=street,
                suite=first_token,
                city=remaining_city,
                zip_code=zip_code,
                status="Parsed",
                notes=notes + ["Unlabeled numeric suite identified from complete structure."],
            )
        notes.append("A numeric token after the suffix remains ambiguous.")
        return _base_result(
            address=body,
            suite="",
            city="",
            zip_code=zip_code,
            status="Review Needed",
            notes=notes,
        )

    if is_plausible_city(tail):
        return _base_result(
            address=street,
            suite="",
            city=tail,
            zip_code=zip_code,
            status="Parsed",
            notes=notes,
        )

    notes.append("The remaining text cannot be separated confidently in the first pass.")
    return _base_result(
        address=body,
        suite="",
        city="",
        zip_code=zip_code,
        status="Review Needed",
        notes=notes,
    )


def parse_address_second_pass(address: str, first_pass: dict[str, str]) -> dict[str, str]:
    """Retry only Review Needed rows with bounded expanded suite patterns."""

    if first_pass["Parse Status"] != "Review Needed":
        return dict(first_pass)

    cleaned = clean_address_text(address)
    if not cleaned:
        return dict(first_pass)

    body, state, zip_code, postal_notes = _extract_postal_parts(cleaned)
    suffix_match = _select_street_suffix(body)
    if suffix_match is None:
        return dict(first_pass)

    street = body[: suffix_match.end()].strip(" ,")
    tail = body[suffix_match.end() :].strip(" ,")
    tokens = tail.split()
    if len(tokens) < 2:
        return dict(first_pass)

    suite_candidate = tokens[0].strip(",")
    city_candidate = " ".join(tokens[1:]).strip(" ,")
    suite_is_expanded = bool(_EXPANDED_SUITE_RE.fullmatch(suite_candidate))
    suite_is_plain_number = bool(_NUMERIC_SUITE_RE.fullmatch(suite_candidate))

    # A bare number still requires state+ZIP evidence.  Alphanumeric or
    # hyphenated tokens such as C1, A-12, or 2-102 are distinctive enough for
    # this bounded second pass when followed by a plausible city.
    suite_allowed = suite_is_expanded and (
        not suite_is_plain_number or bool(state and zip_code)
    )
    if not suite_allowed or not is_plausible_city(city_candidate):
        return dict(first_pass)

    notes = postal_notes + ["Expanded deterministic suite pattern recognized."]
    return _base_result(
        address=street,
        suite=suite_candidate,
        city=city_candidate,
        zip_code=zip_code,
        status="Parsed",
        notes=notes,
    )


def parse_address_detailed(address: str) -> tuple[dict[str, str], ParseDiagnostics]:
    """Return final components plus non-exported stage diagnostics."""

    first = parse_address_first_pass(address)
    final = parse_address_second_pass(address, first)
    second_pass_corrected = (
        first["Parse Status"] == "Review Needed" and final["Parse Status"] == "Parsed"
    )
    method = "Python Second Pass" if second_pass_corrected else "Python"
    diagnostics = ParseDiagnostics(
        original_status=first["Parse Status"],
        final_status=final["Parse Status"],
        notes=final["Parse Notes"],
        review_method=method,
        second_pass_corrected=second_pass_corrected,
    )
    return final, diagnostics


def parse_address(address: str) -> dict[str, str]:
    """Parse one address using both deterministic stages."""

    result, _ = parse_address_detailed(address)
    return result


__all__ = [
    "ParseDiagnostics",
    "STREET_SUFFIXES",
    "SUITE_LABELS",
    "clean_address_text",
    "is_confident_component_structure",
    "is_plausible_city",
    "is_structural_street",
    "parse_address",
    "parse_address_detailed",
    "parse_address_first_pass",
    "parse_address_second_pass",
]
