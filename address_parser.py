"""Deterministic parsing helpers for U.S.-style property addresses.

The parser intentionally favors review over aggressive guessing.  It extracts
the ZIP code and state first, finds a recognized street suffix, then separates
an explicit suite label or a strongly supported bare numeric suite from the
city.  The state is used as structural evidence but is not returned as a
separate output field because the requested database schema does not include it.
"""

from __future__ import annotations

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
    "BUILDING",
    "BLDG",
    "FLOOR",
    "FL",
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
_BARE_SUITE_RE = re.compile(r"^\d{1,4}$")


def clean_address_text(value: Any) -> str:
    """Return a whitespace-normalized address string without changing case."""

    if value is None:
        return ""
    try:
        # Handles pandas/NumPy missing values without importing pandas here.
        if value != value:
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"\s*,\s*", ", ", text)


def _is_structural_street(street: str) -> bool:
    """Check for a house number, a street name, and a recognized suffix."""

    tokens = street.replace(",", " ").split()
    return (
        len(tokens) >= 3
        and bool(_HOUSE_NUMBER_RE.fullmatch(tokens[0]))
        and bool(_STREET_SUFFIX_RE.search(street))
    )


def _is_plausible_city(city: str) -> bool:
    """Accept one or more alphabetic city-name tokens, but no digits."""

    tokens = city.strip(" ,").split()
    return bool(tokens) and any(len(token.strip(".'-")) >= 2 for token in tokens) and all(
        _CITY_TOKEN_RE.fullmatch(token) for token in tokens
    )


def _extract_postal_parts(text: str) -> tuple[str, str, str, list[str]]:
    """Return body, state, ZIP code, and notes from the right side of a string."""

    notes: list[str] = []
    zip_code = ""
    state = ""
    body = text

    zip_match = _ZIP_RE.search(body)
    if zip_match:
        zip_code = zip_match.group("zip")
        body = body[: zip_match.start()].rstrip(" ,")
    else:
        notes.append("ZIP code was not found at the end of the address.")

    state_match = _STATE_RE.search(body)
    if state_match:
        state = state_match.group("state").upper()
        body = body[: state_match.start()].rstrip(" ,")
    else:
        notes.append("Two-letter state abbreviation was not found before the ZIP code.")

    return body.strip(" ,"), state, zip_code, notes


def _split_location(
    body: str, *, allow_unlabeled_suite: bool
) -> tuple[str, str, str, bool, list[str]]:
    """Split a pre-state location into street, suite, and city.

    The boolean return value indicates a conflict or ambiguity that requires
    manual review instead of a merely incomplete result.
    """

    notes: list[str] = []
    suffix_matches = list(_STREET_SUFFIX_RE.finditer(body))
    selected_match: re.Match[str] | None = None

    # The first structurally valid suffix avoids treating "ST" in a city such
    # as "ST LOUIS" as the end of the street address.
    for suffix_match in suffix_matches:
        candidate_street = body[: suffix_match.end()].strip(" ,")
        if _is_structural_street(candidate_street):
            selected_match = suffix_match
            break

    if selected_match is None:
        # A comma can still preserve a useful city split, but a street without
        # a recognized suffix is not considered confidently parsed.
        if "," in body:
            street, city = (part.strip(" ,") for part in body.rsplit(",", 1))
            if street and _is_plausible_city(city):
                notes.append("Street suffix was not recognized; verify the street and city split.")
                return street, "", city, True, notes

        notes.append("A structurally valid street address with a recognized suffix was not found.")
        return body, "", "", True, notes

    street = body[: selected_match.end()].strip(" ,")
    raw_tail = body[selected_match.end() :]
    tail = raw_tail.strip(" ,")

    if not tail:
        notes.append("City was not found after the street address.")
        return street, "", "", False, notes

    explicit_match = _EXPLICIT_SUITE_RE.fullmatch(raw_tail.strip())
    if explicit_match:
        suite = explicit_match.group("suite")
        city = explicit_match.group("city").strip(" ,")
        label = explicit_match.group("label").upper()
        if _is_plausible_city(city):
            notes.append(f"Explicit {label} suite label recognized.")
            return street, suite, city, False, notes
        notes.append("An explicit suite was found, but the remaining city is malformed.")
        # Keep the uncertain remainder in Address so it is never discarded.
        return body, "", "", True, notes

    tokens = tail.split()
    first_token = tokens[0].strip(",")
    remaining_city = " ".join(tokens[1:]).strip(" ,")

    if _BARE_SUITE_RE.fullmatch(first_token):
        if allow_unlabeled_suite and remaining_city and _is_plausible_city(remaining_city):
            notes.append("Unlabeled numeric suite inferred from strong street/city/state/ZIP structure.")
            return street, first_token, remaining_city, False, notes

        notes.append("A number follows the street suffix, but it cannot be separated from the city safely.")
        return body, "", "", True, notes

    if _is_plausible_city(tail):
        return street, "", tail, False, notes

    if re.search(r"\d", tail) or first_token.upper() in {
        label for label in SUITE_LABELS if label != "#"
    }:
        notes.append("Possible suite or other numeric token is ambiguous and was left in Address.")
    else:
        notes.append("The city portion is malformed or cannot be separated confidently.")
    return body, "", "", True, notes


def parse_address(address: str) -> dict[str, str]:
    """Parse one property address into the requested database fields.

    Returns a dictionary containing ``Address``, ``Suite``, ``City``,
    ``Zip Code``, ``Parse Status``, and ``Parse Notes``.
    """

    cleaned = clean_address_text(address)
    if not cleaned:
        return {
            "Address": "",
            "Suite": "",
            "City": "",
            "Zip Code": "",
            "Parse Status": "Review Needed",
            "Parse Notes": "PropertyAddress is blank.",
        }

    body, state, zip_code, postal_notes = _extract_postal_parts(cleaned)
    street, suite, city, needs_review, location_notes = _split_location(
        body, allow_unlabeled_suite=bool(state and zip_code)
    )
    notes = postal_notes + location_notes

    if needs_review:
        status = "Review Needed"
    elif street and city and state and zip_code:
        status = "Parsed"
    elif street or city or zip_code:
        status = "Partial"
    else:
        status = "Review Needed"

    return {
        "Address": street,
        "Suite": suite,
        "City": city,
        "Zip Code": zip_code,
        "Parse Status": status,
        "Parse Notes": " ".join(notes),
    }


__all__ = ["clean_address_text", "parse_address"]
