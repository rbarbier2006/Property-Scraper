"""Optional OpenAI Responses API reviewer with strict local validation.

The model can only propose a rearrangement of one unresolved address at a time.
Every proposal is validated against the original PropertyAddress before it can
be applied.  No tools are supplied to the Responses API request.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
import os
import re
from typing import Any, Literal, Mapping

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from address_parser import SUITE_LABELS, is_confident_component_structure


DEFAULT_OPENAI_MODEL = "gpt-5-nano"
DEFAULT_MAX_REVIEW_ROWS = 25
DEFAULT_TIMEOUT_SECONDS = 30.0

ALLOWED_REVIEW_FIELDS = (
    "Original PropertyAddress",
    "Current Address",
    "Current Suite",
    "Current City",
    "Current Zip Code",
    "Parse Status",
    "Parse Notes",
)

_ZIP_RE = re.compile(r"^\d{5}(?:-\d{4})?$")
_ZIP_AT_END_RE = re.compile(r"(?P<zip>\d{5}(?:-\d{4})?)\s*$")
_STATE_AT_END_RE = re.compile(r"(?:^|[\s,])(?P<state>[A-Za-z]{2})\s*,?\s*$")
_SUITE_RE = re.compile(r"^(?=.{1,16}$)[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_WORD_SUITE_LABELS = {label.upper() for label in SUITE_LABELS if label != "#"}


class AddressReviewProposal(BaseModel):
    """Pydantic Structured Output required from the Responses API."""

    model_config = ConfigDict(extra="forbid")

    corrected_address: str
    corrected_suite: str
    corrected_city: str
    corrected_zip_code: str
    decision: Literal["Corrected", "Unchanged", "Manual Review"]
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(max_length=300)
    information_added: bool
    manual_review_required: bool


@dataclass(frozen=True)
class ReviewerConfig:
    api_key: str = field(default="", repr=False)
    model: str = DEFAULT_OPENAI_MODEL
    max_review_rows: int = DEFAULT_MAX_REVIEW_ROWS
    auto_accept: bool = False
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = 1


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reasons: tuple[str, ...]

    @property
    def summary(self) -> str:
        return "Passed" if self.valid else "Rejected: " + "; ".join(self.reasons)


class MissingAPIKeyError(RuntimeError):
    """Raised without including any secret material."""


class ReviewServiceError(RuntimeError):
    """Safe, user-facing wrapper for provider and structured-output failures."""


def _read_source_value(name: str, secrets: Mapping[str, Any] | None) -> Any:
    if secrets is not None:
        try:
            value = secrets.get(name)
            if value is not None:
                return value
        except Exception:
            pass
    return os.getenv(name)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def load_reviewer_config(secrets: Mapping[str, Any] | None = None) -> ReviewerConfig:
    """Load secrets first, then environment variables, with safe defaults."""

    api_key = str(_read_source_value("OPENAI_API_KEY", secrets) or "").strip()
    model = str(_read_source_value("OPENAI_MODEL", secrets) or DEFAULT_OPENAI_MODEL).strip()
    max_rows_value = _read_source_value("OPENAI_MAX_REVIEW_ROWS", secrets)
    try:
        max_rows = int(max_rows_value) if max_rows_value is not None else DEFAULT_MAX_REVIEW_ROWS
    except (TypeError, ValueError):
        max_rows = DEFAULT_MAX_REVIEW_ROWS
    max_rows = min(max(max_rows, 1), 1000)
    auto_accept = _as_bool(_read_source_value("OPENAI_AUTO_ACCEPT", secrets), False)
    return ReviewerConfig(
        api_key=api_key,
        model=model or DEFAULT_OPENAI_MODEL,
        max_review_rows=max_rows,
        auto_accept=auto_accept,
    )


def create_openai_client(config: ReviewerConfig) -> OpenAI:
    if not config.api_key:
        raise MissingAPIKeyError(
            "AI-Assisted Processing requires OPENAI_API_KEY in Streamlit Secrets "
            "or the environment. Basic Processing and export remain available."
        )
    return OpenAI(
        api_key=config.api_key,
        timeout=config.timeout_seconds,
        max_retries=config.max_retries,
    )


def build_review_payload(
    row: Mapping[str, Any], *, parse_notes: str = ""
) -> dict[str, str]:
    """Build the exact allow-listed data sent for one unresolved row."""

    payload = {
        "Original PropertyAddress": str(row.get("PropertyAddress", "")),
        "Current Address": str(row.get("Address", "")),
        "Current Suite": str(row.get("Suite", "")),
        "Current City": str(row.get("City", "")),
        "Current Zip Code": str(row.get("Zip Code", "")),
        "Parse Status": str(row.get("Parse Status", "")),
        "Parse Notes": str(parse_notes or ""),
    }
    if payload["Parse Status"] != "Review Needed":
        raise ValueError("Only rows marked Review Needed are eligible for AI review.")
    return payload


_REVIEWER_INSTRUCTIONS = """You review one unresolved property address.
You may only rearrange or classify text already present in Original PropertyAddress.
Never guess, complete, correct spelling, look up, or invent a city, ZIP, suite, street,
or any other information. Do not claim verification. Preserve every meaningful token;
suite labels and a two-letter state abbreviation may be omitted from output fields.
If the existing text cannot be separated confidently, return Decision Manual Review,
leave genuinely missing values blank, and set manual_review_required to true.
Set information_added to true if any proposed nonblank information is not directly
traceable to the original text. Keep the explanation short and factual.
"""


def review_address_with_openai(
    payload: Mapping[str, str],
    config: ReviewerConfig,
    *,
    client: Any | None = None,
) -> AddressReviewProposal:
    """Make one Responses API call using Pydantic Structured Outputs."""

    if tuple(payload.keys()) != ALLOWED_REVIEW_FIELDS:
        raise ValueError("AI review payload contains missing or unsupported fields.")
    if payload["Parse Status"] != "Review Needed":
        raise ValueError("Only rows marked Review Needed are eligible for AI review.")

    active_client = client or create_openai_client(config)
    try:
        response = active_client.responses.parse(
            model=config.model or DEFAULT_OPENAI_MODEL,
            input=[
                {"role": "system", "content": _REVIEWER_INSTRUCTIONS},
                {
                    "role": "user",
                    "content": "Review this single unresolved row:\n"
                    + json.dumps(dict(payload), ensure_ascii=False),
                },
            ],
            text_format=AddressReviewProposal,
            max_output_tokens=500,
            store=False,
        )
        if response.output_parsed is None:
            raise ValueError("No structured output was returned.")
        return AddressReviewProposal.model_validate(response.output_parsed)
    except MissingAPIKeyError:
        raise
    except Exception:
        # Never expose provider responses, request contents, or credentials.
        raise ReviewServiceError("AI review could not be completed for this row.") from None


def _tokens(value: str) -> list[str]:
    return [token.upper() for token in _TOKEN_RE.findall(str(value))]


def _counter_is_subset(candidate: Counter[str], source: Counter[str]) -> bool:
    return all(count <= source[token] for token, count in candidate.items())


def _original_tokens_with_allowed_omissions(
    original: str, proposed_suite: str
) -> Counter[str]:
    """Remove only a recognized state and a suite label from original tokens."""

    working = str(original)
    state_to_omit = ""
    zip_match = _ZIP_AT_END_RE.search(working)
    prefix = working[: zip_match.start()].rstrip(" ,") if zip_match else working.rstrip(" ,")
    state_match = _STATE_AT_END_RE.search(prefix)
    if state_match:
        state_to_omit = state_match.group("state").upper()

    original_tokens = _tokens(working)
    if state_to_omit:
        # Remove the last matching state token only.
        for index in range(len(original_tokens) - 1, -1, -1):
            if original_tokens[index] == state_to_omit:
                del original_tokens[index]
                break

    suite_tokens = _tokens(proposed_suite)
    if suite_tokens:
        for index, token in enumerate(original_tokens):
            if token not in _WORD_SUITE_LABELS:
                continue
            start = index + 1
            if original_tokens[start : start + len(suite_tokens)] == suite_tokens:
                del original_tokens[index]
                break
    return Counter(original_tokens)


def _proposal_components(proposal: AddressReviewProposal) -> dict[str, str]:
    return {
        "Address": proposal.corrected_address.strip(),
        "Suite": proposal.corrected_suite.strip(),
        "City": proposal.corrected_city.strip(),
        "Zip Code": proposal.corrected_zip_code.strip(),
    }


def validate_components(
    original: str,
    current: Mapping[str, Any],
    proposed: Mapping[str, Any],
    *,
    require_confident_structure: bool = True,
) -> ValidationResult:
    """Validate traceability, preservation, ZIP, suite, and final structure."""

    reasons: list[str] = []
    values = {
        "Address": str(proposed.get("Address", "")).strip(),
        "Suite": str(proposed.get("Suite", "")).strip(),
        "City": str(proposed.get("City", "")).strip(),
        "Zip Code": str(proposed.get("Zip Code", "")).strip(),
    }
    current_values = {
        key: str(current.get(key, "")).strip() for key in values
    }

    if values["Zip Code"] and not _ZIP_RE.fullmatch(values["Zip Code"]):
        reasons.append("ZIP format is invalid")
    if values["Suite"] and not _SUITE_RE.fullmatch(values["Suite"]):
        reasons.append("suite format is invalid")

    original_zip_match = _ZIP_AT_END_RE.search(str(original).strip())
    original_zip = original_zip_match.group("zip") if original_zip_match else ""
    if not original_zip and values["Zip Code"]:
        reasons.append("a ZIP was added although none exists in the original")
    if original_zip and values["Zip Code"] != original_zip:
        reasons.append("the original ZIP was removed or changed")
    if current_values["Zip Code"] and _ZIP_RE.fullmatch(current_values["Zip Code"]):
        if values["Zip Code"] != current_values["Zip Code"]:
            reasons.append("a valid current ZIP was removed or changed")

    for component in ("Address", "Suite", "City"):
        if current_values[component] and not values[component]:
            reasons.append(f"a nonblank current {component} was replaced with blank")

    raw_original_counter = Counter(_tokens(original))
    for component, value in values.items():
        if value and not _counter_is_subset(Counter(_tokens(value)), raw_original_counter):
            reasons.append(f"{component} contains text unsupported by the original")

    expected_tokens = _original_tokens_with_allowed_omissions(original, values["Suite"])
    proposed_tokens = Counter(
        _tokens(" ".join([values["Address"], values["Suite"], values["City"], values["Zip Code"]]))
    )
    if proposed_tokens != expected_tokens:
        if _counter_is_subset(proposed_tokens, expected_tokens):
            reasons.append("part of the original address was discarded")
        else:
            reasons.append("the proposed components add or duplicate unsupported text")

    if require_confident_structure and not is_confident_component_structure(
        values["Address"], values["City"]
    ):
        reasons.append("the proposed street and city structure is not confident")

    return ValidationResult(valid=not reasons, reasons=tuple(dict.fromkeys(reasons)))


def validate_ai_proposal(
    original: str,
    current: Mapping[str, Any],
    proposal: AddressReviewProposal,
) -> ValidationResult:
    reasons: list[str] = []
    if proposal.information_added:
        reasons.append("the model reported that information was added")
    components = _proposal_components(proposal)
    component_validation = validate_components(
        original,
        current,
        components,
        require_confident_structure=proposal.decision == "Corrected",
    )
    reasons.extend(component_validation.reasons)
    if proposal.decision == "Corrected" and components == {
        key: str(current.get(key, "")).strip() for key in components
    }:
        reasons.append("the response claims a correction but does not change components")
    return ValidationResult(valid=not reasons, reasons=tuple(dict.fromkeys(reasons)))


def is_auto_acceptable(
    proposal: AddressReviewProposal,
    validation: ValidationResult,
) -> bool:
    return (
        validation.valid
        and proposal.decision == "Corrected"
        and proposal.confidence >= 0.95
        and not proposal.information_added
        and not proposal.manual_review_required
    )


def proposal_components(proposal: AddressReviewProposal) -> dict[str, str]:
    """Public normalized component mapping for workflow/UI code."""

    return _proposal_components(proposal)


__all__ = [
    "ALLOWED_REVIEW_FIELDS",
    "AddressReviewProposal",
    "DEFAULT_OPENAI_MODEL",
    "MissingAPIKeyError",
    "ReviewServiceError",
    "ReviewerConfig",
    "ValidationResult",
    "build_review_payload",
    "create_openai_client",
    "is_auto_acceptable",
    "load_reviewer_config",
    "proposal_components",
    "review_address_with_openai",
    "validate_ai_proposal",
    "validate_components",
]
