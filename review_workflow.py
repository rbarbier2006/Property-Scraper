"""Session-scoped AI review, duplicate protection, manual edits, and audit trail."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, MutableMapping

import pandas as pd

from data_processing import ProcessingResult, refresh_processing_result
from openai_reviewer import (
    AddressReviewProposal,
    MissingAPIKeyError,
    ReviewerConfig,
    ValidationResult,
    build_review_payload,
    create_openai_client,
    is_auto_acceptable,
    proposal_components,
    review_address_with_openai,
    validate_ai_proposal,
    validate_components,
)


@dataclass
class ReviewAttempt:
    signature: str
    database_row: int
    payload: dict[str, str]
    current: dict[str, str]
    proposal: AddressReviewProposal | None
    validation: ValidationResult
    error_message: str = ""
    accepted: bool = False


@dataclass(frozen=True)
class ReviewRunSummary:
    requested: int = 0
    completed: int = 0
    failed: int = 0
    auto_accepted: int = 0
    skipped_cached: int = 0


@dataclass(frozen=True)
class ApplyReviewSummary:
    applied: int
    rejected: int
    errors: tuple[str, ...]


def _current_components(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        "Address": str(row.get("Address", "")).strip(),
        "Suite": str(row.get("Suite", "")).strip(),
        "City": str(row.get("City", "")).strip(),
        "Zip Code": str(row.get("Zip Code", "")).strip(),
    }


def make_review_signature(row: Mapping[str, Any], model: str) -> str:
    data = {
        "model": model,
        "PropertyAddress": str(row.get("PropertyAddress", "")),
        **_current_components(row),
        "Parse Status": str(row.get("Parse Status", "")),
    }
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def eligible_row_indices(result: ProcessingResult) -> list[int]:
    return [
        int(index)
        for index in result.database.index[
            result.database["Parse Status"].eq("Review Needed")
        ].tolist()
    ]


def uncached_eligible_count(
    result: ProcessingResult,
    cache: Mapping[str, ReviewAttempt],
    model: str,
) -> int:
    return sum(
        make_review_signature(result.database.loc[index], model) not in cache
        for index in eligible_row_indices(result)
    )


def _diagnostic_value(result: ProcessingResult, row_index: int, column: str) -> str:
    if result.diagnostics.empty:
        return ""
    matches = result.diagnostics.loc[
        result.diagnostics["Database Row"].eq(row_index), column
    ]
    return str(matches.iloc[0]) if not matches.empty else ""


def _set_review_method(result: ProcessingResult, row_index: int, method: str) -> None:
    mask = result.diagnostics["Database Row"].eq(row_index)
    if mask.any():
        result.diagnostics.loc[mask, "Review Method"] = method


def _attempt_audit_record(
    result: ProcessingResult,
    attempt: ReviewAttempt,
    *,
    final_review_status: str,
) -> dict[str, Any]:
    proposal = attempt.proposal
    proposed = proposal_components(proposal) if proposal else {
        "Address": "",
        "Suite": "",
        "City": "",
        "Zip Code": "",
    }
    current_row = result.database.loc[attempt.database_row]
    final = _current_components(current_row)
    return {
        "_Review Signature": attempt.signature,
        "_Database Row": attempt.database_row,
        "Original PropertyAddress": attempt.payload["Original PropertyAddress"],
        "Original Parse Status": _diagnostic_value(
            result, attempt.database_row, "Original Parse Status"
        ),
        "Python Address": attempt.current["Address"],
        "Python Suite": attempt.current["Suite"],
        "Python City": attempt.current["City"],
        "Python Zip Code": attempt.current["Zip Code"],
        "Python Parse Status": "Review Needed",
        "AI Proposed Address": proposed["Address"],
        "AI Proposed Suite": proposed["Suite"],
        "AI Proposed City": proposed["City"],
        "AI Proposed Zip Code": proposed["Zip Code"],
        "AI Confidence": proposal.confidence if proposal else "",
        "AI Explanation": proposal.explanation if proposal else "",
        "AI Decision": proposal.decision if proposal else "",
        "Information Added": proposal.information_added if proposal else "",
        "Manual Review Required": proposal.manual_review_required if proposal else "",
        "Validation Result": (
            "API review failed" if attempt.error_message else attempt.validation.summary
        ),
        "Correction Accepted": attempt.accepted,
        "Review Method": "OpenAI",
        "Final Address": final["Address"],
        "Final Suite": final["Suite"],
        "Final City": final["City"],
        "Final Zip Code": final["Zip Code"],
        "Final Parse Status": str(current_row["Parse Status"]),
        "Final Review Status": final_review_status,
    }


def _upsert_audit(result: ProcessingResult, record: dict[str, Any]) -> None:
    signature = record["_Review Signature"]
    matches = result.ai_review_log["_Review Signature"].eq(signature)
    if matches.any():
        row_index = result.ai_review_log.index[matches][0]
        for column, value in record.items():
            result.ai_review_log.at[row_index, column] = value
    elif result.ai_review_log.empty:
        result.ai_review_log = pd.DataFrame(
            [record], columns=result.ai_review_log.columns
        )
    else:
        result.ai_review_log = pd.concat(
            [result.ai_review_log, pd.DataFrame([record])], ignore_index=True
        )


def _apply_components(
    result: ProcessingResult,
    row_index: int,
    components: Mapping[str, str],
    *,
    method: str,
) -> None:
    for column in ("Address", "Suite", "City", "Zip Code"):
        result.database.at[row_index, column] = str(components[column]).strip()
    result.database.at[row_index, "Parse Status"] = "Parsed"
    _set_review_method(result, row_index, method)


def review_eligible_rows(
    result: ProcessingResult,
    config: ReviewerConfig,
    cache: MutableMapping[str, ReviewAttempt],
    *,
    confirmed: bool,
    maximum_rows: int | None = None,
    client: Any | None = None,
) -> ReviewRunSummary:
    """Review uncached Review Needed rows only after explicit confirmation."""

    if not confirmed:
        return ReviewRunSummary()
    if not config.api_key:
        raise MissingAPIKeyError(
            "AI-Assisted Processing requires OPENAI_API_KEY. "
            "The Python result and manual export are still available."
        )

    eligible = eligible_row_indices(result)
    limit = min(maximum_rows or config.max_review_rows, config.max_review_rows)
    pending: list[tuple[int, str]] = []
    skipped_cached = 0
    for row_index in eligible:
        signature = make_review_signature(result.database.loc[row_index], config.model)
        if signature in cache:
            skipped_cached += 1
            continue
        pending.append((row_index, signature))
        if len(pending) >= limit:
            break

    if not pending:
        return ReviewRunSummary(skipped_cached=skipped_cached)

    active_client = client or create_openai_client(config)
    completed = failed = auto_accepted = 0
    for row_index, signature in pending:
        row = result.database.loc[row_index]
        current = _current_components(row)
        payload = build_review_payload(
            row,
            parse_notes=_diagnostic_value(result, row_index, "Parse Notes"),
        )
        try:
            proposal = review_address_with_openai(payload, config, client=active_client)
            validation = validate_ai_proposal(
                payload["Original PropertyAddress"], current, proposal
            )
            attempt = ReviewAttempt(
                signature=signature,
                database_row=row_index,
                payload=payload,
                current=current,
                proposal=proposal,
                validation=validation,
            )
            completed += 1
            if config.auto_accept and is_auto_acceptable(proposal, validation):
                _apply_components(
                    result,
                    row_index,
                    proposal_components(proposal),
                    method="OpenAI",
                )
                attempt.accepted = True
                auto_accepted += 1
                review_status = "Accepted Automatically"
            elif validation.valid:
                review_status = "Pending Manual Review"
            else:
                review_status = "Rejected by Python Validation"
        except Exception:
            attempt = ReviewAttempt(
                signature=signature,
                database_row=row_index,
                payload=payload,
                current=current,
                proposal=None,
                validation=ValidationResult(False, ("AI review failed",)),
                error_message="AI review failed",
            )
            failed += 1
            review_status = "API Failure - Manual Review"

        cache[signature] = attempt
        _upsert_audit(
            result,
            _attempt_audit_record(
                result, attempt, final_review_status=review_status
            ),
        )

    refresh_processing_result(result)
    return ReviewRunSummary(
        requested=len(pending),
        completed=completed,
        failed=failed,
        auto_accepted=auto_accepted,
        skipped_cached=skipped_cached,
    )


def build_review_editor(
    result: ProcessingResult,
    cache: Mapping[str, ReviewAttempt],
    model: str,
) -> pd.DataFrame:
    """Create the side-by-side editable review table without technical notes."""

    rows: list[dict[str, Any]] = []
    for row_index in eligible_row_indices(result):
        database_row = result.database.loc[row_index]
        current = _current_components(database_row)
        signature = make_review_signature(database_row, model)
        attempt = cache.get(signature)
        proposal = attempt.proposal if attempt else None
        proposed = proposal_components(proposal) if proposal else {
            "Address": "",
            "Suite": "",
            "City": "",
            "Zip Code": "",
        }
        rows.append(
            {
                "Database Row": row_index,
                "Review Signature": signature,
                "Original PropertyAddress": database_row["PropertyAddress"],
                "Current Address": current["Address"],
                "Current Suite": current["Suite"],
                "Current City": current["City"],
                "Current Zip Code": current["Zip Code"],
                "Proposed AI Address": proposed["Address"],
                "Proposed AI Suite": proposed["Suite"],
                "Proposed AI City": proposed["City"],
                "Proposed AI Zip Code": proposed["Zip Code"],
                "AI Confidence": proposal.confidence if proposal else None,
                "AI Explanation": proposal.explanation if proposal else "",
                "Suggestion Valid": bool(attempt and attempt.validation.valid),
                "Accept Suggestion": False,
                "Reject Suggestion": False,
                "Final Address": current["Address"],
                "Final Suite": current["Suite"],
                "Final City": current["City"],
                "Final Zip Code": current["Zip Code"],
            }
        )
    return pd.DataFrame(rows)


def _update_existing_audit(
    result: ProcessingResult,
    signature: str,
    *,
    accepted: bool,
    method: str,
    final_review_status: str,
    row_index: int,
) -> None:
    matches = result.ai_review_log["_Review Signature"].eq(signature)
    if not matches.any():
        return
    log_index = result.ai_review_log.index[matches][0]
    final = _current_components(result.database.loc[row_index])
    updates = {
        "Correction Accepted": accepted,
        "Review Method": method,
        "Final Address": final["Address"],
        "Final Suite": final["Suite"],
        "Final City": final["City"],
        "Final Zip Code": final["Zip Code"],
        "Final Parse Status": result.database.at[row_index, "Parse Status"],
        "Final Review Status": final_review_status,
    }
    for column, value in updates.items():
        result.ai_review_log.at[log_index, column] = value


def _add_manual_audit(
    result: ProcessingResult,
    row_index: int,
    original_current: Mapping[str, str],
    components: Mapping[str, str],
) -> None:
    signature = "manual-" + hashlib.sha256(
        json.dumps(
            {
                "row": row_index,
                "original": result.database.at[row_index, "PropertyAddress"],
                "components": dict(components),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    record = {
        "_Review Signature": signature,
        "_Database Row": row_index,
        "Original PropertyAddress": result.database.at[row_index, "PropertyAddress"],
        "Original Parse Status": _diagnostic_value(result, row_index, "Original Parse Status"),
        "Python Address": original_current["Address"],
        "Python Suite": original_current["Suite"],
        "Python City": original_current["City"],
        "Python Zip Code": original_current["Zip Code"],
        "Python Parse Status": "Review Needed",
        "AI Proposed Address": "",
        "AI Proposed Suite": "",
        "AI Proposed City": "",
        "AI Proposed Zip Code": "",
        "AI Confidence": "",
        "AI Explanation": "",
        "AI Decision": "",
        "Information Added": "",
        "Manual Review Required": "",
        "Validation Result": "Passed",
        "Correction Accepted": True,
        "Review Method": "Manual",
        "Final Address": components["Address"],
        "Final Suite": components["Suite"],
        "Final City": components["City"],
        "Final Zip Code": components["Zip Code"],
        "Final Parse Status": "Parsed",
        "Final Review Status": "Manually Corrected",
    }
    _upsert_audit(result, record)


def apply_review_editor(
    result: ProcessingResult,
    edited: pd.DataFrame,
    cache: Mapping[str, ReviewAttempt],
) -> ApplyReviewSummary:
    """Apply only validated AI suggestions or validated manual rearrangements."""

    applied = rejected = 0
    errors: list[str] = []
    for _, edit in edited.iterrows():
        row_index = int(edit["Database Row"])
        if row_index not in result.database.index:
            continue
        if result.database.at[row_index, "Parse Status"] != "Review Needed":
            continue

        accept = bool(edit.get("Accept Suggestion", False))
        reject = bool(edit.get("Reject Suggestion", False))
        if accept and reject:
            errors.append(f"Row {row_index + 1}: choose Accept or Reject, not both.")
            continue

        signature = str(edit.get("Review Signature", ""))
        attempt = cache.get(signature)
        current = _current_components(result.database.loc[row_index])
        manual_components = {
            "Address": str(edit.get("Final Address", "")).strip(),
            "Suite": str(edit.get("Final Suite", "")).strip(),
            "City": str(edit.get("Final City", "")).strip(),
            "Zip Code": str(edit.get("Final Zip Code", "")).strip(),
        }
        manual_changed = manual_components != current

        if accept:
            if not attempt or not attempt.proposal or not attempt.validation.valid:
                errors.append(
                    f"Row {row_index + 1}: the AI suggestion did not pass Python validation."
                )
                continue
            components = proposal_components(attempt.proposal)
            _apply_components(result, row_index, components, method="OpenAI")
            attempt.accepted = True
            _update_existing_audit(
                result,
                signature,
                accepted=True,
                method="OpenAI",
                final_review_status="Accepted Manually",
                row_index=row_index,
            )
            applied += 1
            continue

        if manual_changed:
            validation = validate_components(
                str(result.database.at[row_index, "PropertyAddress"]),
                current,
                manual_components,
                require_confident_structure=True,
            )
            if not validation.valid:
                errors.append(
                    f"Row {row_index + 1}: manual values were not applied because "
                    f"{'; '.join(validation.reasons)}."
                )
                continue
            _apply_components(
                result, row_index, manual_components, method="Manual"
            )
            if attempt:
                _update_existing_audit(
                    result,
                    signature,
                    accepted=True,
                    method="Manual",
                    final_review_status="Manually Corrected",
                    row_index=row_index,
                )
            else:
                _add_manual_audit(result, row_index, current, manual_components)
            applied += 1
            continue

        if reject:
            rejected += 1
            if attempt:
                _update_existing_audit(
                    result,
                    signature,
                    accepted=False,
                    method="OpenAI",
                    final_review_status="Rejected - Manual Review Required",
                    row_index=row_index,
                )

    refresh_processing_result(result)
    return ApplyReviewSummary(applied=applied, rejected=rejected, errors=tuple(errors))


__all__ = [
    "ApplyReviewSummary",
    "ReviewAttempt",
    "ReviewRunSummary",
    "apply_review_editor",
    "build_review_editor",
    "eligible_row_indices",
    "make_review_signature",
    "review_eligible_rows",
    "uncached_eligible_count",
]
