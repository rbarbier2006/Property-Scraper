"""One Streamlit website for Basic and optional AI-Assisted processing."""

from __future__ import annotations

from dataclasses import replace
import hashlib

import streamlit as st

from data_processing import (
    ProcessingResult,
    create_excel_workbook,
    detect_columns,
    process_dataframe,
    read_csv_bytes,
)
from openai_reviewer import MissingAPIKeyError, load_reviewer_config
from review_workflow import (
    apply_review_editor,
    build_review_editor,
    review_eligible_rows,
    uncached_eligible_count,
)


DOWNLOAD_FILENAME = "processed_personal_property_database.xlsx"
BASIC_MODE = "Basic Processing"
AI_MODE = "AI-Assisted Processing"


def _show_summary(summary: dict[str, int]) -> None:
    first = st.columns(4)
    first[0].metric("Uploaded", summary["total_uploaded"])
    first[1].metric("Personal retained", summary["personal_retained"])
    first[2].metric("Real removed", summary["real_removed"])
    first[3].metric("Other Type excluded", summary["blank_or_unexpected_excluded"])

    second = st.columns(4)
    second[0].metric("Parsed in first pass", summary["first_pass_parsed"])
    second[1].metric("Python second-pass corrections", summary["second_pass_corrected"])
    second[2].metric("Still unresolved", summary["addresses_review_needed"])
    second[3].metric("Eligible for AI review", summary["ai_eligible"])


def _clear_file_session_state() -> None:
    for key in (
        "processing_result",
        "ai_review_cache",
        "ai_review_in_progress",
    ):
        st.session_state.pop(key, None)


def _show_ai_controls(
    result: ProcessingResult,
    config,
) -> None:
    st.subheader("Optional AI review")
    st.warning(
        "AI review can consume API credits. For a public deployment, use private "
        "repository/app access or another real access-control layer before enabling it."
    )
    if not config.api_key:
        st.warning(
            "OPENAI_API_KEY is not configured. AI review is disabled, but manual review "
            "and Excel export remain fully available."
        )

    cache = st.session_state.setdefault("ai_review_cache", {})
    pending = uncached_eligible_count(result, cache, config.model)
    eligible = result.summary["ai_eligible"]
    if eligible == 0:
        st.success("No addresses are currently eligible for AI review.")
        return

    maximum_allowed = min(config.max_review_rows, max(pending, 1))
    rows_to_review = st.number_input(
        "Maximum unresolved rows to review in this run",
        min_value=1,
        max_value=max(config.max_review_rows, 1),
        value=maximum_allowed,
        help=(
            "The hard maximum comes from OPENAI_MAX_REVIEW_ROWS. Only unchanged, "
            "uncached Review Needed rows are sent."
        ),
        disabled=pending == 0,
    )
    auto_accept = st.checkbox(
        "Automatically accept validated corrections with confidence of at least 0.95",
        value=config.auto_accept,
        help="Safest default: off. Python validation still applies when enabled.",
    )
    count_for_button = min(int(rows_to_review), pending)
    if pending == 0:
        st.info(
            "All unchanged unresolved addresses have already been reviewed in this "
            "session. They will not be sent again."
        )

    clicked = st.button(
        f"Review {count_for_button} Unresolved Addresses with AI",
        disabled=(
            not config.api_key
            or pending == 0
            or bool(st.session_state.get("ai_review_in_progress"))
        ),
        type="primary",
    )
    if not clicked:
        return

    st.session_state["ai_review_in_progress"] = True
    try:
        effective_config = replace(config, auto_accept=auto_accept)
        with st.spinner("Reviewing allow-listed unresolved addresses only..."):
            run = review_eligible_rows(
                result,
                effective_config,
                cache,
                confirmed=True,
                maximum_rows=count_for_button,
            )
        if run.completed:
            st.success(
                f"AI review completed for {run.completed} row(s). "
                f"Automatically accepted: {run.auto_accepted}."
            )
        if run.failed:
            st.warning(
                f"AI review could not complete for {run.failed} row(s). Python results "
                "were preserved and those rows remain available for manual review."
            )
    except MissingAPIKeyError as exc:
        st.warning(str(exc))
    except Exception:
        st.warning(
            "AI review could not be started. Python results were preserved; manual "
            "review and Excel export are still available."
        )
    finally:
        st.session_state["ai_review_in_progress"] = False


def _show_manual_review(result: ProcessingResult, mode: str, model: str) -> None:
    st.subheader("Manual review")
    if result.review_needed.empty:
        st.success("No unresolved addresses remain.")
        return

    st.info(
        "Edit only by rearranging text already present in PropertyAddress. Unsupported "
        "additions, missing original tokens, invalid ZIP values, and uncertain street/city "
        "structures are rejected. Leave a row unchanged to keep Review Needed."
    )
    cache = st.session_state.setdefault("ai_review_cache", {})
    editor_frame = build_review_editor(result, cache, model)
    noneditable = [
        "Database Row",
        "Review Signature",
        "Original PropertyAddress",
        "Current Address",
        "Current Suite",
        "Current City",
        "Current Zip Code",
        "Proposed AI Address",
        "Proposed AI Suite",
        "Proposed AI City",
        "Proposed AI Zip Code",
        "AI Confidence",
        "AI Explanation",
        "Suggestion Valid",
    ]
    if mode == BASIC_MODE:
        noneditable.extend(["Accept Suggestion", "Reject Suggestion"])

    edited = st.data_editor(
        editor_frame,
        use_container_width=True,
        hide_index=True,
        disabled=noneditable,
        column_config={
            "Database Row": None,
            "Review Signature": None,
            "Accept Suggestion": st.column_config.CheckboxColumn(
                "Accept Suggestion",
                help="Available only for a suggestion that passed Python validation.",
            ),
            "Reject Suggestion": st.column_config.CheckboxColumn("Reject Suggestion"),
            "AI Confidence": st.column_config.NumberColumn(format="%.2f"),
        },
        key=(
            f"review_editor_{st.session_state.get('source_hash', '')}_"
            f"{len(cache)}_{len(result.review_needed)}_{mode}"
        ),
    )
    if st.button("Apply Review Decisions and Manual Edits"):
        outcome = apply_review_editor(result, edited, cache)
        if outcome.applied:
            st.success(f"Applied {outcome.applied} validated correction(s).")
        if outcome.rejected:
            st.info(f"Recorded {outcome.rejected} rejected suggestion(s).")
        for error in outcome.errors:
            st.warning(error)


def _show_processed_result(
    result: ProcessingResult,
    *,
    mode: str,
    config,
) -> None:
    st.success("Python processing is complete. Manual review and export are available.")
    st.subheader("Processing summary")
    _show_summary(result.summary)

    if mode == AI_MODE:
        _show_ai_controls(result, config)

    _show_manual_review(result, mode, config.model)

    st.subheader("Final database preview")
    st.dataframe(result.database.head(100), use_container_width=True, hide_index=True)
    if len(result.database) > 100:
        st.caption(f"Showing 100 of {len(result.database):,} retained rows.")

    workbook = create_excel_workbook(result)
    st.download_button(
        "Download processed Excel workbook",
        data=workbook,
        file_name=DOWNLOAD_FILENAME,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )


def main() -> None:
    st.set_page_config(page_title="Property CSV Cleaner", page_icon="🏠", layout="wide")
    st.title("Property CSV Cleaner & Address Parser")
    st.write(
        "Filter Personal property records, parse addresses with deterministic Python, "
        "optionally review unresolved rows with OpenAI, and export a clean Excel database."
    )

    mode = st.radio(
        "Processing mode",
        [BASIC_MODE, AI_MODE],
        index=0,
        captions=[
            "Python rules only. No API key or OpenAI data transfer.",
            "Python runs first. Only explicitly confirmed Review Needed rows may be sent.",
        ],
    )
    try:
        config = load_reviewer_config(st.secrets)
    except Exception:
        config = load_reviewer_config()

    if mode == BASIC_MODE:
        st.info(
            "Basic Processing is active. No data will be sent to OpenAI and no API key is required."
        )
    else:
        st.info(
            "AI-Assisted Processing is active, but selecting it does not make an API call. "
            "You must process the CSV and then click the separately labeled AI review button."
        )
        if not config.api_key:
            st.warning(
                "No OpenAI API key is configured. The Python workflow, manual review, and "
                "Excel export will continue to work."
            )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is None:
        st.info("Choose a .csv file to begin. The download appears only after processing.")
        return

    uploaded_bytes = uploaded_file.getvalue()
    source_hash = hashlib.sha256(uploaded_bytes).hexdigest()
    if st.session_state.get("source_hash") != source_hash:
        _clear_file_session_state()
        st.session_state["source_hash"] = source_hash

    try:
        original, encoding = read_csv_bytes(uploaded_bytes)
    except ValueError as exc:
        st.error(f"Could not read the uploaded file. {exc}")
        return
    except Exception:
        st.error("Could not read the uploaded CSV. Check that it is a valid CSV file.")
        return

    st.subheader("Uploaded data")
    st.write(f"**Detected columns:** {', '.join(str(column) for column in original.columns)}")
    st.write(f"**Total uploaded rows:** {len(original):,}")
    st.caption(f"CSV encoding used: {encoding}")
    st.dataframe(original.head(10), use_container_width=True, hide_index=True)

    detection = detect_columns(original.columns)
    if detection.legal_description_column is None:
        st.warning(
            "Legal Description was not found. Processing can continue; no column will be removed."
        )
    for error in detection.errors:
        st.error(error)

    if st.button("Process File", type="primary", disabled=bool(detection.errors)):
        try:
            with st.spinner("Filtering rows and running both deterministic parsing passes..."):
                result = process_dataframe(original)
            st.session_state["processing_result"] = result
            st.session_state["ai_review_cache"] = {}
        except Exception:
            st.error(
                "Processing failed. The uploaded file was not saved; check its columns and values."
            )
            st.session_state.pop("processing_result", None)

    result = st.session_state.get("processing_result")
    if result is not None:
        _show_processed_result(result, mode=mode, config=config)


if __name__ == "__main__":
    main()
