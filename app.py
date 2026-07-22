"""Streamlit interface for the Property CSV Cleaner & Address Parser."""

from __future__ import annotations

import hashlib

import streamlit as st

from data_processing import (
    ProcessingResult,
    create_excel_workbook,
    detect_columns,
    process_dataframe,
    read_csv_bytes,
)


DOWNLOAD_FILENAME = "processed_personal_property_database.xlsx"


def _show_summary(summary: dict[str, int]) -> None:
    first_row = st.columns(4)
    first_row[0].metric("Uploaded", summary["total_uploaded"])
    first_row[1].metric("Personal retained", summary["personal_retained"])
    first_row[2].metric("Real removed", summary["real_removed"])
    first_row[3].metric(
        "Other Type excluded", summary["blank_or_unexpected_excluded"]
    )

    second_row = st.columns(3)
    second_row[0].metric("Parsed", summary["addresses_parsed"])
    second_row[1].metric("Partial", summary["addresses_partial"])
    second_row[2].metric("Review Needed", summary["addresses_review_needed"])


def _show_processed_result(result: ProcessingResult, workbook: bytes) -> None:
    st.success("Processing complete. The Excel database is ready to download.")
    st.subheader("Processing summary")
    _show_summary(result.summary)

    st.subheader("Final database preview")
    st.dataframe(result.database.head(100), use_container_width=True, hide_index=True)
    if len(result.database) > 100:
        st.caption(f"Showing 100 of {len(result.database):,} retained rows.")

    st.subheader("Partial and Review Needed addresses")
    if result.review_needed.empty:
        st.success("No retained addresses require review.")
    else:
        st.warning(
            f"Review {len(result.review_needed):,} row(s). Parse Notes explains why each row "
            "was not considered fully parsed."
        )
        st.dataframe(
            result.review_needed.head(100), use_container_width=True, hide_index=True
        )
        if len(result.review_needed) > 100:
            st.caption(f"Showing 100 of {len(result.review_needed):,} review rows.")

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
        "Upload a property CSV to keep Personal records, parse each PropertyAddress, "
        "review uncertain results, and download a formatted Excel database."
    )
    st.info(
        "Parsing runs locally with deterministic Python rules. No address, geocoding, "
        "AI, or paid web service receives your data."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is None:
        st.info("Choose a .csv file to begin. The download appears only after processing.")
        return

    uploaded_bytes = uploaded_file.getvalue()
    source_hash = hashlib.sha256(uploaded_bytes).hexdigest()
    if st.session_state.get("source_hash") != source_hash:
        st.session_state["source_hash"] = source_hash
        st.session_state.pop("processing_result", None)
        st.session_state.pop("excel_workbook", None)

    try:
        original, encoding = read_csv_bytes(uploaded_bytes)
    except ValueError as exc:
        st.error(f"Could not read the uploaded file. {exc}")
        return
    except Exception as exc:  # Defensive UI boundary for unexpected read failures.
        st.error(f"Could not read the uploaded file: {exc}")
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

    process_clicked = st.button(
        "Process File", type="primary", disabled=bool(detection.errors)
    )
    if process_clicked:
        try:
            with st.spinner("Filtering rows, parsing addresses, and building Excel..."):
                result = process_dataframe(original)
                workbook = create_excel_workbook(result)
            st.session_state["processing_result"] = result
            st.session_state["excel_workbook"] = workbook
        except Exception as exc:  # Keep invalid source data from crashing the app.
            st.error(f"Processing failed: {exc}")
            st.session_state.pop("processing_result", None)
            st.session_state.pop("excel_workbook", None)

    result = st.session_state.get("processing_result")
    workbook = st.session_state.get("excel_workbook")
    if result is not None and workbook is not None:
        _show_processed_result(result, workbook)


if __name__ == "__main__":
    main()
