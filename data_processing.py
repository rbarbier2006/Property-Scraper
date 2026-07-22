"""CSV cleaning and Excel-export services used by the Streamlit interface."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import re
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
import pandas as pd

from address_parser import parse_address


CANONICAL_COLUMNS = {
    "propertyid": "Property ID",
    "geographicid": "Geographic ID",
    "type": "Type",
    "propertyaddress": "PropertyAddress",
    "legaldescription": "Legal Description",
    "ownername": "Owner Name",
    "doingbusinessas": "Doing Business As",
    "appraisedvalue": "Appraised Value",
}

FINAL_PRIMARY_ORDER = [
    "Property ID",
    "Geographic ID",
    "Type",
    "PropertyAddress",
    "Address",
    "Suite",
    "City",
    "Zip Code",
    "Owner Name",
    "Doing Business As",
    "Appraised Value",
    "Parse Status",
    "Parse Notes",
]

SUMMARY_LABELS = [
    ("total_uploaded", "Total uploaded rows"),
    ("personal_retained", "Personal rows retained"),
    ("real_removed", "Real rows removed"),
    ("blank_or_unexpected_excluded", "Blank or unexpected Type rows excluded"),
    ("addresses_parsed", "Addresses successfully parsed"),
    ("addresses_partial", "Partial addresses"),
    ("addresses_review_needed", "Addresses needing review"),
]

PARSED_COLUMNS = ["Address", "Suite", "City", "Zip Code", "Parse Status", "Parse Notes"]


@dataclass(frozen=True)
class ColumnDetection:
    """Result of normalized input-header detection."""

    canonical_to_actual: dict[str, str]
    errors: list[str]

    @property
    def type_column(self) -> str | None:
        return self.canonical_to_actual.get("Type")

    @property
    def property_address_column(self) -> str | None:
        return self.canonical_to_actual.get("PropertyAddress")

    @property
    def legal_description_column(self) -> str | None:
        return self.canonical_to_actual.get("Legal Description")


@dataclass
class ProcessingResult:
    """Processed database, exception review rows, and summary metrics."""

    database: pd.DataFrame
    review_needed: pd.DataFrame
    summary: dict[str, int]


def normalize_header(header: object) -> str:
    """Normalize a header for case/space/punctuation-insensitive matching."""

    return re.sub(r"[^a-z0-9]+", "", str(header).strip().casefold())


def detect_columns(columns: list[object] | pd.Index) -> ColumnDetection:
    """Locate known headers and report missing or ambiguous required columns."""

    normalized_to_actual: dict[str, list[str]] = {}
    for column in columns:
        normalized_to_actual.setdefault(normalize_header(column), []).append(str(column))

    errors: list[str] = []
    for required_normalized, friendly_name in (
        ("type", "Type"),
        ("propertyaddress", "PropertyAddress or Property Address"),
    ):
        matches = normalized_to_actual.get(required_normalized, [])
        if not matches:
            errors.append(f"Missing required column: {friendly_name}.")
        elif len(matches) > 1:
            errors.append(
                f"Multiple columns match {friendly_name}: {', '.join(matches)}. "
                "Rename or remove the duplicate before processing."
            )

    canonical_to_actual: dict[str, str] = {}
    for normalized, canonical in CANONICAL_COLUMNS.items():
        matches = normalized_to_actual.get(normalized, [])
        if len(matches) == 1:
            canonical_to_actual[canonical] = matches[0]

    return ColumnDetection(canonical_to_actual=canonical_to_actual, errors=errors)


def read_csv_bytes(data: bytes) -> tuple[pd.DataFrame, str]:
    """Read uploaded CSV bytes using common encodings while retaining text."""

    if not data:
        raise ValueError("The uploaded CSV is empty.")

    decode_errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            frame = pd.read_csv(
                BytesIO(data),
                encoding=encoding,
                dtype=str,
                keep_default_na=False,
                na_filter=False,
            )
            return frame, encoding
        except UnicodeDecodeError as exc:
            decode_errors.append(f"{encoding}: {exc}")
        except pd.errors.EmptyDataError as exc:
            raise ValueError("The CSV has no header or data columns.") from exc
        except pd.errors.ParserError as exc:
            raise ValueError(f"The CSV structure could not be read: {exc}") from exc

    raise ValueError(
        "The CSV encoding could not be read as UTF-8, UTF-8 with BOM, or Latin-1. "
        + " | ".join(decode_errors)
    )


def _canonicalize_known_columns(frame: pd.DataFrame) -> pd.DataFrame:
    detection = detect_columns(frame.columns)
    if detection.errors:
        raise ValueError(" ".join(detection.errors))

    rename_map = {
        actual: canonical
        for canonical, actual in detection.canonical_to_actual.items()
        if actual != canonical
    }
    return frame.rename(columns=rename_map).copy()


def process_dataframe(frame: pd.DataFrame) -> ProcessingResult:
    """Filter Personal rows, parse addresses, and remove Legal Description."""

    working = _canonicalize_known_columns(frame)
    type_values = working["Type"].astype(str).str.strip().str.casefold()
    personal_mask = type_values.eq("personal")
    real_mask = type_values.eq("real")

    database = working.loc[personal_mask].copy()
    if "Legal Description" in database.columns:
        database = database.drop(columns=["Legal Description"])

    # Preserve any input columns that happen to use a generated-field name.
    # The generated database schema keeps the requested names, while the source
    # data remains available as an extra "Original ..." column.
    generated_collisions: dict[str, str] = {}
    occupied_names = set(database.columns)
    for column in PARSED_COLUMNS:
        if column not in occupied_names:
            continue
        candidate = f"Original {column}"
        suffix = 2
        while candidate in occupied_names:
            candidate = f"Original {column} {suffix}"
            suffix += 1
        generated_collisions[column] = candidate
        occupied_names.add(candidate)
    if generated_collisions:
        database = database.rename(columns=generated_collisions)

    parsed_records = [parse_address(value) for value in database["PropertyAddress"].tolist()]
    parsed_frame = pd.DataFrame(
        parsed_records, index=database.index, columns=PARSED_COLUMNS
    )
    for generated_column in parsed_frame.columns:
        database[generated_column] = parsed_frame[generated_column]

    ordered_primary = [column for column in FINAL_PRIMARY_ORDER if column in database.columns]
    remaining = [column for column in database.columns if column not in ordered_primary]
    database = database.loc[:, ordered_primary + remaining].reset_index(drop=True)

    status_counts = database["Parse Status"].value_counts().to_dict()
    summary = {
        "total_uploaded": int(len(working)),
        "personal_retained": int(personal_mask.sum()),
        "real_removed": int(real_mask.sum()),
        "blank_or_unexpected_excluded": int((~personal_mask & ~real_mask).sum()),
        "addresses_parsed": int(status_counts.get("Parsed", 0)),
        "addresses_partial": int(status_counts.get("Partial", 0)),
        "addresses_review_needed": int(status_counts.get("Review Needed", 0)),
    }

    review_mask = database["Parse Status"].isin(["Partial", "Review Needed"])
    review_needed = database.loc[review_mask].reset_index(drop=True)
    return ProcessingResult(database=database, review_needed=review_needed, summary=summary)


def _summary_frame(summary: dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame(
        [(label, summary[key]) for key, label in SUMMARY_LABELS],
        columns=["Metric", "Count"],
    )


def _format_worksheet(worksheet, *, text_headers: set[str] | None = None) -> None:
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions

    for cell in worksheet[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")

    text_headers = text_headers or set()
    header_lookup = {cell.value: cell.column for cell in worksheet[1]}
    for header in text_headers:
        column_index = header_lookup.get(header)
        if column_index:
            for cell in worksheet.iter_cols(
                min_col=column_index,
                max_col=column_index,
                min_row=2,
                max_row=max(worksheet.max_row, 2),
            ):
                for item in cell:
                    item.number_format = "@"

    for column_cells in worksheet.columns:
        column_index = column_cells[0].column
        header = str(column_cells[0].value or "")
        max_length = len(header)
        for cell in column_cells[1 : min(len(column_cells), 250)]:
            value = "" if cell.value is None else str(cell.value)
            max_length = max(max_length, max((len(line) for line in value.splitlines()), default=0))
        width_cap = 60 if header in {"PropertyAddress", "Address", "Parse Notes"} else 35
        worksheet.column_dimensions[get_column_letter(column_index)].width = min(
            max(max_length + 2, 12), width_cap
        )


def create_excel_workbook(result: ProcessingResult) -> bytes:
    """Build the three-sheet formatted Excel workbook entirely in memory."""

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        result.database.to_excel(writer, index=False, sheet_name="Database")
        result.review_needed.to_excel(writer, index=False, sheet_name="Review Needed")
        _summary_frame(result.summary).to_excel(
            writer, index=False, sheet_name="Processing Summary"
        )

        workbook = writer.book
        text_headers = {
            header
            for header in result.database.columns
            if normalize_header(header).endswith("id")
            or "zipcode" in normalize_header(header)
        }
        _format_worksheet(workbook["Database"], text_headers=text_headers)
        _format_worksheet(workbook["Review Needed"], text_headers=text_headers)
        _format_worksheet(workbook["Processing Summary"])

    return output.getvalue()


__all__ = [
    "ColumnDetection",
    "ProcessingResult",
    "create_excel_workbook",
    "detect_columns",
    "normalize_header",
    "process_dataframe",
    "read_csv_bytes",
]
