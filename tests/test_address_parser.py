from __future__ import annotations

from io import BytesIO

from openpyxl import load_workbook
import pandas as pd
import pytest

from address_parser import (
    parse_address,
    parse_address_detailed,
    parse_address_first_pass,
)
from data_processing import (
    FINAL_PRIMARY_ORDER,
    create_excel_workbook,
    detect_columns,
    process_dataframe,
    read_csv_bytes,
)


def assert_parsed(
    source: str,
    *,
    address: str,
    suite: str = "",
    city: str,
    zip_code: str = "",
) -> None:
    result = parse_address(source)
    assert result["Address"] == address
    assert result["Suite"] == suite
    assert result["City"] == city
    assert result["Zip Code"] == zip_code
    assert result["Parse Status"] == "Parsed"


@pytest.mark.parametrize(
    ("source", "address", "suite", "city", "zip_code"),
    [
        (
            "7744 BROADWAY ST 203 SAN ANTONIO, TX 78216",
            "7744 BROADWAY ST",
            "203",
            "SAN ANTONIO",
            "78216",
        ),
        (
            "910 BROADWAY ST SAN ANTONIO, TX 78215",
            "910 BROADWAY ST",
            "",
            "SAN ANTONIO",
            "78215",
        ),
        (
            "120 BROADWAY ST SAN ANTONIO, TX 78205-1904",
            "120 BROADWAY ST",
            "",
            "SAN ANTONIO",
            "78205-1904",
        ),
        (
            "125 BROADWAY ST 300 SAN ANTONIO, TX 78205-1903",
            "125 BROADWAY ST",
            "300",
            "SAN ANTONIO",
            "78205-1903",
        ),
    ],
)
def test_original_sample_addresses(source, address, suite, city, zip_code) -> None:
    assert_parsed(
        source, address=address, suite=suite, city=city, zip_code=zip_code
    )


@pytest.mark.parametrize(
    ("label_and_suite", "expected_suite"),
    [
        ("SUITE 200", "200"),
        ("STE 200A", "200A"),
        ("UNIT A-12", "A-12"),
        ("APT 4B", "4B"),
        ("#444", "444"),
        ("BUILDING C", "C"),
        ("BLDG 7", "7"),
        ("FLOOR 4", "4"),
        ("FL 2", "2"),
        ("ROOM 12B", "12B"),
    ],
)
def test_explicit_suite_labels(label_and_suite: str, expected_suite: str) -> None:
    assert_parsed(
        f"500 Main Street {label_and_suite} Austin, TX 78701",
        address="500 Main Street",
        suite=expected_suite,
        city="Austin",
        zip_code="78701",
    )


@pytest.mark.parametrize("suite", ["C1", "A7", "12B", "A-12", "2-102", "L100"])
def test_second_pass_expanded_suite_formats(suite: str) -> None:
    source = f"6104 BROADWAY ST {suite} ALAMO HEIGHTS"
    first = parse_address_first_pass(source)
    final, diagnostics = parse_address_detailed(source)
    assert first["Parse Status"] == "Review Needed"
    assert final["Parse Status"] == "Parsed"
    assert final["Address"] == "6104 BROADWAY ST"
    assert final["Suite"] == suite
    assert final["City"] == "ALAMO HEIGHTS"
    assert final["Zip Code"] == ""
    assert diagnostics.second_pass_corrected is True
    assert diagnostics.review_method == "Python Second Pass"


def test_second_pass_hyphenated_suite_with_zip_plus_four() -> None:
    assert_parsed(
        "9310 BROADWAY ST 2-102 SAN ANTONIO 78217-5919",
        address="9310 BROADWAY ST",
        suite="2-102",
        city="SAN ANTONIO",
        zip_code="78217-5919",
    )


def test_numbered_and_directional_streets_are_retained() -> None:
    assert_parsed(
        "311 10TH ST SAN ANTONIO, TX 78215",
        address="311 10TH ST",
        city="SAN ANTONIO",
        zip_code="78215",
    )
    assert_parsed(
        "927 N ALAMO ST SAN ANTONIO, TX 78215",
        address="927 N ALAMO ST",
        city="SAN ANTONIO",
        zip_code="78215",
    )


def test_missing_zip_is_not_invented_and_can_still_parse_confident_structure() -> None:
    assert_parsed(
        "123 MAIN ST AUSTIN, TX",
        address="123 MAIN ST",
        city="AUSTIN",
        zip_code="",
    )


def test_missing_state_does_not_prevent_existing_zip_extraction() -> None:
    assert_parsed(
        "123 MAIN ST AUSTIN 78701",
        address="123 MAIN ST",
        city="AUSTIN",
        zip_code="78701",
    )


def test_no_street_suffix_remains_review_needed_without_guessing() -> None:
    result = parse_address("6900 BROADWAY SAN ANTONIO")
    assert result["Parse Status"] == "Review Needed"
    assert result["Address"] == "6900 BROADWAY SAN ANTONIO"
    assert result["Suite"] == ""
    assert result["City"] == ""
    assert result["Zip Code"] == ""


def test_ambiguous_plain_number_without_postal_context_remains_review_needed() -> None:
    result = parse_address("123 MAIN ST 200 AUSTIN")
    assert result["Parse Status"] == "Review Needed"
    assert result["Suite"] == ""
    assert "200" in result["Address"]


@pytest.mark.parametrize("blank", ["", "   ", None, float("nan")])
def test_blank_address_requires_review(blank) -> None:
    result = parse_address(blank)
    assert result["Parse Status"] == "Review Needed"
    assert result["Address"] == ""


def test_malformed_address_preserves_existing_text() -> None:
    result = parse_address("NOT AN ADDRESS")
    assert result["Parse Status"] == "Review Needed"
    assert result["Address"] == "NOT AN ADDRESS"


@pytest.mark.parametrize(
    "source",
    [
        "1 OAK BLVD DALLAS, TX 75001",
        "2 PINE DR DENVER, CO 80202",
        "3 ELM LN PORTLAND, OR 97201",
        "4 CEDAR PKWY PHOENIX, AZ 85001",
        "5 MAPLE CT MIAMI, FL 33101",
        "6 BIRCH CIR CHICAGO, IL 60601",
        "7 ASPEN PL BOISE, ID 83702",
        "8 WALNUT TER ATLANTA, GA 30301",
        "9 SPRUCE TRL NASHVILLE, TN 37201",
    ],
)
def test_supported_street_suffixes_and_states(source: str) -> None:
    assert parse_address(source)["Parse Status"] == "Parsed"


def make_source_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Property ID": ["001", "002", "003", "004", "005"],
            "Geographic ID": ["000-A", "000-B", "000-C", "000-D", "000-E"],
            "Type": ["Personal", " REAL ", " personal ", "", "Commercial"],
            "PropertyAddress": [
                "910 BROADWAY ST SAN ANTONIO, TX 78215",
                "1 MAIN ST AUSTIN, TX 78701",
                "6104 BROADWAY ST C1 ALAMO HEIGHTS",
                "2 MAIN ST AUSTIN, TX 78701",
                "3 MAIN ST AUSTIN, TX 78701",
            ],
            " legal description ": ["remove me"] * 5,
            "Owner Name": ["A", "B", "C", "D", "E"],
            "Doing Business As": ["AA", "BB", "CC", "DD", "EE"],
            "Appraised Value": ["100", "200", "300", "400", "500"],
            "Custom Field": ["keep-1", "keep-2", "keep-3", "keep-4", "keep-5"],
        }
    )


def test_processing_filters_and_counts_type_values() -> None:
    result = process_dataframe(make_source_frame())
    assert result.database["Property ID"].tolist() == ["001", "003"]
    assert result.summary["personal_retained"] == 2
    assert result.summary["real_removed"] == 1
    assert result.summary["blank_or_unexpected_excluded"] == 2
    assert result.summary["second_pass_corrected"] == 1


def test_final_database_is_clean_and_original_address_is_unchanged() -> None:
    source = make_source_frame()
    original = source.loc[0, "PropertyAddress"]
    result = process_dataframe(source)
    assert result.database.loc[0, "PropertyAddress"] == original
    assert result.database.columns[:12].tolist() == FINAL_PRIMARY_ORDER
    assert "Parse Notes" not in result.database.columns
    assert "AI Explanation" not in result.database.columns
    assert "Legal Description" not in result.database.columns
    assert " legal description " not in result.database.columns
    assert result.database.columns[-1] == "Custom Field"


def test_only_two_final_parse_statuses_exist() -> None:
    source = make_source_frame()
    source.loc[0, "PropertyAddress"] = "MALFORMED"
    result = process_dataframe(source)
    assert set(result.database["Parse Status"]) <= {"Parsed", "Review Needed"}
    assert "Partial" not in set(result.database["Parse Status"])


def test_review_needed_view_contains_no_internal_notes() -> None:
    source = make_source_frame()
    source.loc[0, "PropertyAddress"] = "6900 BROADWAY SAN ANTONIO"
    result = process_dataframe(source)
    assert len(result.review_needed) == 1
    assert "Parse Notes" not in result.review_needed.columns
    assert result.review_needed.iloc[0]["Parse Status"] == "Review Needed"


def test_header_detection_ignores_case_spaces_and_minor_variations() -> None:
    detection = detect_columns([" TYPE ", " property Address ", "Legal description"])
    assert detection.errors == []
    assert detection.type_column == " TYPE "
    assert detection.property_address_column == " property Address "
    assert detection.legal_description_column == "Legal description"


def test_header_detection_reports_missing_required_columns() -> None:
    assert len(detect_columns(["Owner Name", "Legal Description"]).errors) == 2


def test_csv_encodings_and_leading_zero_identifiers() -> None:
    utf8 = (
        "Property ID,Type,PropertyAddress\n"
        '"00123",Personal,"1 MAIN ST AUSTIN, TX 78701"\n'
    ).encode("utf-8-sig")
    frame, encoding = read_csv_bytes(utf8)
    assert encoding == "utf-8-sig"
    assert frame.loc[0, "Property ID"] == "00123"

    latin1 = (
        "Type,PropertyAddress,Owner Name\n"
        'Personal,"1 MAIN ST AUSTIN, TX 78701",Jos\xe9\n'
    ).encode("latin-1")
    frame, encoding = read_csv_bytes(latin1)
    assert encoding == "latin-1"
    assert frame.loc[0, "Owner Name"] == "José"


def test_excel_workbook_has_clean_database_and_four_required_sheets() -> None:
    result = process_dataframe(make_source_frame())
    workbook = load_workbook(BytesIO(create_excel_workbook(result)))
    assert workbook.sheetnames == [
        "Database",
        "Review Needed",
        "Processing Summary",
        "AI Review Log",
    ]
    database = workbook["Database"]
    headers = [cell.value for cell in database[1]]
    assert headers[:12] == FINAL_PRIMARY_ORDER
    assert "Parse Notes" not in headers
    assert "AI Explanation" not in headers
    assert database.freeze_panes == "A2"
    assert database.auto_filter.ref
    assert database["A2"].value == "001"
    assert database["A2"].number_format == "@"
    assert database["A1"].font.bold is True
    assert workbook["AI Review Log"].max_row == 1
