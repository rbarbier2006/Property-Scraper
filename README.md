# Property CSV Cleaner & Address Parser

A beginner-friendly Streamlit application that filters a property CSV to
`Personal` records, parses inconsistent U.S.-style property addresses, removes
`Legal Description`, and creates a formatted Excel database. All processing is
local Python: there are no API keys, geocoding calls, AI services, databases, or
runtime network requests.

## Project structure

```text
property-csv-cleaner/
├── .gitignore                   # Local Python files excluded from version control
├── app.py                       # Streamlit user interface
├── address_parser.py            # Conservative deterministic address parser
├── data_processing.py           # CSV cleaning and in-memory Excel export
├── requirements.txt             # Python dependencies
├── README.md                    # Setup and usage guide
└── tests/
    └── test_address_parser.py   # Parser, cleaning, CSV, and Excel tests
```

## How parsing works

The reusable `parse_address(address: str) -> dict` function normalizes spacing,
extracts a trailing five-digit or ZIP+4 code, identifies a two-letter state, and
then looks for a recognized street suffix. Explicit labels such as `SUITE`,
`STE`, `UNIT`, `APT`, `#`, `BLDG`, and `FLOOR` are handled first. A bare numeric
suite is accepted only when the street, city, state, and ZIP structure is strong.
Uncertain text is retained and the row is flagged instead of silently discarded.

The three statuses mean:

- **Parsed** — street, city, state, and ZIP were confidently identified. Suite
  may legitimately be blank.
- **Partial** — useful pieces were extracted, but an expected piece such as the
  state or ZIP is missing.
- **Review Needed** — the address is blank, malformed, conflicting, or includes
  an ambiguous token that the parser deliberately did not guess.

## Install on Windows

Open PowerShell in this folder, then run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, either use Command Prompt and run
`.venv\Scripts\activate.bat`, or run the environment's Python directly:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

## Install on macOS or Linux

Open a terminal in this folder, then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the application

With the virtual environment activated:

```text
streamlit run app.py
```

Streamlit normally opens `http://localhost:8501` in a browser. To use the app:

1. Upload a `.csv` file.
2. Check the detected headers, uploaded row count, and original preview.
3. Fix the CSV if a required `Type` or `PropertyAddress` header is missing.
4. Select **Process File**.
5. Review the metrics and the separate Partial/Review Needed table.
6. Download `processed_personal_property_database.xlsx`.

The workbook contains, in order:

1. **Database** — Personal rows only, original `PropertyAddress`, parsed fields,
   statuses, notes, and any extra input columns.
2. **Review Needed** — only Partial and Review Needed rows.
3. **Processing Summary** — uploaded, retained, excluded, and parsing counts.

`Legal Description` is removed when present. Its absence produces a nonfatal
warning. UTF-8, UTF-8 with BOM, and Latin-1 CSV files are supported. IDs and ZIP
codes are loaded and exported as text to preserve leading zeros.

## Run the automated tests

From this folder with the environment activated:

```text
pytest -q
```

The tests cover the supplied examples, labeled and unlabeled suites, ZIP+4,
numbered/directional streets, city variations, malformed input, filtering,
header matching, CSV encodings, column preservation, and Excel workbook output.

## Optional Streamlit Community Cloud deployment

1. Put this folder in a GitHub repository. Do not upload confidential property
   CSV files.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create an app from the repository and select `app.py` as the entry point (or
   `property-csv-cleaner/app.py` if this folder is inside a larger repository).
4. Deploy. The platform installs `requirements.txt` automatically.

No secrets are required. Uploaded files are processed in the app session and the
Excel workbook is generated in memory.
