# Property CSV Cleaner & Address Parser

One Streamlit website with two modes:

- **Basic Processing** (default) runs deterministic Python only. No API key is
  required and no address data is sent to OpenAI.
- **AI-Assisted Processing** always runs both Python passes first. It can send
  only explicitly confirmed rows whose current status is exactly
  `Review Needed`.

OpenAI is limited to rearranging text already present in the original
`PropertyAddress`. Python rejects invented, duplicated, discarded, or malformed
information before it can change the database. The app does not use web search,
file search, Code Interpreter, geocoding, function tools, or external address
services.

## Project structure

```text
property-csv-cleaner/
|-- app.py
|-- address_parser.py
|-- data_processing.py
|-- openai_reviewer.py
|-- review_workflow.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- .streamlit/
|   `-- secrets.toml.example
`-- tests/
    |-- test_address_parser.py
    |-- test_openai_reviewer.py
    `-- test_app_safety.py
```

## Processing and data integrity

The strict first parser pass handles recognized streets, explicit suite labels,
and high-confidence numeric suites. A bounded second pass recognizes existing
tokens such as `C1`, `A7`, `12B`, `A-12`, `2-102`, and `L100`. It never creates a
missing city, suite, ZIP, suffix, or spelling correction.

Final status is either:

- **Parsed** -- the existing information was separated confidently.
- **Review Needed** -- the information is missing, malformed, or ambiguous.

`Parse Notes` is internal only. It is not displayed in the final preview and is
not written to Database or Review Needed.

The workbook contains:

1. **Database** -- Personal rows and the clean final database columns.
2. **Review Needed** -- current unresolved rows with the same clean columns.
3. **Processing Summary** -- filtering and review counts.
4. **AI Review Log** -- proposals, validation outcome, acceptance, and final
   values. It has headers but no rows when AI review has not occurred.

## Install

### Windows PowerShell

```powershell
cd "C:\Users\reneb\Documents\New project\property-csv-cleaner"
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### macOS or Linux

```bash
cd /path/to/property-csv-cleaner
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

Windows:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

macOS or Linux:

```bash
streamlit run app.py
```

Open `http://localhost:8501` if the browser does not open automatically.

## Configure OpenAI locally (optional)

Basic Processing works without any key. To test AI-Assisted Processing:

1. Create a project-specific key in the OpenAI API Platform.
2. Copy `.streamlit/secrets.toml.example` to
   `.streamlit/secrets.toml`.
3. Replace only the placeholder in the local `secrets.toml` file:

```toml
OPENAI_API_KEY = "your-real-project-key"
OPENAI_MODEL = "gpt-5-nano"
OPENAI_MAX_REVIEW_ROWS = 25
OPENAI_AUTO_ACCEPT = false
```

`.streamlit/secrets.toml` is ignored by Git. Never paste the real key into source
code, tests, chat, screenshots, or GitHub. Rotate it immediately if exposed.

Normal environment variables with the same names are also supported. Streamlit
Secrets take precedence. If `OPENAI_MODEL` is missing, the app uses
`gpt-5-nano`. There is intentionally no model selector in the interface and no
automatic model fallback.

`OPENAI_MAX_REVIEW_ROWS` limits chargeable row reviews per run. Automatic
acceptance is off by default; even when enabled, Python accepts only a validated
`Corrected` proposal with confidence at least 0.95, no added information, and no
manual-review flag.

## Use the application

1. Leave **Basic Processing** selected or choose **AI-Assisted Processing**.
2. Upload the original CSV and inspect its detected headers and preview.
3. Click **Process File**. No OpenAI call occurs here.
4. Review the Python metrics and unresolved rows.
5. In AI mode, inspect the eligible count and explicitly click the separately
   labeled **Review ... Unresolved Addresses with AI** button. Selecting AI mode
   alone never sends data.
6. Accept/reject validated suggestions, or edit final fields using only text in
   the original address.
7. Download `processed_personal_property_database.xlsx` at any time after Python
   processing, including after an API failure.

Completed and failed AI attempts are cached in the current Streamlit session so
an unchanged address is not resent when the button is clicked again.

## Run tests

All OpenAI responses are mocked. Tests never need or use a real API key.

```powershell
.venv\Scripts\python.exe -m pytest -q
```

## Deploy with Streamlit Community Cloud

1. Upload the project files to a GitHub repository. Do not upload `.venv`, real
   CSV data, `.env`, or `.streamlit/secrets.toml`.
2. At <https://share.streamlit.io/>, create an app using branch `main` and entry
   point `app.py`.
3. For Basic mode, no secret is required.
4. To enable AI mode, open the deployed app's **Settings > Secrets** and enter:

```toml
OPENAI_API_KEY = "your-real-project-key"
OPENAI_MODEL = "gpt-5-nano"
OPENAI_MAX_REVIEW_ROWS = 25
OPENAI_AUTO_ACCEPT = false
```

Do not put these values in the repository. A public app can let visitors consume
your API credits. Prefer a private deployment or a proper platform access-control
layer before enabling AI for public visitors; do not add a hard-coded password.
