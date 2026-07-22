from __future__ import annotations

from streamlit.testing.v1 import AppTest

import review_workflow


def test_page_load_defaults_to_basic_and_makes_no_ai_call(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    calls = []

    def fail_if_called(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("AI review must not run on page load")

    monkeypatch.setattr(review_workflow, "review_eligible_rows", fail_if_called)
    app = AppTest.from_file("app.py").run(timeout=30)
    assert not app.exception
    assert app.radio[0].value == "Basic Processing"
    assert calls == []
    assert len(app.button) == 0
    assert len(app.download_button) == 0


def test_selecting_ai_mode_without_key_is_safe_and_makes_no_call(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    calls = []

    def fail_if_called(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("Selecting AI mode must not call OpenAI")

    monkeypatch.setattr(review_workflow, "review_eligible_rows", fail_if_called)
    app = AppTest.from_file("app.py").run(timeout=30)
    app.radio[0].set_value("AI-Assisted Processing").run(timeout=30)
    assert not app.exception
    assert calls == []
    assert any("No OpenAI API key" in warning.value for warning in app.warning)
    assert len(app.download_button) == 0


def test_interface_has_no_model_selector(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = AppTest.from_file("app.py").run(timeout=30)
    assert not app.exception
    assert len(app.selectbox) == 0
    labels = [radio.label for radio in app.radio]
    assert all("model" not in label.casefold() for label in labels)
