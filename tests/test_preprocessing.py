import pandas as pd

from src.preprocessing import build_combined_text, clean_text


def test_clean_text_removes_markup_urls_email_punctuation_and_numbers():
    text = "Apply at <b>ACME</b>! https://jobs.example/123 contact hiring@example.com for role 2024."

    assert clean_text(text) == "apply at acme contact for role"


def test_clean_text_decodes_html_entities():
    assert clean_text("Research &amp; Development") == "research development"


def test_build_combined_text_handles_nulls_and_missing_columns_without_mutating_input():
    source = pd.DataFrame({"title": ["Data Analyst 2", None], "description": [pd.NA, "<p>Build models</p>"]})

    result = build_combined_text(source, ["title", "description", "requirements"])

    assert result["combined_text"].tolist() == ["Data Analyst 2", "<p>Build models</p>"]
    assert result["clean_text"].tolist() == ["data analyst", "build models"]
    assert result["requirements"].tolist() == ["", ""]
    assert "requirements" not in source.columns
    assert source["title"].tolist()[0] == "Data Analyst 2"