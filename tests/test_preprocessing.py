import pandas as pd
import pytest

from src.preprocessing import build_combined_text, clean_dataset, clean_text


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


def test_clean_dataset_drops_invalid_rows_and_ids_and_fills_optional_values():
    source = pd.DataFrame(
        {
            "job_id": [1, 2, 3, 1, 4],
            "title": [" Analyst ", "  ", "Engineer", "Duplicate", "Designer"],
            "fraudulent": ["0", "0", "invalid", "1", "1"],
            "telecommuting": ["invalid", "0", "1", "1", "0"],
            "company_profile": [None, "Profile", "Profile", "Profile", "Profile"],
            "department": [None, "Sales", "Sales", "Sales", " "],
        }
    )

    cleaned, summary = clean_dataset(source)

    assert cleaned["job_id"].tolist() == [1, 4]
    assert cleaned["fraudulent"].tolist() == [0, 1]
    assert cleaned["telecommuting"].tolist() == [0, 0]
    assert cleaned["company_profile"].tolist() == ["", "Profile"]
    assert cleaned["department"].tolist() == ["Unknown", "Unknown"]
    assert summary["dropped_invalid_rows"] == 2
    assert summary["dropped_duplicate_ids"] == 1
    assert summary["filled_values"]["telecommuting"] == 1
    assert source["title"].iloc[0] == " Analyst "


def test_clean_dataset_requires_identifier_title_and_target_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        clean_dataset(pd.DataFrame({"title": ["Analyst"]}))