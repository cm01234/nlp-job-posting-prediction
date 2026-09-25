import pandas as pd

from src.clean_data import run_cleaning


def test_run_cleaning_writes_combined_and_cleaned_text(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "processed" / "clean.csv"
    pd.DataFrame(
        {
            "job_id": [1, 2],
            "title": ["Senior Analyst 2", "<p>123</p>"],
            "company_profile": [None, None],
            "description": ["<p>Build dashboards</p>", None],
            "requirements": ["SQL &amp; data", None],
            "benefits": [None, None],
            "fraudulent": [0, 1],
        }
    ).to_csv(input_path, index=False)

    cleaned, summary = run_cleaning(input_path, output_path)
    saved = pd.read_csv(output_path, keep_default_na=False)
    original = pd.read_csv(input_path, keep_default_na=False)

    assert cleaned["job_id"].tolist() == [1]
    assert cleaned.loc[0, "combined_text"].startswith("Senior Analyst 2")
    assert cleaned.loc[0, "clean_text"] == "senior analyst build dashboards sql data"
    assert saved.loc[0, "clean_text"] == "senior analyst build dashboards sql data"
    assert "combined_text" in saved.columns
    assert summary["dropped_empty_text"] == 1
    assert len(original) == 2
    assert original.loc[1, "title"] == "<p>123</p>"


def test_run_cleaning_refuses_to_overwrite_input(tmp_path):
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"job_id": [1], "title": ["Analyst"], "fraudulent": [0]}).to_csv(input_path, index=False)
    original = input_path.read_bytes()

    try:
        run_cleaning(input_path, input_path)
    except ValueError as error:
        assert "must differ from the input path" in str(error)
    else:
        raise AssertionError("Cleaning should reject an output path that overwrites the input.")

    assert input_path.read_bytes() == original