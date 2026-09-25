import html
import re
import string

import pandas as pd

REQUIRED_COLUMNS = ("job_id", "title", "fraudulent")
OPTIONAL_TEXT_COLUMNS = ("company_profile", "description", "requirements", "benefits")
OPTIONAL_CATEGORY_COLUMNS = (
    "location",
    "department",
    "salary_range",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
)
BINARY_COLUMNS = ("telecommuting", "has_company_logo", "has_questions")
JOB_TEXT_COLUMNS = ("title", "company_profile", "description", "requirements", "benefits")


def clean_dataset(df):
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

    result = df.copy()
    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = result[column].map(lambda value: value.strip() if isinstance(value, str) else value)
    result = result.replace(r"^\s*$", pd.NA, regex=True)

    target = pd.to_numeric(result["fraudulent"], errors="coerce")
    valid_rows = result["job_id"].notna() & result["title"].notna() & target.isin([0, 1])
    dropped_invalid = int((~valid_rows).sum())
    result = result.loc[valid_rows].copy()
    result["fraudulent"] = target.loc[valid_rows].astype("int64")

    before_deduplication = len(result)
    result = result.drop_duplicates(subset="job_id", keep="first").copy()
    dropped_duplicate_ids = before_deduplication - len(result)
    if result.empty:
        raise ValueError("No valid rows remain after required-field validation.")

    filled_values = {}
    for column in OPTIONAL_TEXT_COLUMNS:
        if column in result.columns:
            missing = result[column].isna()
            filled_values[column] = int(missing.sum())
            result[column] = result[column].fillna("")

    for column in OPTIONAL_CATEGORY_COLUMNS:
        if column in result.columns:
            missing = result[column].isna()
            filled_values[column] = int(missing.sum())
            result[column] = result[column].fillna("Unknown")

    for column in BINARY_COLUMNS:
        if column in result.columns:
            values = pd.to_numeric(result[column], errors="coerce")
            values = values.where(values.isin([0, 1]))
            missing_count = int(values.isna().sum())
            valid_mode = values.mode()
            if missing_count and valid_mode.empty:
                raise ValueError(f"Cannot impute '{column}': no valid 0/1 values are available.")
            if missing_count:
                values = values.fillna(valid_mode.iloc[0])
            result[column] = values.astype("int64")
            if missing_count:
                filled_values[column] = missing_count

    summary = {
        "input_rows": len(df),
        "dropped_invalid_rows": dropped_invalid,
        "dropped_duplicate_ids": dropped_duplicate_ids,
        "output_rows": len(result),
        "filled_values": filled_values,
    }
    return result, summary


def clean_text(text):
    if text is None or pd.isna(text):
        return ""

    text = html.unescape(str(text)).lower()
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"(?:https?://|www\.)\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_combined_text(df, text_columns):
    result = df.copy()
    columns = list(text_columns)
    for column in columns:
        if column not in result.columns:
            result[column] = ""

    text_data = result[columns].fillna("").astype(str)
    result["combined_text"] = text_data.agg(" ".join, axis=1).str.strip()
    result["clean_text"] = result["combined_text"].apply(clean_text)
    return result
