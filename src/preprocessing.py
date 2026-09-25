import html
import re
import string

import pandas as pd


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
    for column in text_columns:
        if column not in result.columns:
            result[column] = ""

    text_data = result[text_columns].fillna("").astype(str)
    result["combined_text"] = text_data.agg(" ".join, axis=1).str.strip()
    result["clean_text"] = result["combined_text"].apply(clean_text)
    return result
