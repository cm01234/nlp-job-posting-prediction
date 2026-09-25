import pandas as pd
import pytest

from src.inspect_dataset import build_inspection_report


def test_build_inspection_report_returns_schema_and_label_percentages():
    df = pd.DataFrame(
        {
            "title": ["Analyst", None, "Engineer", "Designer"],
            "fraudulent": [0, 0, 1, 0],
        }
    )

    schema, distribution = build_inspection_report(df)

    assert schema.loc["title", "dtype"] == "object"
    assert schema.loc["title", "non_null"] == 3
    assert schema.loc["title", "missing"] == 1
    assert distribution.loc[0, "count"] == 3
    assert distribution.loc[0, "percentage"] == 75
    assert distribution.loc[1, "count"] == 1
    assert distribution.loc[1, "percentage"] == 25


def test_build_inspection_report_requires_target_column():
    with pytest.raises(ValueError, match="Target column 'fraudulent'"):
        build_inspection_report(pd.DataFrame({"title": ["Analyst"]}))