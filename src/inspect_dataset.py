import argparse

import pandas as pd

from src.config import DATA_PATH
from src.data_loader import load_csv


def build_inspection_report(df, target_column="fraudulent"):
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the dataset.")

    schema = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "missing": df.isna().sum(),
        }
    )
    label_counts = df[target_column].value_counts(dropna=False, sort=False)
    label_distribution = pd.DataFrame(
        {
            "count": label_counts,
            "percentage": label_counts.div(len(df)).mul(100),
        }
    )
    return schema, label_distribution


def inspect_dataset(data_path=DATA_PATH, target_column="fraudulent"):
    df = load_csv(data_path)
    schema, label_distribution = build_inspection_report(df, target_column)

    print(f"Dataset: {data_path}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumn schema:")
    print(schema.to_string(index=True))
    print(f"\nTarget distribution: {target_column}")
    print(label_distribution.to_string(index=True, formatters={"percentage": "{:.2f}%".format}))
    return schema, label_distribution


def main():
    parser = argparse.ArgumentParser(description="Inspect dataset columns, inferred schema, and target balance.")
    parser.add_argument("--data-path", default=DATA_PATH, help=f"CSV path (default: {DATA_PATH})")
    parser.add_argument("--target-column", default="fraudulent", help="Name of the target label column")
    args = parser.parse_args()
    inspect_dataset(args.data_path, args.target_column)


if __name__ == "__main__":
    main()