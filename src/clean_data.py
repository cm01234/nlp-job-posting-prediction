import argparse
from pathlib import Path

from src.config import DATA_PATH
from src.data_loader import load_csv
from src.preprocessing import JOB_TEXT_COLUMNS, build_combined_text, clean_dataset

OUTPUT_PATH = "data/processed/clean_job_postings.csv"


def run_cleaning(data_path=DATA_PATH, output_path=OUTPUT_PATH):
    source = Path(data_path).resolve()
    destination = Path(output_path).resolve()
    if source == destination:
        raise ValueError("Output path must differ from the input path; raw data will not be overwritten.")

    df = load_csv(data_path)
    cleaned, summary = clean_dataset(df)
    cleaned = build_combined_text(cleaned, JOB_TEXT_COLUMNS)
    empty_text = cleaned["clean_text"].isna() | cleaned["clean_text"].str.strip().eq("")
    summary["dropped_empty_text"] = int(empty_text.sum())
    cleaned = cleaned.loc[~empty_text].copy().reset_index(drop=True)
    summary["output_rows"] = len(cleaned)
    if cleaned.empty:
        raise ValueError("No rows with usable text remain after preprocessing.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(destination, index=False)

    print(f"Input rows: {summary['input_rows']}")
    print(f"Dropped rows with invalid required values: {summary['dropped_invalid_rows']}")
    print(f"Dropped duplicate job IDs: {summary['dropped_duplicate_ids']}")
    print(f"Dropped rows with empty text after preprocessing: {summary['dropped_empty_text']}")
    print(f"Output rows: {summary['output_rows']}")
    print("Filled missing or invalid values:")
    if summary["filled_values"]:
        for column, count in summary["filled_values"].items():
            print(f"  {column}: {count}")
    else:
        print("  none")
    print(f"Cleaned dataset written to: {destination}")
    return cleaned, summary


def main():
    parser = argparse.ArgumentParser(description="Validate and clean job-posting rows and optional values.")
    parser.add_argument("--data-path", default=DATA_PATH, help=f"Input CSV path (default: {DATA_PATH})")
    parser.add_argument("--output-path", default=OUTPUT_PATH, help=f"Output CSV path (default: {OUTPUT_PATH})")
    args = parser.parse_args()
    run_cleaning(args.data_path, args.output_path)


if __name__ == "__main__":
    main()