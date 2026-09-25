# Real vs Fake Job Posting Prediction

This project builds a machine learning pipeline to detect whether a job posting is likely real or fake using natural language processing techniques.

## Project Goal

The objective is to classify job advertisements using text from fields such as title, description, requirements, benefits, and company profile. The project combines:

- text preprocessing
- TF-IDF feature extraction
- multiple classification models
- PostgreSQL-backed data storage
- prediction workflows for new job postings

## Main Features

- NLP preprocessing for job description text
- Missing value handling and data normalization
- TF-IDF vectorization for model input
- Baseline classifiers: Logistic Regression, Naive Bayes, and Linear SVM
- Model comparison using accuracy, precision, recall, and F1-score
- PostgreSQL database support for storing and querying job posting data
- Reusable prediction and evaluation scripts

## Suggested Project Structure

```text
nlp-jo-posting-prediction/
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
├── project_description.md
├── TODO.md
│
├── data/
│   ├── raw/
│   │   └── fake_job_postings.csv
│   ├── processed/
│   │   ├── clean_job_postings.csv
│   │   └── train_test_split.csv
│   └── schema/
│       └── job_postings_schema.sql
│
├── database/
│   ├── init.sql
│   ├── schema.sql
│   └── queries/
│       ├── create_tables.sql
│       └── sample_queries.sql
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict.py
│   └── utils.py
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   └── job_fraud_model.pkl
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_database_load.py
│
├── logs/
│   └── training.log
│
└── .venv/
```

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn
- Joblib
- PostgreSQL
- psycopg2

## Setup

1. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure local environment variables for PostgreSQL, if needed:

```bash
cp .env.example .env
```

4. Prepare the dataset and place it under:

```text
data/raw/fake_job_postings.csv
```

5. Run preprocessing and training scripts from the `src` package.

To inspect the CSV columns, inferred pandas data types, missing counts, and target distribution, run from the project root:

```bash
python -m src.inspect_dataset
```

Use `--data-path` to inspect another CSV and `--target-column` to select a different label column.

## Typical Workflow

1. Load dataset from CSV or PostgreSQL.
2. Inspect schema and label distribution.
3. Clean and preprocess text content.
4. Combine text fields into a single feature column.
5. Split into training and test sets.
6. Apply TF-IDF transformation.
7. Train multiple models.
8. Compare metrics and select the best performer.
9. Save models and vectorizer.
10. Use the prediction pipeline to classify new postings.

## Database Integration

This project also supports storing the job posting dataset in PostgreSQL.

A typical workflow is:

- create table schema matching the dataset fields
- load the CSV into PostgreSQL
- run validation queries
- use SQL queries for EDA and model inputs

This helps make the project more production-ready and easier to integrate with real data pipelines.

## Notes

This is a starter structure intended to support a scalable and maintainable machine learning workflow. The implementation can be expanded with CI, containerization, and deployment tooling as the project matures.

## License

This project is for educational and learning purposes.
