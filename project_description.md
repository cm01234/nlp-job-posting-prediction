# Real vs Fake Job Posting Prediction Using Python NLP

## 1. Project Overview

This project builds a Natural Language Processing (NLP) and machine learning system to predict whether an online job posting is:

- Real (legitimate)
- Fake (fraudulent)

Fake job postings can be used to collect personal information, request money from applicants, or carry out other forms of recruitment fraud. Because job descriptions contain large amounts of text, NLP techniques can help identify linguistic patterns associated with fraudulent postings.

The project uses TF-IDF (Term Frequency–Inverse Document Frequency) to convert job-posting text into numerical features and trains classification models to predict the label.

---

## 2. Problem Statement

Online recruitment platforms contain both legitimate and fraudulent job advertisements.

The goal is to build a machine-learning model that can analyze a job posting and estimate whether it is fraudulent.

### Input

A job posting can include fields such as:

- Job title
- Company profile
- Description
- Requirements
- Benefits
- Employment type
- Location
- Salary information

### Output

```text
Real

or

Fake
```

---

## 3. Project Objectives

The main objectives are:

- Load and inspect a job-posting dataset.
- Clean and preprocess textual data.
- Handle missing values.
- Combine relevant text fields.
- Convert text into numerical features using TF-IDF.
- Train machine-learning classification models.
- Evaluate model performance.
- Compare different models.
- Build a function for predicting new job postings.
- Save the trained model for future use.

---

## 4. Dataset

A commonly used dataset for this project is the Real / Fake Job Posting dataset.

A well-known version contains approximately 18,000 job postings and includes a target variable indicating whether a posting is fraudulent.

The target column is commonly:

```text
fraudulent
```

where:

- 0 = Real job posting
- 1 = Fake job posting

Typical columns include:

```text
job_id
title
location
department
salary_range
company_profile
description
requirements
benefits
telecommuting
has_company_logo
has_questions
employment_type
required_experience
required_education
industry
function
fraudulent
```

> The exact columns may vary depending on the dataset version.

---

## 5. Technologies Used

| Technology | Purpose |
| --- | --- |
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning |
| NLTK | Natural language preprocessing |
| Joblib | Model persistence |

---

## 6. Project Structure

```text
real-fake-job-prediction/
├── data/
│   └── fake_job_postings.csv
├── models/
│   ├── tfidf_vectorizer.pkl
│   └── job_fraud_model.pkl
├── notebooks/
│   └── job_posting_prediction.ipynb
├── src/
│   └── train_model.py
├── requirements.txt
├── README.md
└── project_description.md
```

---

## 7. Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk joblib
```

---

## 8. Import Libraries

```python
import re
import string
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
```

---

## 9. Load the Dataset

```python
df = pd.read_csv("data/fake_job_postings.csv")

print("Dataset shape:", df.shape)
print(df.head())
```

Check the available columns:

```python
print(df.columns.tolist())
```

Check the target distribution:

```python
print(df["fraudulent"].value_counts())
```

---

## 10. Exploratory Data Analysis

### 10.1 Check Missing Values

```python
missing_values = df.isnull().sum()
print(missing_values.sort_values(ascending=False))
```

Visualize missing values:

```python
plt.figure(figsize=(12, 6))

missing_values.sort_values(ascending=False).head(15).plot(
    kind="bar",
    color="steelblue",
)

plt.title("Top Columns With Missing Values")
plt.ylabel("Number of Missing Values")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 10.2 Target Distribution

```python
plt.figure(figsize=(6, 4))

sns.countplot(
    x="fraudulent",
    data=df,
    hue="fraudulent",
    legend=False,
)

plt.title("Real vs Fake Job Postings")
plt.xlabel("Fraudulent")
plt.ylabel("Number of Job Postings")
plt.show()
```

The target variable represents:

- 0 → Real
- 1 → Fake

Fraudulent job postings are typically less common than legitimate postings, which makes class imbalance an important consideration.

---

## 11. Data Preprocessing

Text data may contain:

- Missing values
- HTML tags
- URLs
- Punctuation
- Numbers
- Extra whitespace
- Uppercase and lowercase variations

Create a text-cleaning function:

```python
def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
```

---

## 12. Combine Text Features

A job posting may contain useful information in several columns.

We can combine:

- Title
- Company profile
- Description
- Requirements
- Benefits

Make sure these columns exist:

```python
text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]

for column in text_columns:
    if column not in df.columns:
        df[column] = ""
```

Fill missing values:

```python
df[text_columns] = df[text_columns].fillna("")
```

Combine the fields:

```python
df["combined_text"] = df[text_columns].agg(" ".join, axis=1)
```

Clean the combined text:

```python
df["clean_text"] = df["combined_text"].apply(clean_text)
```

Inspect the result:

```python
print(df[["combined_text", "clean_text"]].head())
```

---

## 13. Remove Empty Records

Some records may not contain useful text.

```python
df = df[df["clean_text"].str.strip() != ""].copy()
print("Dataset shape after cleaning:", df.shape)
```

---

## 14. Define Features and Target

```python
X = df["clean_text"]
y = df["fraudulent"]
```

The text is the input:

```text
X = job posting text
```

The target is:

```text
y = real/fake label
```

---

## 15. Train/Test Split

Divide the data into training and testing sets:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)
```

Using `stratify=y` helps preserve the real/fake class distribution in both sets.

---

## 16. TF-IDF Feature Extraction

Machine-learning algorithms cannot directly understand raw text.

TF-IDF converts text into numerical vectors.

TF-IDF gives higher importance to words that are:

- Frequent in a particular document
- Less common across the entire collection of documents

Create the vectorizer:

```python
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
)
```

Fit the vectorizer on training data only:

```python
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
```

Check the dimensions:

```python
print("Training shape:", X_train_tfidf.shape)
print("Testing shape:", X_test_tfidf.shape)
```

---

## 17. Model 1 — Logistic Regression

Logistic Regression is a strong baseline for text classification.

```python
logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)

logistic_model.fit(X_train_tfidf, y_train)
logistic_predictions = logistic_model.predict(X_test_tfidf)
```

Evaluate:

```python
print(
    classification_report(
        y_test,
        logistic_predictions,
        target_names=["Real", "Fake"],
    )
)
```

---

## 18. Model 2 — Multinomial Naive Bayes

Naive Bayes is commonly used for text classification.

```python
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_predictions = nb_model.predict(X_test_tfidf)
```

Evaluate:

```python
print(
    classification_report(
        y_test,
        nb_predictions,
        target_names=["Real", "Fake"],
    )
)
```

---

## 19. Model 3 — Linear Support Vector Machine

Linear SVM is another strong model for high-dimensional text data.

```python
svm_model = LinearSVC(
    class_weight="balanced",
    random_state=42,
)

svm_model.fit(X_train_tfidf, y_train)
svm_predictions = svm_model.predict(X_test_tfidf)
```

Evaluate:

```python
print(
    classification_report(
        y_test,
        svm_predictions,
        target_names=["Real", "Fake"],
    )
)
```

---

## 20. Compare the Models

Create an evaluation function:

```python
def evaluate_model(name, y_true, y_pred):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
    }
```

Calculate metrics:

```python
results = []
```

---

## Summary

This project demonstrates a practical NLP workflow for classifying fake versus real job postings using text preprocessing, TF-IDF feature extraction, and multiple machine-learning models. It is a strong foundation for building an automated recruiter fraud detection system.


results.append(
    evaluate_model(
        "Logistic Regression",
        y_test,
        logistic_predictions
    )
)

results.append(
    evaluate_model(
        "Naive Bayes",
        y_test,
        nb_predictions
    )
)

results.append(
    evaluate_model(
        "Linear SVM",
        y_test,
        svm_predictions
    )
)

results_df = pd.DataFrame(results)

print(results_df)

Visualize:

results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()

21. Understanding the Evaluation Metrics
Accuracy

Accuracy measures the percentage of predictions that are correct.

Accuracy =
Correct Predictions / Total Predictions

Accuracy alone can be misleading when the dataset is imbalanced.
Precision

Precision answers:

    Of the postings predicted as fake, how many were actually fake?

Precision =
True Positives / (True Positives + False Positives)

Recall

Recall answers:

    Of all the actual fake postings, how many did the model detect?

Recall =
True Positives / (True Positives + False Negatives)

F1 Score

F1 combines precision and recall.

F1 = 2 × Precision × Recall / (Precision + Recall)

For a fraud-detection task, examining precision and recall alongside accuracy is particularly important.
22. Confusion Matrix

A confusion matrix helps visualize classification errors.

For the SVM model:

cm = confusion_matrix(
    y_test,
    svm_predictions
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Real", "Fake"],
    yticklabels=["Real", "Fake"]
)

plt.title("Confusion Matrix — Linear SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

The matrix contains:

                 Predicted
               Real    Fake

Actual Real     TN       FP
Actual Fake     FN       TP

23. Predict a New Job Posting

Create a prediction function:

def predict_job_posting(job_text, model, vectorizer):
    cleaned = clean_text(job_text)

    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)[0]

    if prediction == 1:
        return "Fake"
    else:
        return "Real"

Example:

new_job = """
We are looking for a software developer to join our engineering team.
You will develop web applications, write Python code, work with APIs,
and collaborate with other engineers. Applicants should have experience
with Python, Git, SQL, and software development.
"""

prediction = predict_job_posting(
    new_job,
    svm_model,
    tfidf
)

print("Prediction:", prediction)

Possible output:

Prediction: Real

The output is determined by the trained model and dataset. It should not be treated as a guarantee that an individual job posting is legitimate or fraudulent.
24. Save the Model

Save the trained model:

joblib.dump(
    svm_model,
    "models/job_fraud_model.pkl"
)

Save the TF-IDF vectorizer:

joblib.dump(
    tfidf,
    "models/tfidf_vectorizer.pkl"
)

25. Load the Model Later

loaded_model = joblib.load(
    "models/job_fraud_model.pkl"
)

loaded_vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

Use them:

prediction = predict_job_posting(
    new_job,
    loaded_model,
    loaded_vectorizer
)

print("Prediction:", prediction)

26. Complete Training Script

Save the following code as:

src/train_model.py

import re
import string
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ----------------------------------------
# 1. Text Cleaning
# ----------------------------------------

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    text = re.sub(
        r"\d+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ----------------------------------------
# 2. Load Dataset
# ----------------------------------------

df = pd.read_csv(
    "data/fake_job_postings.csv"
)

print("Original dataset shape:")
print(df.shape)


# ----------------------------------------
# 3. Select Text Columns
# ----------------------------------------

text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

for column in text_columns:

    if column not in df.columns:
        df[column] = ""

df[text_columns] = (
    df[text_columns]
    .fillna("")
)


# ----------------------------------------
# 4. Combine Text
# ----------------------------------------

df["combined_text"] = (
    df[text_columns]
    .agg(" ".join, axis=1)
)

df["clean_text"] = (
    df["combined_text"]
    .apply(clean_text)
)


# ----------------------------------------
# 5. Remove Empty Text
# ----------------------------------------

df = df[
    df["clean_text"].str.strip() != ""
].copy()


# ----------------------------------------
# 6. Features and Target
# ----------------------------------------

X = df["clean_text"]

y = df["fraudulent"]


# ----------------------------------------
# 7. Train/Test Split
# ----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


# ----------------------------------------
# 8. TF-IDF
# ----------------------------------------

tfidf = TfidfVectorizer(

    stop_words="english",

    max_features=50000,

    ngram_range=(1, 2),

    min_df=2,

    max_df=0.95,

    sublinear_tf=True
)


X_train_tfidf = (
    tfidf.fit_transform(X_train)
)

X_test_tfidf = (
    tfidf.transform(X_test)
)


# ----------------------------------------
# 9. Train Model
# ----------------------------------------

model = LinearSVC(
    class_weight="balanced",
    random_state=42
)

model.fit(
    X_train_tfidf,
    y_train
)


# ----------------------------------------
# 10. Predictions
# ----------------------------------------

predictions = model.predict(
    X_test_tfidf
)


# ----------------------------------------
# 11. Evaluation
# ----------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)


print("\nModel Performance")
print("-----------------")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)


print("\nClassification Report")
print("---------------------")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Real",
            "Fake"
        ],
        zero_division=0
    )
)


print("\nConfusion Matrix")
print("----------------")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# ----------------------------------------
# 12. Save Model
# ----------------------------------------

joblib.dump(
    model,
    "models/job_fraud_model.pkl"
)

joblib.dump(
    tfidf,
    "models/tfidf_vectorizer.pkl"
)

print("\nModel saved successfully.")

27. Requirements File

Create:

requirements.txt

with:

pandas
numpy
scikit-learn
matplotlib
seaborn
nltk
joblib

Install everything with:

pip install -r requirements.txt

28. Running the Project

From the project directory:

python src/train_model.py

The program will:

    Load the dataset.

    Clean the job descriptions.

    Combine text fields.

    Split the dataset.

    Generate TF-IDF features.

    Train the Linear SVM model.

    Generate predictions.

    Calculate evaluation metrics.

    Display evaluation information.

    Save the trained model.

29. Possible Improvements
29.1 Add Metadata Features

Text is not the only useful information.

Potential additional features include:

has_company_logo
has_questions
telecommuting
employment_type
required_experience
required_education
industry
function

These can be combined with the text features.
29.2 Address Class Imbalance

Fake job postings may represent only a small portion of the dataset.

Possible techniques include:

    Class weighting

    Stratified splitting

    Oversampling

    Undersampling

    SMOTE for suitable numerical feature representations

    Threshold adjustment

The appropriate approach should be selected based on validation results and the desired error trade-offs.
29.3 Try Other Models

Other models can be evaluated, including:

Logistic Regression
Linear SVM
Naive Bayes
Random Forest
Gradient Boosting
XGBoost
LightGBM

For text-heavy data, linear models often provide a useful baseline because TF-IDF produces high-dimensional sparse features.
29.4 Use Word Embeddings

Instead of TF-IDF, the project could use:

    Word2Vec

    GloVe

    FastText

    Sentence Transformers

These methods can capture semantic relationships between words more effectively than simple word-frequency representations.
29.5 Transformer-Based NLP

A more advanced version could use transformer models such as:

BERT
RoBERTa
DistilBERT
DeBERTa

The text could be fine-tuned for binary classification:

Real vs Fake

This generally requires more computational resources than a TF-IDF model.
30. Error Analysis

Model performance should not be judged only by one metric.

Investigate false positives:

Actual: Real
Predicted: Fake

and false negatives:

Actual: Fake
Predicted: Real

False negatives can be particularly important in fraud-detection applications because they represent fraudulent postings that the system failed to identify.

Example:

test_results = pd.DataFrame({
    "text": X_test,
    "actual": y_test,
    "predicted": predictions
})

false_positives = test_results[
    (test_results["actual"] == 0) &
    (test_results["predicted"] == 1)
]

false_negatives = test_results[
    (test_results["actual"] == 1) &
    (test_results["predicted"] == 0)
]

print("False positives:")
print(false_positives.head())

print("\nFalse negatives:")
print(false_negatives.head())

31. Model Interpretation

For a linear model such as Linear SVM, feature coefficients can provide an indication of which terms are associated with each class.

feature_names = tfidf.get_feature_names_out()

coefficients = model.coef_[0]

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
})

print(
    feature_importance
    .sort_values(
        "coefficient",
        ascending=False
    )
    .head(20)
)

The interpretation should be treated carefully: a word's coefficient represents an association learned from the training data, not proof that the word itself indicates fraud.
32. Final Project Workflow

                    Job Posting
                         |
                         v
                Data Collection
                         |
                         v
                Data Cleaning
                         |
                         v
              Text Preprocessing
                         |
                         v
              Combine Text Fields
                         |
                         v
                    TF-IDF
                         |
                         v
              Machine Learning
                         |
                         v
                  Prediction
                         |
                +--------+--------+
                |                 |
                v                 v
              Real              Fake

33. Example Project Output

A typical execution can produce output similar to:

Original dataset shape:
(17880, 18)

Model Performance
-----------------
Accuracy : 0.97
Precision: 0.78
Recall   : 0.74
F1 Score : 0.76

Classification Report
---------------------

              precision    recall  f1-score   support

        Real       ...
        Fake       ...

Model saved successfully.

These numbers are illustrative only. Actual results depend on the exact dataset version, preprocessing choices, train/test split, and model configuration.
34. Conclusion

This project demonstrates how NLP and machine learning can be applied to detect potentially fraudulent job postings.

The main pipeline consists of:

Raw Job Posting
       ↓
Data Cleaning
       ↓
Text Combination
       ↓
TF-IDF Vectorization
       ↓
Machine Learning
       ↓
Real/Fake Prediction

A TF-IDF representation combined with a linear classifier provides a relatively simple and computationally efficient baseline.

The project can be further improved by incorporating structured job-posting metadata, better handling of class imbalance, systematic hyperparameter tuning, richer semantic representations, and transformer-based NLP models.

Importantly, a machine-learning prediction should be considered a risk signal rather than definitive proof that a job posting is fraudulent. A real-world system should combine model output with other verification procedures.
35. Skills Demonstrated

This project demonstrates practical knowledge of:

    Python

    Natural Language Processing

    Text preprocessing

    Feature engineering

    TF-IDF

    Classification

    Logistic Regression

    Naive Bayes

    Support Vector Machines

    Imbalanced classification

    Model evaluation

    Precision and recall

    F1 score

    Confusion matrices

    Model persistence

    Exploratory data analysis

    Scikit-learn

    Pandas

    Data visualization

36. Suggested GitHub Project Description

# Real vs Fake Job Posting Prediction

An NLP-based machine learning project for detecting potentially fraudulent
job postings. The project preprocesses job descriptions, converts text into
TF-IDF features, and uses machine-learning classifiers to distinguish between
legitimate and fraudulent job advertisements.

Technologies:
Python, Pandas, Scikit-learn, NLP, TF-IDF, Linear SVM, Matplotlib, Seaborn.

The project includes data preprocessing, exploratory data analysis,
classification, model evaluation, confusion-matrix analysis, and a
prediction pipeline for new job postings.
