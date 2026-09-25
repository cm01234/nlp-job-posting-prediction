# TODO

## Project Setup
- [x] Create project folder structure
- [x] Set up virtual environment
- [x] Install required dependencies
- [x] Create `requirements.txt` if missing
- [x] Create `README.md` with project overview

## Data Preparation
- [x] Download or locate the real/fake job posting dataset
- [x] Inspect dataset columns and schema
- [x] Check target label distribution
- [x] Identify missing values and null patterns
- [x] Understand the actual data content before preprocessing
- [x] Document dataset fields and their meaning
- [x] Create a data cleaning pipeline for text fields
- [ ] Remove or fill invalid values
- [ ] Combine relevant text columns into one feature field
- [ ] Clean text content (lowercase, remove HTML, URLs, punctuation, numbers)
- [ ] Remove empty records after preprocessing

## Exploratory Data Analysis
- [ ] Visualize missing value distribution
- [ ] Plot target class distribution
- [ ] Review sample real and fake postings
- [ ] Identify important text patterns related to fake postings

## Feature Engineering
- [ ] Define input feature matrix from cleaned text
- [ ] Define target variable (`fraudulent`)
- [ ] Split data into train and test sets with stratification
- [ ] Apply TF-IDF vectorization
- [ ] Tune vectorizer parameters if needed

## Modeling
- [ ] Train a Logistic Regression baseline model
- [ ] Train a Multinomial Naive Bayes model
- [ ] Train a Linear SVM model
- [ ] Compare model performance using metrics
- [ ] Select the best-performing model
- [ ] Save trained vectorizer and model artifacts

## Evaluation
- [ ] Compute accuracy, precision, recall, and F1 score
- [ ] Generate classification report
- [ ] Inspect confusion matrix
- [ ] Check class imbalance impact
- [ ] Document model strengths and weaknesses

## Database / Storage
- [ ] Design PostgreSQL schema based on the dataset structure
- [ ] Create PostgreSQL database and tables for job posting data
- [ ] Define column types and constraints for each field
- [ ] Load the dataset into PostgreSQL
- [ ] Validate imported records and data integrity
- [ ] Create queries for analysis and model input extraction

## Deployment / Usage
- [ ] Build a prediction function for new job postings
- [ ] Create a reusable script for inference
- [ ] Test prediction behavior on sample text
- [ ] Validate output formatting for real/fake classification

## Documentation / Finalization
- [ ] Summarize methodology in project docs
- [ ] Add usage examples to README
- [ ] Review project structure for completeness
- [ ] Confirm reproducibility and requirements list
- [ ] Finalize project presentation or report
