# Dataset Profile and Data Dictionary

## Source and inspection

Source file: `data/raw/fake_job_postings.csv`
Records: 17,880
Columns: 18

The CSV has no declared column types. The types below are inferred from field names and observed values. Empty strings and whitespace-only cells are counted as missing. The source parsed into 17,880 rows with no inconsistent row widths; `job_id` values are unique.

## Target distribution

| `fraudulent` | Meaning | Records | Share |
| --- | --- | ---: | ---: |
| 0 | Real / not marked fraudulent | 17,014 | 95.2% |
| 1 | Fraudulent | 866 | 4.8% |

The target is strongly imbalanced. Use stratified splits and class-aware evaluation metrics when modeling.

## Missing values

| Field | Blank records | Share |
| --- | ---: | ---: |
| `job_id` | 0 | 0.0% |
| `title` | 0 | 0.0% |
| `location` | 346 | 1.9% |
| `department` | 11,553 | 64.6% |
| `salary_range` | 15,012 | 84.0% |
| `company_profile` | 3,308 | 18.5% |
| `description` | 1 | <0.1% |
| `requirements` | 2,696 | 15.1% |
| `benefits` | 7,214 | 40.3% |
| `telecommuting` | 0 | 0.0% |
| `has_company_logo` | 0 | 0.0% |
| `has_questions` | 0 | 0.0% |
| `employment_type` | 3,471 | 19.4% |
| `required_experience` | 7,050 | 39.4% |
| `required_education` | 8,105 | 45.3% |
| `industry` | 4,903 | 27.4% |
| `function` | 6,455 | 36.1% |
| `fraudulent` | 0 | 0.0% |

The one blank description belongs to a fraudulent posting. Blank optional fields should not be treated as the literal string `"nan"`; choose explicit missing-value handling during preprocessing.

## Content observations

- Reviewed examples include a real `Marketing Intern` posting and a posting labeled fraudulent titled `IC&E Technician`. These are illustrative examples, not evidence that any single phrase determines the label.
- Text ranges from short titles to long company profiles, descriptions, requirements, and benefits; punctuation, mixed casing, and encoded HTML entities occur.
- A scan found HTML-like tags or entities in at least one of the five main text fields (`title`, `company_profile`, `description`, `requirements`, `benefits`) in 7,552 records. URL-like text occurred in those fields in 150 records.
- `location` is a free-form composite (for example, country/state/city segments) and may have empty segments. `salary_range` is source text, not a normalized numeric value.
- Several categorical fields have many blanks, so missingness itself may be informative. Preserve the original fields until preprocessing choices are evaluated.

## Field dictionary

| Field | Inferred type | Meaning |
| --- | --- | --- |
| `job_id` | Identifier (integer-like text) | Dataset identifier for a posting; keep as an identifier rather than a model feature. |
| `title` | Text | Advertised job title. |
| `location` | Text | Posting location, often country/state/city segments. |
| `department` | Text / category | Employer department associated with the role. |
| `salary_range` | Text | Advertised salary range as supplied; not normalized. |
| `company_profile` | Text | Employer or organization profile. |
| `description` | Text | Main description of the job and role. |
| `requirements` | Text | Qualifications, skills, or requirements for applicants. |
| `benefits` | Text | Benefits offered with the position. |
| `telecommuting` | Binary integer (0/1) | Whether telecommuting is indicated. |
| `has_company_logo` | Binary integer (0/1) | Whether a company logo is indicated. |
| `has_questions` | Binary integer (0/1) | Whether the application includes questions. |
| `employment_type` | Text / category | Employment arrangement, such as full-time or contract. |
| `required_experience` | Text / category | Required experience level. |
| `required_education` | Text / category | Required education level. |
| `industry` | Text / category | Industry associated with the posting. |
| `function` | Text / category | Occupational function associated with the posting. |
| `fraudulent` | Binary integer (0/1) | Target label: 0 means real/not fraudulent; 1 means fraudulent. |

The SQL template currently calls `function` `function_name`; align that name with the source column before using the template to load this CSV.