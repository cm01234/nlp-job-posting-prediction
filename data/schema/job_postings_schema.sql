-- PostgreSQL schema template for job postings data
-- Adjust column types based on the actual dataset used.

CREATE TABLE IF NOT EXISTS job_postings (
    job_id TEXT,
    title TEXT,
    location TEXT,
    department TEXT,
    salary_range TEXT,
    company_profile TEXT,
    description TEXT,
    requirements TEXT,
    benefits TEXT,
    telecommuting INTEGER,
    has_company_logo INTEGER,
    has_questions INTEGER,
    employment_type TEXT,
    required_experience TEXT,
    required_education TEXT,
    industry TEXT,
    function_name TEXT,
    fraudulent INTEGER
);
