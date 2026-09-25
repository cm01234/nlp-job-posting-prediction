-- Sample PostgreSQL queries

SELECT COUNT(*) FROM job_postings;

SELECT fraudulent, COUNT(*)
FROM job_postings
GROUP BY fraudulent;

SELECT title, description, fraudulent
FROM job_postings
LIMIT 10;
