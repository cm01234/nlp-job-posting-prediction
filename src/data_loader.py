import pandas as pd


def load_csv(path):
    return pd.read_csv(path)


def load_data_from_db(conn):
    query = "SELECT * FROM job_postings;"
    return pd.read_sql(query, conn)
