from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer():
    return TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
