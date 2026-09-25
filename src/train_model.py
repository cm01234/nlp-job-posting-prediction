from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    return model


def train_multinomial_nb(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model


def train_linear_svm(X_train, y_train):
    model = LinearSVC(class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    return model
