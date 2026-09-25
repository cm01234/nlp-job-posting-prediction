def predict_label(model, vectorizer, text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return "Fake" if prediction[0] == 1 else "Real"
