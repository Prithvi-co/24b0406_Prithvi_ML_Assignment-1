 import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def predict_review_sentiment(model, vectorizer, review):
    review_vector = vectorizer.transform([review])
    return model.predict(review_vector)[0]

def main():
    np.random.seed(42)

    positive_reviews = [
        "Amazing movie with a fantastic story!",
        "Brilliant acting and direction.",
        "Loved the cinematography and soundtrack.",
        "The film was truly inspiring and heartwarming.",
        "Great performances by the cast.",
        "An emotional rollercoaster done right.",
        "A masterpiece of modern cinema.",
        "Absolutely loved every moment.",
        "Beautifully shot and well-acted.",
        "A delightful surprise and worth watching."
    ] * 5

    negative_reviews = [
        "Terrible movie with a weak plot.",
        "Awful acting and bad direction.",
        "Boring and a complete waste of time.",
        "Very disappointing experience.",
        "The film lacked depth and emotion.",
        "Uninspiring and forgettable.",
        "Bad script and even worse execution.",
        "Couldn't sit through the entire movie.",
        "Poor performances all around.",
        "One of the worst films I've seen."
    ] * 5

    reviews = positive_reviews + negative_reviews
    sentiments = ['positive'] * 50 + ['negative'] * 50

    df = pd.DataFrame({'Review': reviews, 'Sentiment': sentiments})

    vectorizer = CountVectorizer(stop_words='english', max_features=500)
    X = vectorizer.fit_transform(df['Review'])
    y = df['Sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, y_pred))

    test_review = "This movie was incredible with amazing acting"
    print("Predicted Sentiment:", predict_review_sentiment(model, vectorizer, test_review))

if __name__ == "__main__":
    main()
