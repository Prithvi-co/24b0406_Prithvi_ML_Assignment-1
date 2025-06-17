  import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def text_preprocess_vectorize(texts, vectorizer):
    return vectorizer.transform(texts)

def main():
    np.random.seed(42)

    good_feedback = [
        "Excellent quality and fast delivery.",
        "Very satisfied with the purchase.",
        "Great product, will buy again.",
        "Amazing experience and top service.",
        "Value for money and works perfectly.",
        "Highly recommended for everyone.",
        "Outstanding performance and durability.",
        "Loved the packaging and product.",
        "Best decision ever made.",
        "Fantastic! Totally worth it."
    ] * 5

    bad_feedback = [
        "Terrible quality and poor service.",
        "Very disappointed with the product.",
        "Worst purchase I ever made.",
        "Not working as expected.",
        "Broke after one use.",
        "Customer service was unhelpful.",
        "Waste of money and time.",
        "Poorly made and feels cheap.",
        "Regret buying this item.",
        "Defective and returned immediately."
    ] * 5

    texts = good_feedback + bad_feedback
    labels = ['good'] * 50 + ['bad'] * 50

    df = pd.DataFrame({'Text': texts, 'Label': labels})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    vectorizer = TfidfVectorizer(max_features=300, lowercase=True, stop_words='english')
    X = vectorizer.fit_transform(df['Text'])
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    new_texts = [
        "This product is amazing and works well.",
        "I regret buying it. Terrible quality."
    ]
    new_vectors = text_preprocess_vectorize(new_texts, vectorizer)
    predictions = model.predict(new_vectors)

    for text, pred in zip(new_texts, predictions):
        print(f"'{text}' → Prediction: {pred}")

if __name__ == "__main__":
    main()
