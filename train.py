import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

from preprocess import clean_text

# Load dataset (TSV format)
df = pd.read_csv("spam.csv", sep='\t', names=['label', 'text'])

# Map labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Drop missing values
df.dropna(inplace=True)

# Clean text
df['text'] = df['text'].apply(clean_text)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Models
nb_model = MultinomialNB()
lr_model = LogisticRegression(max_iter=1000)

# Train
nb_model.fit(X_train_tfidf, y_train)
lr_model.fit(X_train_tfidf, y_train)

# Evaluate
nb_pred = nb_model.predict(X_test_tfidf)
lr_pred = lr_model.predict(X_test_tfidf)

print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_pred))
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))

print("\nNaive Bayes Report:\n", classification_report(y_test, nb_pred))
print("\nLogistic Regression Report:\n", classification_report(y_test, lr_pred))

# Choose best model
best_model = nb_model if accuracy_score(y_test, nb_pred) > accuracy_score(y_test, lr_pred) else lr_model

# Save
joblib.dump(best_model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n✅ Model and vectorizer saved successfully.")