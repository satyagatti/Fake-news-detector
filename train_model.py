import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load Fake and True datasets
df_fake = pd.read_csv("Fake.csv")  # Fake news data
df_true = pd.read_csv("True.csv")  # Real news data

# Add a label column: 1 = Fake, 0 = Real
df_fake["label"] = 1
df_true["label"] = 0

# Combine both datasets
df = pd.concat([df_fake, df_true], ignore_index=True)

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Check dataset structure
print(df.head())

# Select features and labels
X = df["text"]  # News content
y = df["label"]  # Labels (1 = Fake, 0 = Real)

# Convert text data to numerical vectors using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit vectorizer on training data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate model accuracy
accuracy = model.score(X_test_tfidf, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save trained model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model and vectorizer saved successfully!")