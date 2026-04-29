import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load data
df = pd.read_csv("data.csv")

# 🔹 Clean data (VERY IMPORTANT)
df = df.dropna()  # remove missing values
df['text'] = df['text'].str.strip()  # remove extra spaces
df = df[df['text'] != ""]  # remove empty text rows

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])

# Target
y = df['sentiment']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained successfully!")