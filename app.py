import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk

nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="NLP Model Comparison", layout="wide")
st.title("🍽 Restaurant Review Classification – NLP")

# ---------------- LOAD DATA ----------------
dataset = pd.read_csv(
    "Restaurant_Reviews.tsv",
    delimiter="\t",
    quoting=3
)

# ---------------- PREPROCESSING ----------------
corpus = []
ps = PorterStemmer()
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower().split()
    review = [
        ps.stem(word)
        for word in review
        if word not in set(stopwords.words('english'))
    ]
    corpus.append(' '.join(review))

# ---------------- TF-IDF ----------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# ---------------- MODEL SELECTION ----------------
model_name = st.selectbox(
    "Select Machine Learning Model",
    [
        "Logistic Regression",
        "K-Nearest Neighbors (KNN)",
        "Linear Support Vector Machine",
        "Decision Tree (CART)",
        "Random Forest",
        "Naive Bayes (Multinomial)"
    ]
)

# ---------------- MODEL INITIALIZATION ----------------
if model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_name == "K-Nearest Neighbors (KNN)":
    model = KNeighborsClassifier(n_neighbors=5, metric="cosine")
elif model_name == "Linear Support Vector Machine":
    model = LinearSVC()
elif model_name == "Decision Tree (CART)":
    model = DecisionTreeClassifier(random_state=0)
elif model_name == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=0)
elif model_name == "Naive Bayes (Multinomial)":
    model = MultinomialNB()

# ---------------- TRAIN MODEL ----------------
model.fit(X_train, y_train)

# ---------------- PREDICTION ----------------
y_pred = model.predict(X_test)

# ---------------- METRICS ----------------
accuracy = accuracy_score(y_test, y_pred)
bias = model.score(X_train, y_train)
variance = model.score(X_test, y_test)
cm = confusion_matrix(y_test, y_pred)

# ---------------- DISPLAY RESULTS ----------------
st.subheader("📊 Model Performance")
st.write(f"**Accuracy:** {accuracy:.3f}")
st.write(f"**Bias (Train Accuracy):** {bias:.3f}")
st.write(f"**Variance (Test Accuracy):** {variance:.3f}")
st.subheader("Confusion Matrix")
st.write(cm)
