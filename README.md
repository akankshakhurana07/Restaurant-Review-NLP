# 🍽️ Restaurant Review Classification – NLP

<div align="center">

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit_App-8B5CF6?style=for-the-badge&labelColor=1E293B&logo=streamlit&logoColor=white)](https://restaurant-review-nlp-dgjvu3bpy5ebrg72qfjyvw.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11+-3572A5?style=for-the-badge&logo=python&logoColor=white&labelColor=1E293B)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white&labelColor=1E293B)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-1.6-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white&labelColor=1E293B)](https://scikit-learn.org/)

</div>

<br>

> **A fully interactive Streamlit dashboard that classifies restaurant reviews as Positive or Negative using NLP preprocessing (TF-IDF + Stemming) and lets you compare 6 different ML models in real time.**

---

## 📺 Live App

👉 **[Click here to open the app](https://restaurant-review-nlp-dgjvu3bpy5ebrg72qfjyvw.streamlit.app/)**

---

## 🎯 Features

| Feature | Description |
|---|---|
| 📂 **File Upload** | Upload your own `.tsv` dataset directly from the browser — no local setup needed |
| ✏️ **NLP Preprocessing** | Regex cleaning → Lowercasing → Stopword removal → Porter Stemming |
| 📊 **TF-IDF Vectorization** | Converts cleaned text into numerical feature vectors |
| 🤖 **6 ML Models** | Train and evaluate 6 classifiers with a single click |
| 📈 **Metrics Dashboard** | Accuracy, Precision, Recall, F1 Score, Bias–Variance gap — all in one view |
| 🟦 **Confusion Matrix** | Interactive Plotly heatmap — hover to explore |
| 🌙 **Dark UI** | Fully custom dark-themed interface with gradient accents |

---

## 🤖 Models Supported

| # | Model | Description |
|---|---|---|
| 1 | **Logistic Regression** | Fast linear classifier, great baseline |
| 2 | **K-Nearest Neighbors (KNN)** | Instance-based, uses cosine similarity |
| 3 | **Linear SVM** | Finds the best separating hyperplane |
| 4 | **Decision Tree (CART)** | Interpretable tree-based model |
| 5 | **Random Forest** | Ensemble of 100 decision trees |
| 6 | **Naive Bayes (Multinomial)** | Probabilistic, fast, works well on text |

---

## 📁 Project Structure

```
Restaurant-Review-NLP/
│
├── app.py                   # Main Streamlit application
├── requirements.txt         # Python dependencies
├── Restaurant_Reviews.tsv   # Dataset (1000 reviews)
└── README.md                # This file
```

---

## 📦 Dataset Format

The app expects a **TSV (Tab-Separated Values)** file with two columns:

| Review | Liked |
|---|---|
| Wow... Couldn't believe how amazing the food was here. | 1 |
| Crap... This place is horrible, never coming back. | 0 |

- **Review** → Raw text of the restaurant review
- **Liked** → `1` (Positive) or `0` (Negative)

> 💡 Default dataset has **1000 rows**. The app uses the first 1000 rows automatically.

---

## 🚀 How to Run Locally

### 1️⃣ Clone the repo

```bash
git clone https://github.com/akanksha-hurana07/Restaurant-Review-NLP.git
cd Restaurant-Review-NLP
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the app

```bash
streamlit run app.py
```

> The app will open automatically at **http://localhost:8501**

---

## 📊 How It Works (Pipeline)

```
Raw Text Reviews
       │
       ▼
┌─────────────────┐
│  1. Clean Text  │  → Remove special characters (regex)
└────────┬────────┘
         ▼
┌─────────────────┐
│  2. Lowercase   │  → Convert everything to lowercase
└────────┬────────┘
         ▼
┌─────────────────┐
│  3. Stopwords   │  → Remove common English words (the, is, at…)
└────────┬────────┘
         ▼
┌─────────────────┐
│  4. Stemming    │  → Reduce words to root form (running → run)
└────────┬────────┘
         ▼
┌─────────────────┐
│  5. TF-IDF      │  → Convert text to numerical vectors
└────────┬────────┘
         ▼
┌─────────────────┐
│  6. Train/Test  │  → 80% train, 20% test split
└────────┬────────┘
         ▼
┌─────────────────┐
│  7. Model Train │  → Train selected ML model
└────────┬────────┘
         ▼
┌─────────────────┐
│  8. Evaluate    │  → Accuracy, Precision, Recall, F1, Confusion Matrix
└─────────────────┘
```

---

## 📈 Sample Results

| Model | Test Accuracy |
|---|---|
| Logistic Regression | ~78% |
| KNN (k=5) | ~72% |
| Linear SVM | ~77% |
| Decision Tree | ~70% |
| Random Forest | ~74% |
| Naive Bayes | ~76% |

> ⚠️ Results may vary slightly based on train/test split.

---

## 🛠️ Tech Stack

| Technology | Use |
|---|---|
| **Python 3.11+** | Core language |
| **Streamlit** | Web app framework |
| **NLTK** | Stopwords + Porter Stemmer |
| **Scikit-Learn** | ML models + TF-IDF + Metrics |
| **Plotly** | Interactive confusion matrix heatmap |
| **Pandas / NumPy** | Data handling |

---

## 🔧 Deployed On

| Platform | Link |
|---|---|
| **Streamlit Cloud** | [restaurant-review-nlp-dgjvu3bpy5ebrg72qfjyvw.streamlit.app](https://restaurant-review-nlp-dgjvu3bpy5ebrg72qfjyvw.streamlit.app/) |
| **GitHub Repo** | [github.com/akanksha-hurana07/Restaurant-Review-NLP](https://github.com/) |

---

## 💡 Future Improvements

- [ ] Add **BERT / Transformer-based** model comparison
- [ ] Add **cross-validation** (k-fold) for more reliable scores
- [ ] Allow user to input a **custom review** and get a live prediction
- [ ] Add **word cloud** visualization of top positive/negative words
- [ ] Export results as **PDF report**

---

## 📝 License

This project is open source and available under the **MIT License**.

---

<div align="center">

Made with ❤️ &nbsp; | &nbsp; Powered by Streamlit & Scikit-Learn

</div>
