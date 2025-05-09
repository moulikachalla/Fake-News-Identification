# Fake-News-Identification
This project identifies whether a news article is **real** or **fake** using Natural Language Processing (NLP) and Machine Learning. It covers a full pipeline including data preprocessing, vectorization (Bag of Words & TF-IDF), model building (Random Forest, Naive Bayes), and real-time predictions.

---

## Project Overview

“Fake news” refers to misleading or completely false news articles. With the rise of AI and Natural Language Processing, it's now possible to automatically detect fake news based on article content.

This project demonstrates how to:
- Analyze and preprocess text
- Apply machine learning models
- Classify news as **REAL** or **FAKE**
- Perform real-time predictions for new input articles

---

##  Dataset Info

-  File: `news.csv`
-  Size: **6,335 rows × 4 columns**
- 📌 Columns:
  - `Id`
  - `TitleofThe News`
  - `NewsText`
  - `Prediction` (FAKE/REAL)
-  Source: Data-Flair training (open source)

---

##  Process Breakdown

### 1. Preprocessing
- Lowercasing, removing punctuation, stopwords
- Lemmatization using **NLTK**
- Duplicate removal & newline cleaning
- Creating a clean corpus for modeling

### 2. Feature Extraction
- ✅ **Bag of Words (CountVectorizer)** with `max_features=2000`
- ✅ **TF-IDF Vectorizer** with `max_features=2000`

### 3. Data Splitting
- Split into **train/test** with 75:25 ratio
- Encoded labels: `FAKE → 0`, `REAL → 1`

### 4. Modeling
- ✅ **Naive Bayes (GaussianNB)**
  - Accuracy on BOW: **49.2%**
  - Accuracy on TF-IDF: **86.55%**
- ✅ **Random Forest**
  - Accuracy on BOW: **48.99%**
  - Accuracy on TF-IDF: **88.38%**

✅ **Selected Final Model**: Random Forest on TF-IDF data

---

##  Real-Time Prediction

- User inputs a custom news article
- Preprocessing is applied (same as training)
- Article is vectorized using the trained **TF-IDF**
- Model outputs `REAL` or `FAKE`

---

##  Evaluation Metrics

- ✅ Accuracy Score
- ✅ Confusion Matrix
- ✅ Classification Report (Precision, Recall, F1)

---

## 📈 Visualization

- Word cloud generated after text cleaning
- Vocabulary extracted from vectorizer features
- Head of feature matrices (`data1`, `data2`)

---

##  Model Saving

Saved using **joblib**:
- `Deployment/finalmodel.pkl`: Trained Random Forest
- `Deployment/vectorizer.pkl`: TF-IDF vectorizer

Can be reused for predictions without retraining.

---


## 🧰 Technologies Used

- Python
- pandas, numpy
- scikit-learn
- NLTK
- matplotlib, seaborn
- WordCloud
- joblib


