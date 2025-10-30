# 🩺 Clinical Text Classification using Machine Learning

This project focuses on **classifying clinical transcriptions** into their respective **medical specialties** using Natural Language Processing (NLP) and supervised learning.  
The goal is to automate specialty tagging of medical text — reducing manual categorization time and improving information retrieval for healthcare data.

---

## 🎯 Objectives
- Preprocess and clean raw medical text data.
- Extract meaningful features using **TF-IDF** vectorization.
- Train a classification model to predict the **medical subspecialty**.
- Evaluate using multiple metrics — **Precision, Recall, F1-Score, Accuracy, and Support**.
- Save the trained model for real-time prediction.

---

## 📊 Dataset

**Dataset Name:** `mtsamples.csv`  
**Source:** [MTSamples Clinical Transcription Dataset](https://www.mtsamples.com/)  
**Records:** ~5,000 clinical transcriptions

| Column | Description |
|---------|--------------|
| `#` | Unique ID |
| `description` | Short description of the transcription |
| `medical_specialty` | Target label — medical subspecialty |
| `sample_name` | Type of case (e.g., Discharge Summary, Operative Report) |
| `transcription` | Full clinical note |
| `keywords` | Keywords related to transcription |

> Some transcriptions may represent general notes (not tied to one specialty); these are treated as overlapping or "Other" classes.

---

## 🧹 Data Preprocessing
- Selected key columns: `transcription`, `medical_specialty`, `sample_name`
- Dropped missing or null rows
- Text cleaning:
  - Lowercasing
  - Removing punctuation and extra spaces
  - Stopword removal
  - Lemmatization (optional)
- Combined very low-frequency labels under `"Other"`
- Split data into **70% training** and **30% testing**

---

## 🧠 Feature Extraction
Used **TF-IDF (Term Frequency – Inverse Document Frequency)** to transform textual data into numeric feature vectors.
- Captures importance of terms across documents.
- Removes bias from common medical terms like “patient”, “doctor”, “normal”.

---

## 🤖 Model Training

**Model Used:** `Multinomial Naive Bayes`  
**Reason:** Works best with sparse word frequency data like TF-IDF vectors and gives fast, interpretable results.

**Steps:**
1. Built a `Pipeline` combining `TfidfVectorizer` and `MultinomialNB`.
2. Performed **GridSearchCV** (5-fold CV) for hyperparameter tuning of:
   - `alpha` (smoothing)
   - `ngram_range`
   - `min_df`, `max_df`
3. Selected the best model based on **macro-averaged F1-score**.

---

## 📈 Evaluation Metrics
Evaluated on test data using:
- **Precision**
- **Recall**
- **F1-Score**
- **Support**
- **Macro Average**
- **Weighted Average**
- **Accuracy**

**Sample Output:**
