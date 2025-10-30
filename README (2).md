# 🩺 Clinical Text Classification using Machine Learning

This project focuses on **automated classification of medical transcriptions** into their corresponding **medical specialties** using Natural Language Processing (NLP) and supervised learning models.  
The goal is to preprocess raw clinical text, extract meaningful textual features, and train models to accurately classify the specialty.

---

## 🎯 Objectives

- Preprocess and clean raw medical transcriptions.  
- Extract features using **TF-IDF (Term Frequency–Inverse Document Frequency)**.  
- Train a **Multinomial Naive Bayes** classifier and fine-tune hyperparameters using cross-validation.  
- Evaluate performance with multiple metrics — **precision**, **recall**, **F1-score**, **support**, **macro/weighted averages**, and **accuracy**.

---

## 📊 Dataset

**Dataset:** `mtsamples.csv`  
**Source:** Open-source clinical transcription dataset (MTSamples).  
**Size:** ~5,000 medical transcriptions.  

### Columns:
| Column | Description |
|--------|--------------|
| `#` | Transcription ID |
| `description` | Short description of the medical note |
| `medical_specialty` | Target variable – medical subspecialty (e.g., Cardiology, Neurology) |
| `sample_name` | Type of document (e.g., Discharge Summary, Consultation) |
| `transcription` | Complete medical transcription text |
| `keywords` | Keywords related to the document |

> Some entries describe note types (e.g., discharge summary) rather than true specialties — these were treated as overlapping classes.

---

## 🧹 Data Preprocessing

1. **Column Selection:**  
   Selected relevant columns — `transcription`, `medical_specialty`, `sample_name`.  
2. **Missing Values:**  
   Dropped rows missing text or specialty labels.  
3. **Text Cleaning:**  
   - Lowercasing  
   - Removing punctuation and stopwords  
   - Optional lemmatization using `nltk`  
4. **Data Splitting:**  
   70:30 train-test ratio, stratified by medical specialty.  

---

## 🧠 Feature Extraction

- Used **TF-IDF Vectorizer** to convert text into numerical features.  
- Parameters tuned: `max_df`, `min_df`, and `ngram_range`.  
- Captures the importance of words and bigrams while ignoring high-frequency stopwords.

---

## 🤖 Model Training & Fine-Tuning

- **Model Used:** `MultinomialNB` (Multinomial Naive Bayes)  
- **Hyperparameter tuning:** GridSearchCV with 5-fold cross-validation  
- **Parameter tuned:** `alpha` (smoothing factor)  
- **Pipeline:** TF-IDF → MultinomialNB  

```python
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.9, min_df=3, ngram_range=(1,2), stop_words='english')),
    ('clf', MultinomialNB())
])
