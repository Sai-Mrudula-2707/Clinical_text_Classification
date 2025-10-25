# Clinical Text Classification


This project focuses on classifying medical texts into different medical subspecialties using machine learning techniques. It involves preprocessing the data, extracting features, and training a classification model to predict the subspecialty of a given medical text.

#  Objectives:

2. Evaluate your model on each subspecialty of medicine
3. Code is modular and bug-free.
4. Use the following evaluation metrics (precision, recall, F1-score, support, macro avg, weighted avg, accuracy). Using more evaluation metrics is encouraged.
5. Spend time commenting and explaining your code.

# Dataset

The dataset used in this project is named mtsamples.csv, which is a CSV file containing the following columns:

- `#`: Transcription ID
- `description`: Description of the transcription
- `medical_specialty`: Subspecialty of medicine (target variable)
- `sample_name`: Name of the sample
- `transcription`: Clinical text transcription
- `keywords`: Keywords associated with the transcription

Note: Some of the transcriptions have labels that do not mention a subspecialty of medicine, but rather a type of note. These notes can be treated as overlaps and do not need to be classified into specific specialties.

-----------------------------------------------------------
# Clinical_3.ipynb File
-----------------------------------------------------------
# Requirements

Python 3.x
pandas
numpy
scikit-learn (sklearn)
Dataset

The project utilizes the mtsamples.csv dataset, which contains information about medical texts, including the transcription, medical specialty, and sample name. The dataset should be placed in the same directory as the project files.

# Setup

1. Install the required dependencies by running the following command:

```
  pip install pandas numpy scikit-learn
```
2. Clone the project repository from GitHub

3. Change to the project directory:
```
   cd medical-text-classification
```
5. Run the script:

# How it works

Data Preprocessing: The script loads the mtsamples.csv dataset and selects the relevant columns (transcription, medical_specialty, sample_name). Any rows with missing values are dropped.

Data Splitting: The dataset is split into training and testing sets using a 70:30 ratio (can be changed). The training set is used for model training and the testing set for evaluating the model's performance.

Feature Extraction: Text features are extracted using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer. The training and testing data are transformed into vectorized representations.

Model Selection and Fine-tuning: The script uses a Multinomial Naive Bayes classifier to classify the medical subspecialties. A grid search is performed to find the best value for the hyperparameter alpha using 5-fold cross-validation.

Model Training and Evaluation: The best model obtained from the grid search is trained on the training data and evaluated on the testing data. The classification report, including precision, recall, and F1-score, is printed for the subspecialty classification using Multinomial Naive Bayes.


# Results

The script prints the classification report for the subspecialty of medicine using Multinomial Naive Bayes. The report includes metrics such as precision, recall, and F1-score for each class.

Example output:
```
Subspecialty of Medicine Classification Report using Multinomial Naive Bayes:
                            precision    recall  f1-score   support

                 Allergy / Immunology       1.00      0.95      0.98        37
                        Bariatrics       1.00      0.96      0.98        27
                      Cardiology         0.98      0.96      0.97       110
...

```



-------------------------
# clinical_2.ipynb file
-------------------------

## Requirements

To run the project, the following requirements should be met:

- Python 3.7 or above
- Required Python libraries (scikit-learn, pandas, numpy, matplotlib, etc.)
- Jupyter Notebook or JupyterLab

## Project Structure

The project consists of the following files:

- `clinical_2.ipynb`: Jupyter Notebook containing the code for data preprocessing, feature extraction, model training, and evaluation.
- `mtsamples.csv`: CSV file containing the sample transcriptions dataset.

## Usage

1. Clone the repository:
  
2. Install the required libraries. You can use pip to install the dependencies:


3. Open the Jupyter Notebook

4. Execute the code cells in the notebook sequentially to preprocess the data, extract features, train the classifiers, and evaluate their performance.

5. Modify and fine-tune the classifiers and parameters according to your requirements. Experiment with different algorithms and techniques to improve the classification accuracy.

## Results

The project provides evaluation metrics such as precision, recall, F1-score, support, macro avg, weighted avg, and accuracy to assess the performance of the classifiers. The classification report and accuracy scores are displayed for each classifier tested.

## License
..



