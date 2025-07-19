# MACHINE-LEARNING-MODEL-IMPLEMENTATION

COMPANY: CODTECH

NAME: RAUSHAN KUMAR

INTERN ID: CT06DG962

DOMAIN: PYTHON PROGRAMMING

DURATION: 6 WEEKS

MENTOR: NEELA SANTOSH KUMAR



# Spam Detection using Naive Bayes - Machine Learning Model
## Project Overview
This project demonstrates a complete pipeline for building a spam email/SMS classifier using machine learning. The model leverages the Naive Bayes algorithm to classify text messages as either spam or ham (not spam).

Using scikit-learn, pandas, TfidfVectorizer, and other core libraries, we perform end-to-end text classification: from data loading, preprocessing, feature extraction, to model training and evaluation.

This repository is a great starting point for beginners looking to learn natural language processing (NLP) and text classification using real-world datasets.

# Problem Statement
With the overwhelming amount of digital communication today, filtering spam messages has become crucial. This project aims to create a lightweight, fast, and accurate spam detection model that:

Cleans and transforms text data

 Extracts important features using TF-IDF

 Trains and tests a machine learning model

 Predicts whether a new message is spam or not

 Dataset
We use the SMS Spam Collection Dataset, which contains over 5,000 labeled SMS messages.

Columns:

label: Classification as spam or ham (non-spam)

message: The raw SMS content

The dataset is read as:
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
🚀 Features & Steps
📘 1. Import Libraries
We import all essential libraries for data handling, visualization, and machine learning:

pandas, numpy for data manipulation

matplotlib, seaborn for visualization

scikit-learn for model training and evaluation

# 2. Load Dataset
We load and format the dataset using pandas, renaming columns and selecting only relevant data.

# 3. Data Preprocessing
We convert categorical labels into binary values (ham → 0, spam → 1) to make them machine-readable.

df['label'] = df['label'].map({'ham': 0, 'spam': 1})
# 4. Train-Test Split
The dataset is split into 80% training and 20% testing to evaluate the model’s performance accurately.


X_train, X_test, y_train, y_test = train_test_split(...)
# 5. Text Vectorization
We use TF-IDF (Term Frequency-Inverse Document Frequency) to transform the message texts into numerical feature vectors, a crucial step in text classification.

vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
# 6. Model Training
We train a Multinomial Naive Bayes model, which is especially effective for text data due to its probabilistic nature.

model = MultinomialNB()
model.fit(X_train_vec, y_train)
#  7. Model Evaluation
We use several metrics to evaluate model performance:

Accuracy

Classification Report (Precision, Recall, F1-score)

Confusion Matrix

print("Accuracy:", accuracy_score(y_test, y_pred))
# 8. Predicting Custom Messages
We test the trained model on custom user-input messages and classify them as spam or ham.

sample = ["Free entry in 2 a wkly comp...", "Hey, how are you?"]
print(model.predict(vectorizer.transform(sample)))

# How to Run This Project
Clone the repository:

git clone https://github.com/your-username/spam-detector-nb.git
cd spam-detector-nb
Install required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn
Run the Jupyter notebook:

jupyter notebook SpamDetection.ipynb
Make sure the spam.csv file is present in the working directory.
# Results
The model achieves high accuracy and performs well on both training and test data. Sample output includes:



<img width="1621" height="725" alt="image" src="https://github.com/user-attachments/assets/48320385-5846-413e-a9eb-3dbd92da3322" />
