# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 21:04:05 2024

@author: Nishant
"""
'''
Business Objective:
The primary objective is to use a Naïve Bayes model to predict whether a given tweet about a real disaster is real (1) or fake (0). This prediction can be valuable for quickly identifying and responding to real disasters based on social media activity.

Constraints:

Data Quality: The accuracy of the model heavily relies on the quality and relevance of the training data. Ensuring the dataset contains reliable information about real and fake disaster tweets is crucial.
Model Interpretability: If stakeholders require interpretability of the model's predictions, it might restrict the choice of complex algorithms or features.
Data Privacy: Ensuring compliance with privacy regulations and ethical guidelines when using social media data for analysis.
Model Deployment: Considering the feasibility and practicality of deploying the model in real-time or near real-time scenarios for timely response to disaster situations.
'''
# Importing necessary libraries
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("C:/3-Naive Bayes/NB_Car_AD.csv", encoding="ISO-8859-1")

# Data cleaning
df.fillna(value='missing', inplace=True)

# EDA
df.info()
df.describe()

# Visualizations
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True)
plt.title('Histogram of gender')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

#Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Age', data=df)
plt.title('Boxplot of Age')
plt.xlabel('Age')
plt.ylabel('Values')
plt.show()

#Scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(df['User ID'])
plt.title('scatterplot of User ID')
plt.xlabel('User ID')
plt.ylabel('Values')
plt.show()

# Model Building
def preprocess_text(text):
    # Clean text data (example: remove special characters, convert to lowercase)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

df['cleaned_keyword'] = df['Gender'].apply(preprocess_text)

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_keyword'], df['Gender'], test_size=0.2)

# Feature extraction
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Training the Naive Bayes model
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Predicting on test data
X_test_counts = vectorizer.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
predictions = classifier.predict(X_test_tfidf)

# Model evaluation
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
