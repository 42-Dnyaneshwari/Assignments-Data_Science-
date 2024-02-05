# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:10:31 2024

@author: Nishant
"""

#1.1. Business Objective:
'''
The primary objective is to prepare a classification model using 
the Naive Bayes algorithm for the salary dataset.
'''

#1.2. Constraints:
'''
There are no specific constraints mentioned in the problem statement,
 so we can assume that the main goal is to build an accurate
 classification model without any specific limitations.
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
df = pd.read_csv("C:/3-Naive Bayes/Diasaster_tweets_NB.csv", encoding="ISO-8859-1")

# Data cleaning
df.fillna(value='missing', inplace=True)
# EDA
df.info()
df.describe()
df.isnull().sum()
# Visualizations
plt.figure(figsize=(10, 6))
sns.histplot(df['keyword'], kde=True)
plt.title('Histogram of keyword')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

#Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='id', data=df)
plt.title('Boxplot of id')
plt.xlabel('id')
plt.ylabel('Values')
plt.show()

#Scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(df['target'])
plt.title('scatterplot of target')
plt.xlabel('target')
plt.ylabel('Values')
plt.show()

# Model Building
def preprocess_text(text):
    # Clean text data (example: remove special characters, convert to lowercase)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

df['cleaned_keyword'] = df['keyword'].apply(preprocess_text)

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_keyword'], df['target'], test_size=0.2)

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
