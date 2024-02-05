
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
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
email_data = pd.read_csv("C:/3-Naive Bayes/SalaryData_test.csv", encoding="ISO-8859-1")

# Data Cleaning
def clean_text(text):
    # Clean text data (example: remove special characters, convert to lowercase)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

email_data['cleaned_education'] = email_data['education'].apply(clean_text)
email_data = email_data.loc[email_data['cleaned_education'] != ""]

#droping Null values
email_data.isnull().sum()#
email_data.fillna(email_data['User ID'].mean())
email_data.fillna(value='missing', inplace=True)
email_data.isnull().sum()#


# Exploratory Data Analysis (EDA)
# Distribution of 'native' variable
plt.figure(figsize=(8, 6))
sns.countplot(x='native', data=email_data)
plt.title('Distribution of Native Language')
plt.xlabel('Native Language')
plt.ylabel('Count')
plt.show()

# Visualizations
plt.figure(figsize=(10, 6))
sns.histplot(email_data['Age'], kde=True)
plt.title('Histogram of Age')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

#Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='User ID', data=email_data)
plt.title('Boxplot of User ID')
plt.xlabel('id')
plt.ylabel('Values')
plt.show()

#Scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(email_data['target'])
plt.title('scatterplot of target')
plt.xlabel('target')
plt.ylabel('Values')
plt.show()


# Feature extraction
X = email_data['cleaned_education']
y = email_data['native']

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert text data into numerical features using CountVectorizer and TfidfTransformer
vectorizer = CountVectorizer(binary=True)
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

# Model Building
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Predictions
predictions = classifier.predict(X_test_tfidf)

# Model evaluation
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
