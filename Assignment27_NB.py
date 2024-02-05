# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:43:55 2024

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

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
df=pd.read_csv("C:/3-Naive Bayes/SalaryData_Train.csv",encoding="ISO-8859-1")

import re
def cleaning_education(i):
    w=[]
    i=re.sub("{^A-Za-z""}+"," ",i).lower()
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))

cleaning_education("Hope you are having a good week. Just checking in")

df.education=df.education.apply(cleaning_education)
df=df.loc[df.education!="",:]

from sklearn.model_selection import train_test_split
email_train,email_test=train_test_split(df,test_size=0.2)

def split_into_words(i):
    return [word for word in i.split(" ")]


emails_bow=CountVectorizer(analyzer=split_into_words).fit(df.education)
all_emails_matrix=emails_bow.transform(df.education)

train_emails_matrix=emails_bow.transform(email_train.education)
test_emails_matrix=emails_bow.transform(email_test.education)

tfidf_Transformer=TfidfTransformer().fit(all_emails_matrix)
train_tfidf=tfidf_Transformer.transform(train_emails_matrix)
test_tfidf=tfidf_Transformer.transform(test_emails_matrix)

test_tfidf.shape


from sklearn.naive_bayes import MultinomialNB as MB
classifer_mb=MB()
classifer_mb.fit(train_tfidf,email_train.native)

test_pred_m=classifer_mb.predict(test_tfidf)
accuracy_test_m=np.mean(test_pred_m==email_test.native)
accuracy_test_m
###############################################################



