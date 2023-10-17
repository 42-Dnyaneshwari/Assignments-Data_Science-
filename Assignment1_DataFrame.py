# -*- coding: utf-8 -*-
"""
Created on Tue May 30 23:16:36 2023

@author: Nishant
"""


#Assignment 1
################################################
#creatating a dict dataframe 
import pandas as pd
Dict={
      'Age':[10,11,12,13,14,15,16,17,18,19],
      'Default':[0,0,0,0,0,0,0,0,0,0],
      'Balance':[2143,29,2,1506,1,231,447,2,121,593],
      'Housing':[1,1,1,0,0,1,1,1,1,1],
      'Loan':[0,0,0,0,1,0,0,0,0,0]
      }
df=pd.DataFrame(Dict)
print(df)
#############################################
column_names=['C1','C2','C3','C4','C5']
row_lable=['R0','R1','R2','R3','R4','R5','R6','R7','R8','R9']
df=pd.DataFrame(Dict,columns=column_names,index=row_lable)
print(df)
#################################################

#DataFrame Properties
df.shape    #(10,5)
df.size     #50
df.columns 
df.index
df.dtypes
####################################################

#Accessing one column content
df['Age']
df[['Age','Default']]
#######################################

#Select certain row and assign it to the another DataFrame
df1=df[5:]
#df1 will select all the records starting from 5 to total size
df2=df[:5]
#df2 will select all the records statring from 0 to 4
##############################################

#describe method
df.describe()
################################################

#Rename()
df=pd.DataFrame(Dict,index=row_lable)
df.columns={'C1':'A','C2':'B','C3':'D','C4':'E','C5':'T'}
df2=df.rename({'C1':'A','C2':'B','C3':'D'},axis=1)
df2=df.rename({'C4':'V','C5':'K'},axis='columns')
df2=df.rename(columns={'C1':'A','C2':'B','C3':'D','C4':'E','C5':'T'})
#############################################################

