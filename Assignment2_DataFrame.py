# -*- coding: utf-8 -*-
"""
Created on Tue May 30 23:14:54 2023

@author: Nishant
"""


##########################################################
#ASSIGNMENT 2
import pandas as pd
tech=(
      {
      'Customer':['BU79786','QZ44356',None,'WW63253','HB64268','OC83172','XZ87318','CF85061','DY87989','BQ94931'],
      'State':['Washington','Arizona','Nevada','California','Washington','Oregon','Oregon','Arizona',None,'Oregon'],
      'Response':['No','No',None,'No','No','No','No','No','No','No'],
      'Education':['Bachelor','Bachelor','Bachelor','Bachelor','Bachelor','Bachelor','College','Master','Bachelor','College'],
      'Income':[123.00,343.00,432.00,728.00,None,821.88,923.0,123.98,234.09,456.90]
      }
      )
df=pd.DataFrame(tech)
print(df)
df.dtypes
print(df.dtypes)
####################################


df=pd.DataFrame(tech)
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
df['Customer']
df[['Customer','State']]
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
df=pd.DataFrame(tech,index=row_lable)
df.columns={'C1':'A','C2':'B','C3':'D','C4':'E','C5':'T'}
df2=df.rename({'C1':'A','C2':'B','C3':'D'},axis=1)
df2=df.rename({'C4':'V','C5':'K'},axis='columns')
df2=df.rename(columns={'C1':'A','C2':'B','C3':'D','C4':'E','C5':'T'})
#############################################################

#Drop row
#drop by lable
df1=df.drop([['R0','R2']])
#################################################

#drop by index or position
df1=df.drop(df.index[[1,3]])
##########################################

#Delete row by index range
df1=df.drop(df.index[2:])
#########################################

#when u have default index then 
#how to delete a row
df=pd.DataFrame(tech)
df1=df.drop(0)
#######################################
df=pd.DataFrame(tech)
#############################################
df1=df.drop([1,2])
#############################################
df1=df.drop(range(0,2))
###########################################


###############################################
#drop column
df=pd.DataFrame(tech)
print(df)

#drop column by name
df1=df.drop(['Customer'],axis=1) #drop customer
print(df1)

#drop column by lable explicitly
df2=df.drop(labels=['Customer'],axis=1) #drop customer
print(df)
df2=df.drop(labels=['Response'],axis=1) #drop response
print(df)
#####################################################