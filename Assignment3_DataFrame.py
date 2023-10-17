# -*- coding: utf-8 -*-
"""
Created on Tue May 30 23:17:40 2023

@author: Nishant
"""

#Boston Data
#Assignment3
import pandas as pd
df=pd.read_csv('C:/1-Python/boston_data.csv.xls')
print(df)
df.dtypes
#DataFrame Properties
df.shape    #(404,14)
df.size     #5656
df.columns 
df.index
df.dtypes
####################################################

#Accessing one column content
df['crim']
df[['crim','age']]
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
df1=df.drop(['crim'],axis=1) #drop customer
print(df1)

#drop column by lable explicitly
df2=df.drop(labels=['crim'],axis=1) #drop customer
print(df)
df2=df.drop(labels=['rad'],axis=1) #drop response
print(df)
#####################################################
#using inplace=true
df=pd.DataFrame(tech)

df.drop(df.columns[[1]],axis=1,inplace=True)
#################################################
import pandas as pd
#drop 2 or more column by lable name
df=pd.DataFrame(tech)
df2=df.drop(['crim','indus'],axis=1)
print(df2)
############################################


df2=df.drop(df.columns[[0,1]],axis=1)
print(df2)
#############################################

#drop a column from  list of column
df=pd.DataFrame(tech)
lst=['crim','indus']
df2=df.drop(lst,axis=1)
print(df2)
##########################################

#remove column from dataframe inplace
df.drop(df.columns[1],axis=1,inplace=True)
df
################################################

#pandas select rows by index position / lable
#using iloc  
#syntac
#df.iloc[startindex : endindex , startcolumn : endcolumn]
#example
df2=df.iloc[:,0:2] #select all row and 2 i.e.(0,1) column
#this line uses the slicing operator to get DataFrame
df2
##################################################

df2=df.iloc[0:2,:] #selct 2 row i.e.(0,1) and all columns
df2

#both
df2=df.iloc[0:2,0:2]
df2
df2=df.iloc[:,1:3] #select all row and (1,2) column
########################################################

#selct row using index 
df2=df.iloc[2]
df2

df2=df.iloc[[1,2,3]]#select row by index list
df2=df.iloc[1:5] #select row 1,2,3,4
df2=df.iloc[:1] #select row first
df2=df.iloc[:3] #select row 0,1,2
df2=df.iloc[-1:] #select row last
df2=df.iloc[-3:] #select row 3 last
df2=df.iloc[::2] #select alternate row 
##################################################3

#loc
#select row by index lable
df2=df.loc['R2'] #select row 2
df2=df.loc[['R0','R1','R3']] #select row 2,3,4
df2=df.loc['R1':'R3'] #select row 1,2
df2=df.loc['R1':'R5':2] #
df2
###################################################

#pandas select column using names
df2=df['indus']

df2=df[['indus','crim','rad']]

#loc to slice 
#df.loc[:,start:stop:step]
#select multiple columns
df2=df.loc[:,['crim','indus']]

#select random column
df2=df.loc[:,['crim','indus','rad']]

#select column by range
df2=df.loc[:,'crim','rad']

#select column between two columns
df2=df.loc[:,'indus':'rad']

#all the column upto Duration
df2=df.loc[:,'rad':]

#before duration all column will be selected
df2=df.loc[:,:'rad']
############################################

#query all row with courses equal to spark
#pandas dataframe.query()
#equal conditions
df2=df.query("crim=='spark'")
print(df2)

#not equal conditions
df2=df.query("crim != 'spark'")
print(df2)
############################################