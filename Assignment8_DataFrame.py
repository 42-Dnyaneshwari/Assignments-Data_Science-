# -*- coding: utf-8 -*-
"""
Created on Wed May 31 08:56:49 2023


@author: Dnyanehswari..
"""
#ethnic_diversity Data
#ASSIGNMENT 8
###########################################################
import pandas as pd
df=pd.read_csv('C:/1-Python/ethnic_diversity.csv')
print(df)
df.dtypes
#DataFrame  Properties
df.shape    #(768, 9)
df.size     #6912
df.columns 
df.index
df.dtypes
####################################################

#Accessing one column content
df[' age']
df[['EmpID',' age']]
#######################################

#Select certain row and assign it to the another DataFChurn_out_ratee
df1=df[5:]
#df1 will select all the records starting from 5 to total size
df2=df[:5]
#df2 will select all the records statring from 0 to 4
##############################################

#describe method
df.describe()
################################################

#Rename()
#Column
df=pd.read_csv('C:/1-Python/ethnic_diversity.csv')
#renaming column
df.columns={'C1':'C','C2':'D','C3':'A','C4':'B','C5':'P',
            'C6':'E','C7':'T','C8':'V','C9':'F',
            'C10':'W','C11':'N','C12':'H','C13':'L'}
df2=df.rename({'C1':'C','C2':'D','C3':'A','C4':'B',
               'C5':'P','C6':'E','C7':'T','C8':'V',
               'C9':'F','C10':'W','C11':'N',
               'C12':'H','C13':'L'},axis=1)
df2
df2=df.rename({'C4':'VVV','C5':'KKK'},axis='columns')
df2
df2=df.rename(columns={'C1':'A','C2':'B'})
df2
#############################################################
###################################
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
df=pd.read_csv('C:/1-Python/ethnic_diversity.csv')

df1=df.drop(0)
#######################################
df=pd.read_csv('C:/1-Python/ethnic_diversity.csv')

df1=df.drop([1,2])
#############################################
df=pd.read_csv('C:/1-Python/ethnic_diversity.csv')

df1=df.drop(range(0,2))
###########################################


###############################################
#drop column
df=pd.read_csv('C:/1-Python/ethnic_diversity.csv')

print(df)

#drop column by name
df1=df.drop(['age'],axis=1) #drop customer
print(df1)

#drop column by lable explicitly
df2=df.drop(labels=['age'],axis=1) #drop customer
print(df)
df2=df.drop(labels=['age'],axis=1) #drop response
print(df)
#####################################################
#using inplace=true
df=pd.read_csv('C:/1-Python/ethnic_diversity.csv')
df.drop(df.columns[[1]],axis=1,inplace=True)
#################################################
import pandas as pd
#drop 2 or more column by lable name
df2=df.drop(['age'],axis=1)
print(df2)
############################################

#
df2=df.drop(df.columns[[0,1]],axis=1)
print(df2)
#############################################

#drop a column from  list of column
#df=pd.DataFChurn_out_ratee(tech)
lst=['Sex','age']
df2=df.drop(lst,axis=1)
print(df2)
##########################################

#remove column from datafChurn_out_ratee inplace
df.drop(df.columns[1],axis=1,inplace=True)
df
################################################

#pandas select rows by index position / lable
#using iloc  
#syntac
#df.iloc[startindex : endindex , startcolumn : endcolumn]
#example
df2=df.iloc[:,0:2] #select all row and 2 i.e.(0,1) column
#this line uses the slicing operator to get DataFChurn_out_ratee
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
df2=df['age']

df2=df[['Zip','age']]

#loc to slice 
#df.loc[:,start:stop:step]
#select multiple columns
df2=df.loc[:,['','EmpID']]

#select random column
df2=df.loc[:,['Sex','EmpID','age']]

#select column by range
df2=df.loc[:,'Sex','EmpID']

#select column between two columns
df2=df.loc[:,'Sex':'EmpID']

#all the column upto Duration
df2=df.loc[:,'EmpID':]

#before duration all column will be selected
df2=df.loc[:,:' EmpID']
############################################

#query all row with courses equal to spark
#pandas datafChurn_out_ratee.query()
#equal conditions
df2=df.query("Position=='Good'")
print(df2)

#not equal conditions
df2=df.query("Position != 'Good'")
print(df2)

#Adfing column to pandas datafChurn_out_ratee
df=pd.read_csv('C:/1-Python/Diabetes.csv')

print(df)
tutors=['ram','sham','megha','ravi','hell','pani','tani']
df2=df.assign(TutorsAssigned=tutors)
print(df2)
##############################################

#adfing multiple columns to pandas DataFrame 
MNCCompanies=['Tata','Bata','HCL','TCS','Heaven','Google','HELLO']
df2=df.assign(MNCComp=MNCCompanies,TutorsAssigned=tutors)
df2
#####################################################

#derive column from existing column

df2=df.assign(Dis_per=lambda x : x.Salary_hike * x.Salary_hike / 100)
print(df2)
######################################

#append column to an existing datafChurn_out_ratee
df2['MNCC']=MNCCompanies
df2
#########################################################
df=pd.read_csv('C:/1-Python/ethnic_diversity.csv')

#adf column adf specific position

df2.insert(0,'Tutors',tutors)
print(df2)
###################################################


import pandas as pd
import numpy as np
df=pd.read_csv('C:/1-Python/ethnic_diversity.csv')

print(df)
#print(df)
###################################################
print(df.columns)
df.dtypes
df2=df.rename(columns={'Sex':'Sex_List'})
print(df2.columns)
##################################################

#u can also rename() using axis=1 or axis='columns
df2=df.rename({'Sex':'Sex_List'},axis=1)

df2=df.rename({'Sex':'Sex_List'},axis='columns')

print(df2)

#replace existing datafChurn_out_rate with inplace=True which returns None
df2=df.rename({'Sex':'Sex_list'},axis=1,inplace=True)
df.columns

import pandas as pd
import numpy as np
df=pd.read_csv('C:/1-Python/ethnic_diversity.csv')

print(df)
#quick way to get no of row count
row_count=len(df.index) #7
row_count=len(df.axes[0]) #7

#to return no of rows
row_count=df.shape[0] #400
#to return no of columns
col_count=df.shape[1] #11
print("column count is:",col_count)
print("rows count is:",row_count)
#######################################################

#pandas apply() function to columns

import pandas as pd
import numpy as np
df=pd.read_csv('C:/1-Python/ethnic_diversity.csv')

print(df)
#adding 3 in all the columns
def add_3(x):
    return x+3
df2['Sex']=df['Sex'].apply(add_3)
print(df2['Sex'])

#multipy by 3 to columnn B
def mul_3(x):
    return x*3
df2["Sex"]=df["Sex"].apply(mul_3)
print(df2['Sex'])

#apply() on multiple columns i.e. 
#adfing 4 into column A,B.
def add_4(x):
    return x+4
df2[["Sex"," Race"]]=df[["Sex"," Race"]].apply(add_4)
print(df2[["Sex"," Race"]])

# apply() lamda funtion to each column on pandas datafChurn_out_ratee
df2=df.apply(lambda x : x +10)
print(df2)

#transfrom() ,sum()
#same as apply()
def mul_3(x):
    return x*3
df2=df.transform(mul_3)
print(df2)
################################
df2["State"]=df["State"].map(lambda x : x +10)
print(df2["State"])

#using numpy funtion on single column 
#using datafChurn_out_ratee.apply() and [] operator
import numpy as np
df2['State']=df['State'].apply(np.square)
print(df2)
########################################################

#USING NUMPY.SQUARE FUNTION
df['State']=np.square(df['State'])
print(df['State'])
##################################

#using pandas groupby() 
import pandas as pd
import numpy as np
df=pd.read_csv('C:/1-Python/ethnic_diversity.csv')

print(df)


#groupby()
#single funtion
df2=df.groupby(['State']).sum()
print(df2)
######################################################
#for multiple columns
df2=df.groupby(['State','age']).sum()
print(df2)
##############################################

#adf an index to datafChurn_out_ratee
#reset_index()
#group bu single columns
df2=df.groupby(['State',' Race']).sum().reset_index()
print(df2)
#####################################################

#group by multiple columns
df2=df.groupby(['State',' Race']).sum()
print(df2)
#######################################]

#pandas datafChurn_out_ratee to get columns name
import pandas as pd
import numpy as np
df=pd.read_csv('C:/1-Python/ethnic_diversity.csv')

print(df)
df.columns
#get the name of column from headers
column_headers=list(df.columns.values)
print("the column header is:",column_headers)
#using list(df.columns)
column_headers=list(df.columns)
print("the column header is:",column_headers)
#using list(df)
column_headers=list(df)
print("the column header is:",column_headers)
#####################################################

#wenever there are less accuracy we go for shuffling 
#because after that we will get accurate answer
#shuffling of rows in pandas DataFChurn_out_ratee
#SHUFFLE ALL ROWS AND RETURN ALL ROWS
df1=df.sample( frac=1)
print(df1)
###################################################### 


