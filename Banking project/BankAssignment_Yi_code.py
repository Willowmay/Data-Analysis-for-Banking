
# coding: utf-8

# ## Assignment 2 - please answer the questions as thoroughly as possible, and provide detailed interpretation whenever possible

# In[1]:


import pandas as pd
df=pd.read_csv("banklist.csv")
df.head()


# ## Question 1 (20 points)

# In[ ]:


#Look at the data set: First, make sure there are no missing values in
#any of the categorical columns.If there are, replace them with
#a 'M' value. Then, consider that Acquiring Institution column is the institution that holds various banks.
#What are the top 4 acquiring institutions?$


# In[2]:


print(len(df))


# In[3]:


df.info()


# In[4]:


#Looking for NaN
na = df.isnull()
na


# In[5]:


na.sum() # There is no missing values in the dataset


# In[6]:


mask = df.duplicated(keep=False)
len(mask) # Check if there any duplicated rows


# In[7]:


len(df) # The results show there is no duplicated rows


# In[11]:


pd.options.display.max_rows=None #Show all rows


# In[12]:


a=df['Acquiring Institution'].value_counts()
a #Counting Values for Acquiring Institution Column


# In[13]:


a.sort_index() #Sorted alphabetically by Acquiring Institution 
#There are some institutions different in Name, but stand for the same Acquiring Institution
#For example, U.S. Bank N.A. and U.S. Bank, N.A.;Stearns Bank N.A. and Stearns Bank, N.A.;CharterBank and Charter Bank


# In[20]:


#Combine the same Acquiring Institution
df["Acquiring Institution"]= df["Acquiring Institution"].replace("CharterBank", "Charter Bank")
df["Acquiring Institution"]= df["Acquiring Institution"].replace("Stearns Bank N.A.", "Stearns Bank, N.A.")
df["Acquiring Institution"]= df["Acquiring Institution"].replace("U.S. Bank N.A.", "U.S. Bank, N.A.")


# In[21]:


newseq_df=df.sort_values('Acquiring Institution')
newseq_df # sorting the result in Acquiring Institution column


# In[22]:


pd.options.display.max_rows=60 #Show maximun 60 rows


# In[23]:


a=df['Acquiring Institution'].value_counts()
a.sort_values(ascending=False).iloc[0:4] # Top 4 acquiring institutions


# In[24]:


#Resorting the Top 4, since the Top1 Acquirer shows No Acquier as the result
greatAI=a.sort_values(ascending=False).iloc[1:5]
greatAI
# Resulut is below:
# U.S. Bank, N.A.                        14
# State Bank and Trust Company           12
# First-Citizens Bank & Trust Company    11
# Ameris Bank                            10


# In[ ]:


#How many city names start with the letter 'A' and how many don't? Use the str vectorized series attribute


# In[25]:


acity=df['City'].str.startswith('A').sum() 
print(acity)# Number of city names start with the letter 'A'


# In[26]:


len(df['City'])-acity # Number of city names don't start with the letter 'A'


# In[ ]:


#What are the cities whose names consist of 2 words and how many are there?


# In[27]:


#Count number of words in City column and add a new column Word_Count
print ("Adding a new column Word_Count using the existing columns in DataFrame:")
df['Word_Count'] = df['City'].str.split(' ').str.len()
df


# In[28]:


Twocities=df.loc[df['Word_Count'] == 2] # Select out 2 number words in City column
Twocities 


# In[29]:


len(Twocities) # There are 130 cities whose name consist of 2 words


# ## Question 2 (50 points)

# In[136]:


#use various apply and group by methods studied in class to obtain 
#the city for each state corresponding to the smallest number of certificates (CERT column). 
#Then, display only the city, state combinations along with the number of certificates
#(data frame with 3 columns) for which the Acquiring Institution 
#contains 3 words (for this exercise, you can count symbols
#like & and others as words for each Acquiring Institution)


# In[30]:


## Method 1 Using Group By
# Group by State: for all years capture the city for each state corresponding to the smallest number of certificates (CERT column).
grouped=df.groupby(['ST'],as_index=False)['CERT'].min()


# In[31]:


result = pd.merge(grouped,df, on='CERT', how='inner') # Using merge method
result['City'] # Obtain the City


# In[32]:


## Method 2 Using Group By
# Group by State: for all years capture the city for each state corresponding to the smallest number of certificates (CERT column).
new1=df.loc[df.reset_index().groupby(['ST'])['CERT'].idxmin()]


# In[35]:


#Count number of words in Acquiring Institution column and add a new column Ac_Word_Count
print ("Adding a new column Ac_Word_Count using the existing columns in DataFrame:")
new1['Ac_Word_Count'] = new1['Acquiring Institution'].str.split(' ').str.len()
new1


# In[36]:


#Display only the city, state combinations along with the number of certificates
#(data frame with 3 columns) for which the Acquiring Institution contains 3 words
Thwords1=new1.loc[new1['Ac_Word_Count'] == 3]
Thwords1[['City','ST','CERT']]


# In[38]:


## Method 3 Using apply method
idx = new1['Acquiring Institution'].apply(lambda x: len(x.split(' ')) == 3)
new1.loc[idx, ['City','ST','CERT']]


# In[243]:


## Method 4 using groupby and apply together
# Building up a function to find minimum number
def filter_group(x, col):
    return x[x[col] == x[col].min()]
df2 = df.groupby('ST',group_keys=False).apply(lambda x: filter_group(x,'CERT'))
# Find out 3 numbers in Acquiring Institution column
df2 = df2.loc[df2['Acquiring Institution'].apply(lambda x: len(x.split(' '))) == 3]
df2[['City','ST','CERT']] # Select City, ST and CERT three columns as the result


# ## Question 3 (30 points)

# In[ ]:


#finally, let us obtain a distribution of CERT column across all records. Then, let us subset 
#the original data frame to only consider records for which CERT is smaller than 10th percentile of the total. 
#Then, for these records, use pivot table
#method to obtain the min, max, and average CERT 
#for each ST combination for which 
#the number of records is more than 1. We will have index for 
#state, and then 4 columns: ST,min_CERT, max_CERT, mean_CERT for CERT (make sure
#you do not have a multi-level data frame! 
#Finally, display a bar plot for the first 3 states in the resulting
#data frame: each of the 3 states will be on x axis, 
#and the corresponding 3 summary statistics will be the 3 bars - stacked bar plot.


# In[39]:


#Deustribution of CERT column
distri=df['CERT'].describe(percentiles=[.1])
distri


# In[40]:


# Subset by CERT is smaller than 10th percentile of the total
per=df[df['CERT'] < df['CERT'].quantile(.10)]
per.drop(['Word_Count'], axis=1) # Drop Word_Count column from last question


# In[41]:


#Use pivot table method to obtain min, max, and average CERT
import numpy as np
table=pd.pivot_table(per,index=["ST"],values=["CERT"],
               aggfunc={"CERT":[np.min, np.max, np.mean,len]},fill_value=0)
table=table["CERT"]
table


# In[42]:


draw=table[table['len']>1][0:3] # Select first 3 states with each 
#ST combination for which the number of records is more than 1
draw


# In[43]:


#Index for state, and then 4 columns: 
#ST,min_CERT, max_CERT, mean_CERT for CERT 
data = {'ST':['FL','GA','IL'],
        'min_CERT': [5672, 151, 916], 
        'max_CERT': [9619, 10054, 10086],
        'mean_CERT': [7645.500000, 4625.857143, 5583.750000]}
draw1 = pd.DataFrame(data, index = ['FL','GA','IL'])
draw1


# In[204]:


#plotting result
import matplotlib.pyplot as plt
draw1.plot(kind='bar',rot=True)

