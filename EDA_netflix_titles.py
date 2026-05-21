import pandas as pd # Data Manipulation
import numpy as np # mathematical operation
import seaborn as sns # visualization
import matplotlib.pyplot as plt # visualization

# Load the Data set
df = pd.read_csv('/content/netflix_titles.csv')

df.info()

df.isnull().sum() # check missing values
print(df.isnull().sum())

df.duplicated().sum() # to find duplicate values
print(df.duplicated().sum())
df.drop_duplicates(inplace = True) # to drop duplicate values

# missing value treatment for "object " variable
df['director'] = df['director'].fillna(df['director'].mode()[0])
df['cast'] = df['cast'].fillna(df['cast'].mode()[0])
df['country'] = df['country'].fillna(df['country'].mode()[0])
df['date_added'] =df['date_added'].fillna(df['date_added'].mode()[0])
df['duration'] = df['duration'].fillna(df['duration'].mode()[0])
df['rating'] = df['rating'].fillna(df['rating'].mode()[0])

df.isnull().sum() # check missing values
print(df.isnull().sum())

df.describe().T

# outliers
plt.figure(figsize = (10,5))
sns.boxplot(df['release_year'],color ='red')
plt.title("Outlier Check : Release Year")
plt.xlabel('Year')
plt.show()

# capping the Outliers
floor = df['release_year'].quantile(0.05)
df['release_year'] = df['release_year'].clip(lower =floor)

# log transformation methiod to reduce impact of outliers
df['release_year'] = np.log(df['release_year'])

# outliers after treatment
plt.figure(figsize = (10,5))
sns.boxplot(df['release_year'],color ='red')
plt.title("Outlier Check : Release Year")
plt.xlabel('Year')
plt.show()

df.shape

df.dtypes

from sklearn.preprocessing import LabelEncoder
import numpy as np
from numpy import astype

# convert all the 'obj' variables into 'numerical' variables

# 1.Identify all'object' columns

object_cols = df.select_dtypes(include = ['object']).columns

# 2.Intialize the LabelEncoder
le = LabelEncoder()

# 3. Loop through columns and convert to int
for col in object_cols:
    df[col] = le .fit_transform(df[col].astype(str))

# Handle any float values
df = df.astype(int)

#check the results
print(df.dtypes)

df.corr()

import seaborn as sns
import matplotlib.pyplot as plt

# Increase fig size
plt.figure(figsize=(20,15))

# 2.Focus on the most important coorelations
corr = df.corr()

# 3.Create a mask to hide the top half
mask = np.triu(np.ones_like(corr))

# 4.Plot the heatmap
sns.heatmap(corr, mask = mask, annot = True, fmt ='.2f' , cmap = 'coolwarm', annot_kws = {"size": 8}, cbar_kws = {"shrink":8})
plt.title('correlation Heatmap',fontsize = 20)
plt.xticks(rotation =45, ha = 'right')
plt.show()

corr = df.corr()

# Plot the Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr,annot=True,  cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Bi variate , univariate , Multivariable analysis

# prepare the data
top_ratings = df['rating'].value_counts().index[:10]
top_ratings_data = df[df['rating'].isin(top_ratings)]

# create the plot
plt.figure(figsize = (10,8))
sns.countplot(data = top_ratings_data, x= 'rating', hue = 'type', palette= 'magma',order= top_ratings)


#Aesthitics
plt.title('Top 10 Ratings : Movies vs TV Shows', fontsize=19)
plt.xlabel('Rating',fontsize = 12)
plt.ylabel('Count',fontsize = 12)
plt.legend(title = 'content type')
plt.xticks(rotation = 45, ha = 'right')
plt.show()