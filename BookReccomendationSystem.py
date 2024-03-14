#!/usr/bin/env python
# coding: utf-8

# # Project Title : Book Recommendation

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


books = pd.read_csv("Books.csv", encoding="latin-1", error_bad_lines=False)
users = pd.read_csv("Users.csv",encoding="latin-1", error_bad_lines=False)
ratings = pd.read_csv("Ratings.csv", encoding="latin-1", error_bad_lines=False)


# #### Books data

# In[7]:


books.head(5)


# In[8]:


print("No. of observations:{}\nNo. of parameters:{}".format(books.shape[0],books.shape[1]))


# In[9]:


books.isnull().sum()


# In[10]:


books.dtypes


# In[11]:


books.duplicated().sum()


# In[12]:


books['Year-Of-Publication'].unique()


# In[13]:


#since year data has some object it it, we shall convert it into null data
books['Year-Of-Publication'] = pd.to_numeric(books['Year-Of-Publication'],errors='coerce')
books['Year-Of-Publication'].isna().sum()


# In[14]:


#since year data has the year 0 and 2022 which is invalid, we shall convert it into null data
books.loc[(books['Year-Of-Publication'] > 2022) | (books['Year-Of-Publication'] == 0), 'year'] = np.NAN
#Replacing null data with median 
books['Year-Of-Publication'].fillna(books['year'].median() , inplace = True)
books['Year-Of-Publication'].isna().sum()


# In[15]:


#Finding and replacing null data from publisher
books.loc[books['Publisher'].isna()]


# In[16]:


#Replacing null data from publisher
books['Publisher'].fillna('other' , inplace = True)
books['Publisher'].isna().sum()


# In[17]:


#Finding and replacing null data from author
books.loc[books['Book-Author'].isna()]


# In[18]:


#replacing null data from publisher
books['Book-Author'].fillna("Unknown" , inplace = True)
books['Book-Author'].isna().sum()


# #### User data

# In[19]:


users.head(5)


# In[20]:


users.isna().sum()


# In[21]:


sns.set(style="whitegrid")
sns.boxplot(x='Age',data=users)


# In[22]:


print(sorted(users['Age'].unique()))


# In[23]:


#removing age above 100 and below 5 
users.loc[(users['Age'] > 100) | (users['Age'] < 5) , 'Age' ] = np.NAN


# In[24]:


users['Age'].isna().sum()


# In[25]:


#Filling the null values with mean
users['Age'].fillna(users['Age'].mean(), inplace = True)


# In[26]:


users.duplicated().sum()


# In[27]:


#Next, can we expand the 'Location' field to break it up into 'City', 'State', and 'Country'.
user_location_expanded = users.Location.str.split(',', 2, expand=True)
user_location_expanded.columns = ['city', 'state', 'country']
users = users.join(user_location_expanded)


# In[28]:


users.drop(columns=['Location'], inplace=True)
users.head()


# #### Rating data

# In[29]:


ratings.head(5)


# In[30]:


ratings.shape


# In[31]:


ratings.isna().sum()


# In[32]:


ratings.duplicated().sum()


# In[33]:


ratings.loc[ratings['Book-Rating'] == 0]


# In[34]:


ratings['Book-Rating'].hist(bins=10)


# In[35]:


publications = {}
for year in books['Year-Of-Publication']:
    if str(year) not in publications:
        publications[str(year)] = 0
    publications[str(year)] +=1

publications = {k:v for k, v in sorted(publications.items())}

fig = plt.figure(figsize =(50, 20))
plt.bar(list(publications.keys()),list(publications.values()), color = 'blue')
plt.ylabel("Number of books published")
plt.xlabel("Year of Publication")
plt.title("Number of books published by yearly")
plt.xticks(rotation=45)
plt.margins(x = 0)
plt.show()


# In[36]:


plt.figure(figsize=(10,8))
sns.countplot(x="Book-Rating", data=ratings)


# In[37]:


top_cities = users.city.value_counts()[:10]
#print(f'The 10 cities with the most users are:\n{top_cities}')
plt.figure(figsize=(15,7))
sns.barplot(y=top_cities.index,x=top_cities.values)
plt.title('City-wise Count of Users')


# In[38]:


top_countries = users.country.value_counts()[:10]
#print(f'The 10 countries with the most users are:\n{top_countries}')
plt.figure(figsize=(15,7))
sns.barplot(y=top_countries.index,x=top_countries.values)
plt.title('Country-wise Count of Users')


# In[39]:


# Explicit Ratings
plt.figure(figsize=(12,10))
data = ratings[ratings['Book-Rating'] != 0]
sns.countplot(x="Book-Rating", data=data)
plt.title("Explicit Ratings")


# In[40]:


#number of books published by an author (top-20)
plt.figure(figsize=(16,8))
sns.countplot(y="Book-Author", data=books,order=books['Book-Author'].value_counts().index[0:20])
plt.title("Number of books by an author (Top 20)")


# In[41]:


# number of books published by publisher (top 20)
plt.figure(figsize=(16,7))
sns.countplot(y="Publisher", data=books,order=books['Publisher'].value_counts().index[0:20])
plt.title("Number of books published by a publisher (Top 20)")


# In[42]:


# Plotting of ratings 
plt.figure(figsize=(15,8))
sns.countplot(y="Book-Title", data=books, order=books['Book-Title'].value_counts().index[0:10])
plt.title("Number of Ratings for a book (Top 10)")


# ## Simple Popularity based Recommendation System

# In[43]:


ratings_count = ratings.groupby(by=['ISBN'])['Book-Rating'].sum()
ratings_count = pd.DataFrame(ratings_count)
top10 = ratings_count.sort_values('Book-Rating' , ascending=False).head(10)
print("The following books are recommended")
top10.merge(books , left_index=True , right_on= 'ISBN')


# # Recommendation using KNN

# In[44]:


# merging datasets
merged_data = pd.merge(books, ratings, on='ISBN', how='inner')
merged_data = pd.merge(merged_data, users, on='User-ID', how='inner')

data1 = (merged_data.groupby(by = ['Book-Title'])['Book-Rating'].count().reset_index().rename(columns = {'Book-Rating': 'Total-Rating'}))


# In[45]:


data1


# In[46]:


print(sorted(data1['Total-Rating'].unique()))


# In[47]:


data.head(5)


# In[48]:


data = pd.merge(data1, merged_data, on='Book-Title', left_index = False)
data = data[data['Total-Rating'] >= 50]
data = data.reset_index(drop = True)

#building a matrix
from scipy.sparse import csr_matrix
df = data.pivot_table(index = 'Book-Title', columns = 'User-ID', values = 'Book-Rating').fillna(0)
matrix = csr_matrix(df)


# In[49]:


from sklearn.neighbors import NearestNeighbors
book_name = input("Enter a book name: ")
model = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model.fit(matrix)

distances, indices = model.kneighbors(df.loc[book_name].values.reshape(1, -1), n_neighbors =10)
print("\nRecommended books:\n")
for i in range(0, len(distances.flatten())):
    if i > 0:
        print(df.index[indices.flatten()[i]]) 


# In[50]:


df1 = data[data['Book-Title']=="Charlie and the Chocolate Factory"]
#getting the details who read Charlie and the Chocolate Factory
df1


# In[51]:


df2 = data[data['Book-Title']=="Matilda"]#getting the details who read Matilda
df2


# In[52]:


x=set(df1["User-ID"].values)#contain the userid of those who read Charlie and the chocolate factory
y=set(df2["User-ID"].values)#contain the userid of those who read matilda


# In[53]:


similar_users = x & y
print(similar_users)


# In[54]:


data[data["User-ID"]==74689].sort_values('Book-Rating', ascending=False).head(5)


# In[55]:


data[data["User-ID"]==251394].sort_values('Book-Rating', ascending=False).head(5)


# In[56]:


data[data["User-ID"]==115490].sort_values('Book-Rating', ascending=False).head(5)


# In[57]:


data[data["User-ID"]==249628].sort_values('Book-Rating', ascending=False).head(5)


# In[58]:


data[data["User-ID"]==174791].sort_values('Book-Rating', ascending=False).head(5)


# #taking similiar user-id (who has read and given the book good rating)(74689, 251394, 115490, 249628, 174791) the following are the books

# In[59]:


list = ('Stormy Weather','Charlie and the Chocolate Factory' , 'The Fellowship of the Ring (The Lord of the Rings, Part 1)' ,'A Wizard of Earthsea (Earthsea Trilogy, Book 1)','Reading Lolita in Tehran: A Memoir in Books' , 'Night' , 'Harriet the Spy' , 'The Scottish Bride (Bride Trilogy (Paperback))','The Two Towers (The Lord of the Rings, Part 2)' , 'The Secret Garden')


# In[60]:


#Top 10 books
list


# In[61]:


import pickle


# In[62]:


pickle.dump(df,open("df.pkl","wb"))


# In[63]:


pickle.dump(model,open("model.pkl","wb"))


# In[64]:


pickle.dump(data,open("data.pkl","wb"))


# In[ ]:





# In[ ]:




