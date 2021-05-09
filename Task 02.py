#!/usr/bin/env python
# coding: utf-8

# # Spark Foundation Data science & Buisness Analyst Task:02
# By Anjali Jha

# # Prediction using Unsupervised ML
# Aim: From the given ‘Iris’ dataset, predict the optimum number of clusters
# and represent it visually.

# # Step I: Importing all the relevant libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder


# # Step II: Importing dataset

# In[2]:


data = pd.read_csv('Iris.csv')
data


# # Selecting Columns

# In[5]:


x=data.iloc[:,[1,2,3,4]].values


# # To find Cluster size

# In[6]:


Error =[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i).fit(x)
    kmeans.fit(x)
    Error.append(kmeans.inertia_)
plt.plot(range(1, 11), Error)
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.show()
print(Error)


# # For cluster size 3 Predicted the Y_Kmean based on ('Sepallength','Sepalwidth','Petallength','Petalwidth')

# In[7]:


kmean = KMeans(n_clusters=3)
y_kmean = kmean.fit_predict(x)
print(y_kmean)

kmean.cluster_centers_


# # Plotting Sepallength, Sepalwidth, cluster_centers

# In[9]:


#SepalLenfth & SepalWidth

data1=data[y_kmean==0]#Iris-versicolour
data2=data[y_kmean==1]#Iris-setosa
data3=data[y_kmean==2]#Iris-virginica

plt.scatter(data1.SepalLengthCm,data1.SepalWidthCm,color= 'red',label="Iris-versicolour")

plt.scatter(data2.SepalLengthCm,data2.SepalWidthCm,color= 'green',label='Iris-setosa')

plt.scatter(data3.SepalLengthCm,data3.SepalWidthCm,color= 'Blue',label='Iris-virginica')

plt.scatter(kmean.cluster_centers_[:, 0], kmean.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')

plt.xlabel('SepalLengthCM')
plt.ylabel('SepalWidthCM')
plt.legend()


# # Plotting Sepalwidth, PetalWidth cluster_centers

# In[10]:


#Sepalwidth & PetalLength

data1=data[y_kmean==0]#Iris-versicolour
data2=data[y_kmean==1]#Iris-setosa
data3=data[y_kmean==2]#Iris-virginica

plt.scatter(data1.SepalWidthCm,data1.PetalLengthCm,color= 'red',label="Iris-versicolour")

plt.scatter(data2.SepalWidthCm,data2.PetalLengthCm,color= 'green',label='Iris-setosa')

plt.scatter(data3.SepalWidthCm,data3.PetalLengthCm,color= 'Blue',label='Iris-virginica')

plt.scatter(kmean.cluster_centers_[:, 1], kmean.cluster_centers_[:,2], s = 100, c = 'yellow', label = 'Centroids')

plt.xlabel('SepalWidthCM')
plt.ylabel('PetalLengthCm')
plt.legend()


# # Plotting PetalLength, PetalWidth, Cluster Center

# In[11]:


#PetalLength & PetalWidth 

data1=data[y_kmean==0]#Iris-versicolour
data2=data[y_kmean==1]#Iris-setosa
data3=data[y_kmean==2]#Iris-virginica

plt.scatter(data1.PetalLengthCm,data1['PetalWidthCm'],color= 'red',label="Iris-versicolour")

plt.scatter(data2.PetalLengthCm,data2['PetalWidthCm'],color= 'green',label='Iris-setosa')

plt.scatter(data3.PetalLengthCm,data3['PetalWidthCm'],color= 'Blue',label='Iris-virginica')

plt.scatter(kmean.cluster_centers_[:, 2], kmean.cluster_centers_[:,3], s = 100, c = 'yellow', label = 'Centroids')

plt.xlabel('PetalLengthCm')
plt.ylabel('PetalWidthCm')
plt.legend()


# # 3d Plotting of SepalLength,PetalLength,PetalWidth and cluster centres

# In[14]:


#3d Plotting of SepalLength,PetalLength,PetalWidth and cluster centres
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

data1=data[y_kmean==0]#Iris-versicolour
data2=data[y_kmean==1]#Iris-setosa
data3=data[y_kmean==2]#Iris-virginica

ax.scatter(data1.SepalLengthCm,data1.PetalLengthCm,data1['PetalWidthCm'],color= 'red',label="Iris-versicolour")

ax.scatter(data2.SepalLengthCm,data2.PetalLengthCm,data2['PetalWidthCm'],color= 'green',label='Iris-setosa')

ax.scatter(data3.SepalLengthCm,data3.PetalLengthCm,data3['PetalWidthCm'],color= 'Blue',label='Iris-virginica')

ax.scatter(kmean.cluster_centers_[:, 0],kmean.cluster_centers_[:, 2], kmean.cluster_centers_[:,3],s=100,  c = 'yellow', label = 'Centroids')

#ax.scatter(X, Y, Z)
# Method 1
ax.set_xlabel('SepalLengthCm')
ax.set_ylabel('PetalLengthCm')
ax.set_zlabel('PetalWidthCm')
plt.show()


# # Plotting SepalWidth,PetalLength,PetalWidth and cluster center

# In[15]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

data1=data[y_kmean==0]#Iris-versicolour
data2=data[y_kmean==1]#Iris-setosa
data3=data[y_kmean==2]#Iris-virginica

ax.scatter(data1.SepalWidthCm,data1.PetalLengthCm,data1['PetalWidthCm'],color= 'red',label="Iris-versicolour")

ax.scatter(data2.SepalWidthCm,data2.PetalLengthCm,data2['PetalWidthCm'],color= 'green',label='Iris-setosa')

ax.scatter(data3.SepalWidthCm,data3.PetalLengthCm,data3['PetalWidthCm'],color= 'Blue',label='Iris-virginica')

ax.scatter(kmean.cluster_centers_[:, 1],kmean.cluster_centers_[:, 2], kmean.cluster_centers_[:,3],s=100,  c = 'yellow', label = 'Centroids')

#ax.scatter(X, Y, Z)
# Method 1
ax.set_xlabel('SepalWidthCm')
ax.set_ylabel('PetalLengthCm')
ax.set_zlabel('PetalWidthCm')
plt.show()


# In[ ]:




