#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder


# In[16]:


data = pd.read_csv('Iris.csv')
data


# In[17]:


x=data.iloc[:,[1,2,3,4]].values


# In[18]:


kmean = KMeans(n_clusters=3)
y_kmean = kmean.fit_predict(x)
print(y_kmean)

kmean.cluster_centers_


# In[19]:


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


# In[20]:


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


# In[21]:


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


# In[22]:


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


# In[23]:




data1=data[y_kmean==0]#Iris-versicolour
data2=data[y_kmean==1]#Iris-setosa
data3=data[y_kmean==2]#Iris-virginica

plt.scatter(data1.SepalLengthCm,data1.PetalLengthCm,data1['PetalWidthCm'],color= 'red',label="Iris-versicolour")

plt.scatter(data2.SepalLengthCm,data2.PetalLengthCm,data2['PetalWidthCm'],color= 'green',label='Iris-setosa')

plt.scatter(data3.SepalLengthCm,data3.PetalLengthCm,data3['PetalWidthCm'],color= 'Blue',label='Iris-virginica')

plt.scatter(kmean.cluster_centers_[:, 0],kmean.cluster_centers_[:, 2], kmean.cluster_centers_[:,3],  c = 'yellow', label = 'Centroids')

plt.xlabel('PetalLengthCm')
plt.ylabel('PetalWidthCm')
plt.legend()


# In[24]:


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


# In[25]:



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




