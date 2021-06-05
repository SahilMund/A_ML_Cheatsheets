# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values
#using the elbow method to find the number of clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++', max_iter=300, n_init=10,random_state=0)
    #fitting object to dataset
    kmeans.fit(X)
    ##computing cluster sum of squares
    wcss.append(kmeans.inertia_)
#plotting the graph for elbow method
plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('Number Of Clusters')
plt.ylabel('wcss')
plt.show()

#Applying k means to the mall dataset
kmeans=KMeans(n_clusters=5,init='k-means++', max_iter=300, n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(X)
#visualizing the clusters
plt.scatter(X[y_kmeans==0,0], X[y_kmeans==0,1],s=50,c='red',     label='cluster1')
plt.scatter(X[y_kmeans==1,0], X[y_kmeans==1,1],s=50,c='blue',    label='cluster2')
plt.scatter(X[y_kmeans==2,0], X[y_kmeans==2,1],s=50,c='green',   label='cluster3')
plt.scatter(X[y_kmeans==3,0], X[y_kmeans==3,1],s=50,c='cyan',    label='cluster4')
plt.scatter(X[y_kmeans==4,0], X[y_kmeans==4,1],s=50,c='magenta', label='cluster5')

#Plotting thr centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300,c='yellow', label='centriods')
plt.title('Clusters of clients')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()