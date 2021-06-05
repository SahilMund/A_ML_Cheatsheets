# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values
#plotting the dendogranm to find the optimal no. of clusters
import scipy.cluster.hierarchy as sch
dendrogram= sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('customers')
plt.ylabel('Euclidian Distance')
plt.show()

#Fit HC to mall dataset
from sklearn.cluster import AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc=hc.fit_predict(X)
#Visualizing the clusters
plt.scatter(X[y_hc==0,0], X[y_hc==0,1],s=50,c='red',     label='cluster1')
plt.scatter(X[y_hc==1,0], X[y_hc==1,1],s=50,c='blue',    label='cluster2')
plt.scatter(X[y_hc==2,0], X[y_hc==2,1],s=50,c='green',   label='cluster3')
plt.scatter(X[y_hc==3,0], X[y_hc==3,1],s=50,c='cyan',    label='cluster4')
plt.scatter(X[y_hc==4,0], X[y_hc==4,1],s=50,c='magenta', label='cluster5')
plt.title('Clusters of clients')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()