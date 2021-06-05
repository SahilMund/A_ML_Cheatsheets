import numpy as np,pandas as pd,matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data=pd.read_csv('C:/Users/Sahil/Desktop/xclara.csv')
print(data.head(5))  #here we print only 5 values from our xclara data

k=4     # number of cluster

'''choosing a model'''
kmean=KMeans(n_clusters=k)      #here n_cluster identifies how many group we want to form i.e. here 2

 #train a model
kmean=kmean.fit(data) # as in data we only have input dataso no need to go for splitting of datas
labels=kmean.labels_       #labels is the array that contains all the cluster numbers or labels

 #centroids are the points which are used to group all the datas or the points that clusterd data into different groups

centroids=kmean.cluster_centers_      #here 2 different group form and the data will be clusterd accordingly

''' testing data '''
x_test=[[4.6,67],[2.88,60],[4.65,98],[30.4,56],[-1.33,5.6],[45.555,-1.22]]
prediction=kmean.predict(x_test)
print(prediction)

'''for same values we got 0 '''

#visualisation
colors=['blue','red','green','black']
y=0
for x in labels:            #according to labels values i was gonna starta a for loop and according to labels we plot the graph
    plt.scatter(data.iloc[y,0],data.iloc[y,1],color=colors[x])
    y+=1
'''iloc is for go to that index location and [y,0] for v1 and y index starts from 0 to 3000 and for [y,1] it goes to v2 and and also it goes upto 0 to 3000
& colors give the scsattered data different colors'''

for x in range(k):     #loop will run 2 times as k=2
    lines=plt.plot(centroids[x,0],centroids[x,1],'kx')
    #plot and categorised into 2 different groups so [x,0]and [x,1] for 2 grp 2 indexing and 'kx' is for crosspoint of the centorid & also categorise them into 2 categories
    plt.setp(lines,ms=15.0) # for a large centroid we use plt.setp
    plt.setp(lines,mew=2.0)  #different size of 2 groups (here if we are write 'ms' instead of 'mew' then the cross is very small)
title=('number of clusters (k)={} ').format(k)
plt.title('clustering')
plt.xlabel('v1')
plt.ylabel('v2')
plt.show()
