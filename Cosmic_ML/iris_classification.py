#in sklearn there is a function named datasets and from that diff datas i want to import iris dataset and load_iris helps in data gathering
from sklearn.datasets import load_iris
#for chhosing a model we here import kneighbourclassifier which classify the item and o/p in the form of category or group
from sklearn.neighbors import KNeighborsClassifier
#prepare the data

iris=load_iris()

#chhosing a model
knn=KNeighborsClassifier(n_neighbors=6)

'''train the model,more the training more accuarate model
(iris.data is training data & iris.target is testing data)'''
knn.fit(iris.data,iris.target)
#skipping evaluation & hyper parameter tuning

#prediction of category
a=knn.predict([[4,352,5,2],])
'''o/p is [2], here it displays iris.target values so now by using iris.target_names
we get its category i.e. for 2 it is virginica'''
print(iris.target_names[a])
'''o/p is ['virginica']'''
