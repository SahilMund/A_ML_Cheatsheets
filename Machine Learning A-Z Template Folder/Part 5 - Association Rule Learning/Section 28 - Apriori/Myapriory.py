# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
#creating our input in the form of list of products in a list of transactions
transactions=[]#list containing the transactions
for i in range (0,7501):#This loop is to create the list containing the transactions
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])#this loop is to insert the list of products in the list of transactions

#training the apiori dataset
from apyori import apriori
rules=apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)

#visualising the results
results=list(rules)
#for viewing the relations between different products in the results
view_results = []
for i in range(0, len(results)):
    view_results.append('RULE:\t{}\nSUPP:\t{}\nCONF:\t{}\nLIFT:\t{}\n'.format(list(results[i][0]), str(results[i][1]), str(results[i][2][0][2]), str(results[i][2][0][3])))