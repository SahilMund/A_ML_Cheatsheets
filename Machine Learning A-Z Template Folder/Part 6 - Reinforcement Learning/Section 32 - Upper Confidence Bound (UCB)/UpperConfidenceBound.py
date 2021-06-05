# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_optimisation.csv')

#implementing UCB
N=10000 #Representing the number of customers
d=10 #Representing the no of ad versions
ads_selected=[]
#[0]*d represents a vector containing only zeros of size d. It is because initially the sumber of selection and sum of reward of each round is 0
numbers_of_selections=[0] * d #no. of selection is the variable representing no. of times ad i as selected upto round n
sum_of_reward=[0]*d #sum_of_reward is the variable representing sum of reward of ad i upto round n
total_reward=0
#calculating avg reward and confidence at each round
import math
for n in range(0,N): #for all customers on social media
    max_upper_bound=0
    ad=0
    for i in range(0,d): #for all add versions
        if numbers_of_selections[i]>0:
           avg_reward = sum_of_reward[i] / numbers_of_selections[i]
           #caclulateing confidence(of upper bound only)
           delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
           upper_bound=avg_reward+delta_i
        else:
          upper_bound = 1e400
        
        if upper_bound>max_upper_bound:
            max_upper_bound=upper_bound
            ad=i
    ads_selected.append(ad)
    numbers_of_selections[ad]=numbers_of_selections[ad]+1
    reward=dataset.values[n,ad]
    sum_of_reward[ad]=sum_of_reward[ad]+reward
    total_reward=total_reward+reward

#Visualizing the results
plt.hist(ads_selected)
plt.title('Histogram of Ads selections')
plt.xlabel('Ads')
plt.ylabel('No.of times each ad was selected')
plt.show()

        