# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_optimisation.csv')

N=10000 #Representing the number of customers
d=10 #Representing the no of ad versions
ads_selected=[]
#[0] * d represents a vector containing only zeros of size d. It is because initially the numbers_of_rewards_1 and snumbers_of_rewards_0 of each round is 0
numbers_of_rewards_1=[0] * d #no. of times ad gets reward 1 upto round n
numbers_of_rewards_0=[0]*d #no. of times ad gets reward 0 upto round n
total_reward=0
#calculating avg reward and confidence at each round
import random
for n in range(0,N): #for all customers on social media
    max_random=0
    ad=0
    for i in range(0,d): #for all add versions
        random_beta=random.betavariate(numbers_of_rewards_1[i]+1,numbers_of_rewards_0[i]+1)
        
        if random_beta>max_random:
            max_random=random_beta
            ad=i
    ads_selected.append(ad)
    reward=dataset.values[n,ad]
    if reward==1:
        numbers_of_rewards_1[ad]=numbers_of_rewards_1[ad]+1
    else:
        numbers_of_rewards_0[ad]=numbers_of_rewards_0[ad]+1
    total_reward=total_reward + reward

#Visualizing the results
plt.hist(ads_selected)
plt.title('Histogram of Ads selections')
plt.xlabel('Ads')
plt.ylabel('No.of times each ad was selected')
plt.show()

        