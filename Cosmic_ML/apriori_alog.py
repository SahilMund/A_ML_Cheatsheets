from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
from mlxtend.frequent_patterns import apriori
dataset=[['milk','onion','nutmeg','kidney beans','eggs','yogurt'],
         ['dill','onion','nutmeg','kidney beans','eggs','yogurt'],
         ['milk','apple','kidney beans','eggs'],
         ['milk','unicorn','corn','kidney beans','yogurt'],
         ['corn','onion','onion','kidney beans','ice creams','eggs']]

te=TransactionEncoder()
Trans_array=te.fit(dataset).transform(dataset) #converting values according to the dataframe
df=pd.DataFrame(Trans_array,columns=te.columns_) #df is dataframe
print(df)  #if values are present than ture else false

'''finding support for each values'''

ap=apriori(df,min_support=0.6,use_colnames=True) #0.6 is the threshold value & use_colnames displays the values present in colmn otherwise it will give the index no of data items
#print(ap)  # the values that are less than threshold values are neglated
'''create a new column for apriori model i.e. for no of items '''
ap['length']=ap['itemsets'].apply(lambda x:len(x)) #here length means no. of items from itemset
print(ap)

'''lambda take the values from itemsets & and check the length for the x and then print it's corresponding length'''


print(ap[(ap['length']==2) & (ap['support']>=0.8)])
'''here only the values will displyed from ap whose only  contain 2 dataitems in a datasets & whose support value % is more than 80%'''


