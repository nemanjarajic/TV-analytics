import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import string
import seaborn as sns
from matplotlib.pyplot import pie, axis, show
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import stopwords
from collections import Counter




df = pd.read_csv('Eluvio_DS_Challenge.csv')

#Data exploring
df.info()

#Videos with no interaction and videos that have dislikes
print('No up votes', round(df['up_votes'].value_counts()[0]/len(df)*100,2), '% of dataset')
print('No down votes', round(df['down_votes'].value_counts()[0]/len(df)*100,2), '% of dataset')

#Null cells in the dataframe
nan_df = df[df.isnull().any(axis=1)]
print(nan_df.head())

#Unique categorys in the dataframe
print(df['category'].unique())

#All videos have 0 downvotes column has no important info
del df['down_votes']

#splits date created to just the year
df['date_created'] = df['date_created'].apply(lambda s: s.split('-')[0])

'''
#Bar graph of video uploaded per year
videos_per_year = df['date_created'].value_counts(ascending = True)
ax = videos_per_year.plot.bar()
ax.set_title('Videos uploaded per year')
ax.set_ylabel('Number of Videos')
ax.set_xlabel('Year')
plt.show()
'''

'''
#Number of videos authors have created
author_frequency = df['author'].value_counts()
print(author_frequency)

#Top 10 popular authors based on total upvotes
popular_authors = df.groupby('author')['up_votes'].sum().sort_values(ascending = False)
newdf = pd.concat([popular_authors,author_frequency], axis=1)
newdf.columns = ['up_votes', 'frequency']
newdf['average_likes'] = newdf.apply(lambda row: row.up_votes/row.frequency,axis = 1)
print(newdf.sort_values(by='frequency',ascending=False))
print(newdf)
ax1 = popular_authors.head(10).plot.bar()
plt.show()
'''

'''
print(df.groupby(['title', 'author'], sort = False)['up_votes'].max().sort_values(ascending = False))

sns.boxplot(y=df.up_votes)
plt.show()
'''

'''
4
stop = stopwords.words('english')

df['title'] = df['title'].str.lower()
df['title'] = df['title'].str.replace('[^\w\s]','')
df['title'] = df['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

most_common_words = Counter(" ".join(df['title']).split()).most_common(100)
print(most_common_words)
'''
'''
#Correlation matrix for over 18 and votes 
corrmatrix = df.corr()
plt.figure(figsize=(25,16))
hm = sns.heatmap(corrmatrix,annot = True, linewidths=.5,cmap='coolwarm_r')
hm.set_title(label='Heat map of dataset', fontsize = 20)
plt.show()
'''


