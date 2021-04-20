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
del df['category']
del df['over_18']

#splits date created to just the year
df['month'] = df['date_created'].apply(lambda s: s.split('-')[1]).astype('int16')
df['year'] = df['date_created'].apply(lambda s: s.split('-')[0]).astype('int16')

del df['date_created']
#print(df.head(10))

titles = df['title']
title_lengths = [len(title) for title in titles]
df['title_length'] = title_lengths

df['time_created'] = pd.to_timedelta(df['time_created'], unit='s')
df['time_created'] = df['time_created'].apply(lambda s: str(s))
df['time_created'] = df['time_created'].apply(lambda s: s.split(' ')[2])

print(df.head(10))

'''
df.info()
#Bar graph of video uploaded per year
videos_per_year = df['year'].value_counts().sort_index()
print(videos_per_year)
ax = videos_per_year.plot.bar()
ax.set_title('Videos uploaded per year')
ax.set_ylabel('Number of Videos')
ax.set_xlabel('Year')
plt.show()
'''
'''
years = [2008,2009,2010,2011,2012,2013,2014,2015,2016]
for year in years:
    month = df['month'].loc[df['year'] == year].value_counts().sort_index()
    ax1 = month.plot.bar()
    ax1.set_title(str(year)+' Videos per month')
    ax1.set_ylabel('Videos')
    ax1.set_xlabel('Month')
    plt.show()
'''

'''
#Correlation matrix for over 18 and votes 
corrmatrix = df.corr()
plt.figure(figsize=(25,16))
hm = sns.heatmap(corrmatrix,annot = True, linewidths=.5,cmap='coolwarm_r')
hm.set_title(label='Heat map of dataset', fontsize = 20)
plt.show()
'''

'''
#Number of videos authors have created
author_frequency = df['author'].value_counts()
print(author_frequency.head(10))
'''


'''

#Top 10 popular authors based on total upvotes
popular_authors = df.groupby('author')['up_votes'].sum().sort_values(ascending = False)
newdf = pd.concat([popular_authors,author_frequency], axis=1)
newdf.columns = ['up_votes', 'frequency']
newdf['average_likes'] = newdf.apply(lambda row: row.up_votes/row.frequency,axis = 1)
print(newdf.loc[newdf['frequency']>500].sort_values(by='average_likes',ascending=False).head(10))

print(newdf['average_likes'].loc[newdf['frequency']>500].mean())
sns.boxplot(newdf['average_likes'].loc[newdf['frequency']>500])
plt.show()
'''
'''
#Most viewed videos in the year 2015 and 2016
print(df.loc[df['year'] == 2015].sort_values(by='up_votes',ascending=False).head(20))
print(df.loc[df['year'] == 2016].sort_values(by='up_votes',ascending=False).head(20))
'''
print(df.sort_values(by='up_votes',ascending=False).head(10))
'''
#Pair plot
sns.set()
cols = ['up_votes', 'year', 'month']
sns.pairplot(df[cols],size = 2.5)
plt.show()
'''

'''
plt.hist(df['title_length'],bins = 500)
plt.show()
'''

'''
#Most common words with videos that have over 200 up votes
stop = stopwords.words('english')
newdf = df.loc[(df['up_votes']>200)]
newdf['title'] = newdf['title'].str.lower()
newdf['title'] = newdf['title'].str.replace('[^\w\s]','')
newdf['title'] = newdf['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

most_common_words = Counter(" ".join(newdf['title']).split()).most_common(100)
print(most_common_words)
'''

'''
#Videos uploaded at certain hours
df['time_created'] = df['time_created'].apply(lambda s: s.split(':')[0]).astype('int16')
time_uploaded = df['time_created'].value_counts().sort_index()
ax3 = time_uploaded.plot.bar()
ax3.set_title('Upload times of videos')
ax3.set_xlabel('Hour')
ax3.set_ylabel('Number of videos')
plt.show()
'''

'''
#likes per hour
df['time_created'] = df['time_created'].apply(lambda s: s.split(':')[0]).astype('int16')

cols = ["time_created","up_votes"]
timeDf = pd.DataFrame(cols)
for i in range(0,24):
    #sum = df['up_votes'].loc[df['time_created'] == i].sum()
    sum = df['up_votes'].loc[(df['time_created'] == i) & (df['year'] == 2014)].sum()
    tempdf = {'time_created' : i, 'up_votes': sum}
    timeDf = timeDf.append(tempdf, ignore_index = True)

timeDf = timeDf.iloc[2:]
timeDf.reset_index()

timeDf.plot(x='time_created',y='up_votes',kind='bar')
plt.show()
'''
