# eluvio_DS_ML
Eluvio coding challenge for data science and machine learning 

# Data exploring
DataFrame information
```
df.info()
nan_df = df[df.isnull().any(axis=1)]
print(nan_df.head())
print(df['category'].unique())

'#'   Column        Non-Null Count   Dtype 
 0   time_created  509236 non-null  int64
 1   date_created  509236 non-null  object
 2   up_votes      509236 non-null  int64 
 3   down_votes    509236 non-null  int64 
 4   title         509236 non-null  object
 5   over_18       509236 non-null  bool
 6   author        509236 non-null  object
 7   category      509236 non-null  object
dtypes: bool(1), int64(3), object(4)

Empty DataFrame
Columns: [time_created, date_created, up_votes, down_votes, title, over_18, author, category]
Index: []
['worldnews']
```
From the dataframe there are 509236 rows. None of the rows have null entries. The down votes column has no entry larger than 0.
18% of all videos have no up votes and no down votes. Category column for all rows is worldnews.

Removed unnesscary columns. Edit the date_created column to only hold the year
```
del df['down_votes']
del df['category']
df['date_created'] = df['date_created'].apply(lambda s: s.split('-')[0])
```

![alt text](https://github.com/nemanjarajic/eluvio_DS_ML/blob/main/uploads%20per%20year.png)

Increase in videos uploaded per year in 2013 to 2016 compared to 2008 to 2012

![alt text](https://github.com/nemanjarajic/eluvio_DS_ML/blob/main/correlation%20matrix.png)

The correlation matrix of the dataset shows that there is a low correlation between the number of likes a video has and the age restriction.
There is a slight correlation between the year and the up votes but that can be attributed to the increase of videos uploaded per year.
