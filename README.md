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

![alt text](https://github.com/nemanjarajic/eluvio_DS_ML/blob/main/uploads%20per%20year.png)

There is an increase in videos uploaded per year in 2013 to 2016 compared to 2008 to 2012

![alt text](https://github.com/nemanjarajic/eluvio_DS_ML/blob/main/correlation%20matrix.png)

The correlation matrix of the dataset shows that there is a low correlation between the number of likes a video has and the age restriction.
There is a slight correlation between the year and the up votes but that can be attributed to the increase of videos uploaded per year.

![alt text](https://github.com/nemanjarajic/eluvio_DS_ML/blob/main/pairplot.png)

From the pair plot we can see the number of videos with higher likes increases each year.

The columns that do not hold any important information are the over_18, down_votes, and category.
The date_created column was modified to only hold the year.
```
del df['down_votes']
del df['category']
del df['over_18']
df['date_created'] = df['date_created'].apply(lambda s: s.split('-')[0])
```
## Data Manipulation

Finding the creator that has the highest likes per video by counting the number of up votes a user has and dividing it by the number of videos posted.

```
                  up_votes  frequency  average_likes
maxwellhill        1985416       4023     493.516281
Libertatea          832102       2108     394.735294
Wagamaga            580121       1490     389.342953
kulkke              333311       1199     277.990826
anutensil          1531544       5730     267.285166
EightRoundsRapid    254670       1223     208.233851
NinjaDiscoJesus     492582       2448     201.218137
pnewell             297270       1562     190.313700
PanAfrica           219742       1183     185.749789
madazzahatter       428966       2503     171.380743
```

Top 10 video creators that average the most likes per video having a minimum of 1000 videos published sorted by average_likes

Looking at the most common words in the year 2015 where the most videos were uploaded and there a larger number of videos with high likes.
Also the year 2016 is explored

Most common words in titles with upvotes over 200 in 2015
```
[('us', 728), ('says', 604), ('new', 518), ('isis', 486), ('world', 413), ('russia', 376), ('china', 342), ('russian', 337), ('government', 333), ('people', 321), ('years', 301), ('state', 299), ('first', 290), ('said', 285), ('police', 272), ('saudi', 269), ('president', 261), ('uk', 251), ('found', 249), ('killed', 245), ('islamic', 235), ('syria', 227), ('one', 221), ('two', 212), ('turkey', 209), ('could', 208), ('minister', 207), ('war', 201), ('attack', 198), ('israel', 194), ('year', 193), ('iran', 193), ('germany', 193), ('million', 191), ('court', 190), ('report', 188), ('country', 181), ('india', 180), ('military', 170), ('putin', 166), ('say', 164), ('un', 161), ('group', 159), ('oil', 158), ('paris', 157), ('billion', 156), ('ukraine', 156), ('climate', 156), ('attacks', 149), ('arabia', 148), ('city', 146), ('syrian', 143), ('may', 142), ('death', 141), ('france', 138), ('north', 137), ('south', 136), ('german', 136), ('nuclear', 135), ('chinese', 134), ('officials', 134), ('time', 133), ('would', 131), ('korea', 130), ('calls', 129), ('rights', 128), ('scientists', 128), ('human', 128), ('deal', 126), ('law', 125), ('change', 125), ('french', 124), ('australia', 124), ('eu', 124), ('thousands', 124), ('refugees', 124), ('canada', 123), ('international', 123), ('forces', 
123), ('israeli', 122), ('children', 121), ('ban', 121), ('dead', 121), ('public', 119), ('former', 118), ('three', 118), ('leader', 117), ('turkish', 117), ('obama', 117), ('power', 115), ('man', 114), ('iraq', 114), ('air', 113), ('security', 112), ('trade', 
112), ('army', 111), ('united', 110), ('prime', 110), ('japan', 110), ('women', 109)]
```
Most common words in title with up votes over 200 in 2016
```
[('us', 699), ('says', 559), ('new', 437), ('isis', 327), ('world', 305), ('people', 304), ('china', 301), ('president', 286), ('first', 278), ('police', 278), ('years', 277), ('government', 269), ('turkey', 267), ('russia', 265), ('uk', 242), ('state', 236), ('north', 235), ('said', 230), ('saudi', 226), ('korea', 219), ('attack', 200), ('killed', 197), ('country', 196), ('women', 195), ('found', 192), ('minister', 192), ('million', 191), ('germany', 191), ('un', 185), ('south', 185), ('could', 183), ('year', 183), 
('one', 183), ('eu', 182), ('report', 178), ('india', 176), ('russian', 172), ('say', 167), ('two', 163), ('syria', 162), ('war', 161), ('military', 159), ('scientists', 158), ('human', 158), ('german', 152), ('climate', 151), ('death', 150), ('rights', 147), ('chinese', 144), ('may', 143), ('japan', 142), ('islamic', 136), ('ban', 135), ('french', 132), ('time', 132), ('oil', 130), ('british', 130), ('france', 128), ('court', 127), ('thousands', 126), ('arabia', 124), ('news', 124), ('europe', 121), ('global', 120), ('australia', 118), ('israel', 117), ('last', 117), ('law', 116), ('attacks', 115), ('dead', 112), ('children', 112), ('refugees', 111), ('man', 110), ('canada', 110), ('iran', 109), ('calls', 109), ('turkish', 108), ('since', 107), ('nuclear', 106), ('officials', 106), ('three', 105), ('group', 103), ('air', 102), ('european', 102), ('would', 102), ('change', 102), ('public', 101), ('city', 99), ('muslim', 98), ('international', 97), ('billion', 97), ('party', 96), ('child', 96), ('back', 94), ('leader', 94), ('syrian', 94), ('arrested', 94), ('set', 93), ('according', 93), ('pakistan', 92)]
```


