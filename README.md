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

Most common words in titles with upvotes over 200
```
[('us', 3553), ('says', 2315), ('new', 1999), ('world', 1762), ('russia', 1529), ('police', 1510), ('people', 1505), ('government', 1501), ('china', 1461), ('years', 1225), ('first', 1134), ('president', 1120), ('russian', 1117), ('uk', 1116), ('isis', 1096), ('state', 1065), ('israel', 1053), ('one', 1027), ('north', 1017), ('war', 999), ('said', 988), ('killed', 959), ('korea', 939), ('found', 911), ('country', 890), ('two', 838), ('un', 828), ('court', 821), ('minister', 818), ('military', 815), ('year', 806), ('could', 805), ('syria', 802), ('ukraine', 791), ('women', 775), ('india', 775), ('saudi', 775), ('attack', 772), ('million', 756), ('south', 723), ('germany', 720), ('report', 718), ('say', 714), ('death', 712), ('chinese', 706), ('man', 698), ('law', 692), ('turkey', 684), ('iran', 658), ('islamic', 657), ('rights', 656), ('would', 631), ('human', 629), ('may', 627), ('news', 625), ('japan', 622), ('british', 620), ('canada', 614), ('german', 610), ('time', 609), ('oil', 600), ('children', 583), ('dead', 578), ('israeli', 576), ('ban', 575), ('eu', 570), ('france', 560), ('nuclear', 550), ('snowden', 547), ('putin', 546), ('calls', 540), ('internet', 539), ('international', 535), ('city', 535), ('group', 526), ('last', 510), ('iraq', 507), ('billion', 505), ('french', 502), ('thousands', 502), ('officials', 500), ('public', 496), ('australia', 494), ('use', 490), ('australian', 488), ('nsa', 483), ('three', 479), ('global', 477), ('syrian', 473), ('woman', 470), ('europe', 470), ('climate', 469), ('security', 465), ('obama', 463), ('since', 462), ('drug', 462), ('protest', 461), ('european', 460), ('united', 459), ('power', 459)]
```



