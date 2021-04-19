# eluvio_DS_ML
Eluvio coding challenge for data science and machine learning 

Analytical Insight of Dataset

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

![alt_text](https://github.com/nemanjarajic/eluvio_DS_ML/blob/main/videospermonth.png)
Increase in videos uploaded per month over the 8 year period

![alt text](https://github.com/nemanjarajic/eluvio_DS_ML/blob/main/correlation%20matrix.png)

The correlation matrix of the dataset shows that there is a low correlation between the number of likes a video has and the age restriction.
There is a slight correlation between the year and the up votes but that can be attributed to the increase of videos uploaded per year.

![alt text](https://github.com/nemanjarajic/eluvio_DS_ML/blob/main/updatedpairplot.png)

From the pair plot we can see the number of videos with higher likes increases each year and per month the number of up votes on videos looks to be around the same.


The columns that do not hold any important information are the over_18, down_votes, and category.
The date_created column was seperated into a months column and a year column.

```
del df['down_votes']
del df['category']
del df['over_18']

df['month'] = df['date_created'].apply(lambda s: s.split('-')[1]).astype('int16')
df['year'] = df['date_created'].apply(lambda s: s.split('-')[0]).astype('int16')

del df['date_created']
```


### Finding the creator that has the highest likes per video by counting the number of up votes a user has and dividing it by the number of videos posted.
Creators with the most published videos
```
davidreiss666    8897
anutensil        5730
DoremusJessup    5037
maxwellhill      4023
igeldard         4013
readerseven      3170
twolf1           2923
madam1           2658
nimobo           2564
madazzahatter    2503
```
Top 10 video creators that average the most likes per video having a minimum of 500 videos published sorted by average_likes
```
                    up_votes  frequency  average_likes
maxwellhill          1985416       4023     493.516281
Libertatea            832102       2108     394.735294
Wagamaga              580121       1490     389.342953
green_flash           205554        638     322.184953
mepper                223369        699     319.555079
the_last_broadcast    154012        514     299.634241
kulkke                333311       1199     277.990826
anutensil            1531544       5730     267.285166
_Perfectionist        145825        664     219.615964
EightRoundsRapid      254670       1223     208.233851
```


A minimum of 500 uploads was chosen to limit the amount of outliers 

![alt text](https://github.com/nemanjarajic/eluvio_DS_ML/blob/main/votesboxplot.png)

The average amount of up likes per videos is about 70 with the top uploaders being the outliers

### Looking at the most common words with vides having more than 200 up votes

Most common words in all years
```
[('us', 3553), ('says', 2315), ('new', 1999), ('world', 1762), ('russia', 1529), ('police', 1510), ('people', 1505), ('government', 1501), ('china', 1461), ('years', 1225), ('first', 1134), ('president', 1120), ('russian', 1117), ('uk', 1116), ('isis', 1096), ('state', 1065), ('israel', 1053), ('one', 1027), ('north', 1017), ('war', 999), ('said', 988), ('killed', 959), ('korea', 939), ('found', 911), ('country', 890), ('two', 838), ('un', 828), ('court', 821), ('minister', 818), ('military', 815), ('year', 806), ('could', 805), ('syria', 802), ('ukraine', 791), ('women', 775), ('india', 775), ('saudi', 775), ('attack', 772), ('million', 756), ('south', 723), ('germany', 720), ('report', 718), ('say', 714), ('death', 712), ('chinese', 706), ('man', 698), ('law', 692), ('turkey', 684), ('iran', 658), ('islamic', 657), ('rights', 656), ('would', 631), ('human', 629), ('may', 627), ('news', 625), ('japan', 622), ('british', 620), ('canada', 614), ('german', 610), ('time', 609), ('oil', 600), ('children', 583), ('dead', 578), ('israeli', 576), ('ban', 575), ('eu', 570), ('france', 560), ('nuclear', 550), ('snowden', 547), ('putin', 546), ('calls', 540), ('internet', 539), ('international', 535), ('city', 535), ('group', 526), ('last', 510), ('iraq', 507), ('billion', 505), ('french', 502), ('thousands', 502), ('officials', 500), ('public', 496), ('australia', 494), ('use', 490), ('australian', 488), ('nsa', 483), ('three', 479), ('global', 477), ('syrian', 473), ('woman', 470), ('europe', 470), ('climate', 469), ('security', 465), ('obama', 463), ('since', 462), ('drug', 462), ('protest', 461), ('european', 460), ('united', 459), ('power', 459)]
```

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
From the two years the United States, ISIS, China, and Russia were a few of the top most common words in titles the users published videos on.

### Insight of which videos recieved the most amount of up votes for the two most recent years due to high user traffic

Top 10 videos in 2016
```
450818     18:01:46     13244  2.6 terabyte leak of Panamanian shell company ...         mister_geaux      4  2016           277
449809     11:19:33     11108  Hundreds of thousands of leaked emails reveal ...               Xiroth      3  2016           100
500786     12:47:15     10394  Feeding cows seaweed could slash global greenh...                 mvea     10  2016           225
465396     06:31:42     10289  Every reference to the Great Barrier Reef remo...  Flamo_the_Idiot_Boy      5  2016           141
495329     14:26:23     10239  Iceland s capital Reykjavik to switch off stre...            tiribazus      9  2016           122
447311     09:47:32     10161  TTIP: secrecy around talks is  profoundly unde...             Wagamaga      3  2016           129
437930     15:35:45      9520      Gravitational waves from black holes detected         Andromeda321      2  2016            45
494404     15:45:05      9492  The UN just declared antibiotic resistance “th...            MarcoshLA      9  2016            86
480475     16:46:36      9467  Norway considers giving mountain to Finland as...        bimonscificon      7  2016           190
446779     12:13:04      9319  FIFA admits to World Cup hosting bribes, asks ...              jurvand      3  2016            70
```

Top 10 videos in 2015
```
377200     16:41:11     21253  A biotech startup has managed to 3-D print fak...            KRISHNA53      6  2015           289
391415     12:57:59     13435  Twitter has forced 30 websites that archive po...        joeyoungblood      8  2015           139
391318     22:09:28     12333  The police officer who leaked the footage of t...     navysealassulter      8  2015           243
390252     23:06:08     11288  Paris shooting survivor suing French media for...            seapiglet      8  2015            98
397215     00:14:48     10922  Brazil s Supreme Court has banned corporate co...        DoremusJessup      9  2015            92
390494     00:30:33     10515  ISIS beheads 81-year-old pioneer archaeologist...        DawgsOnTopUGA      8  2015           188
388230     15:58:55     10377  Brazilian radio host famous for exposing corru...              fiffers      8  2015           122
389011     13:10:59     10086  India sues Nestle for nearly $100m because lea...            damsteegt      8  2015            72
412603     21:05:26      9967   Shootings  reported in central Paris: Reports...              emr1028     11  2015            90
354925     13:09:46      9954  France decrees new rooftops must be covered in...              pnewell      3  2015           174
```

The top videos from the two years show that there is an interest in scientific findings or tech related videos from the userbase.
Foreign affiars was also popular but makes sense because the category title is world news.
Though the United States had several videos published in the two years there was not much traction with the users.
