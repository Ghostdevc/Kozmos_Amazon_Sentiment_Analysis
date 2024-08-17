# !pip install nltk
# !pip install textblob
# !pip install wordcloud


from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud
import nltk

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


##################################################
# 1. Text Preprocessing
##################################################

df = pd.read_excel('../Kozmos_Amazon_Sentiment_Analysis/Analysis/dataset/amazon.xlsx')
df.head()

###############################
# Normalizing Case Folding
###############################

df['Review'] = df['Review'].str.lower()

###############################
# Punctuations
###############################

df['Review'] = df['Review'].str.replace('[^\w\s]', '')

###############################
# Numbers
###############################

df['Review'] = df['Review'].str.replace('\d', '')

###############################
# Stopwords
###############################


nltk.download('stopwords')

sw = stopwords.words('english')

df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

###############################
# Rarewords
###############################

temp_df = pd.Series(' '.join(df['Review']).split()).value_counts()

temp_df.head()

drops = temp_df[temp_df < 1000]

drops.head()

df['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

###############################
# Lemmatization
###############################

nltk.download('wordnet')
df['Review'] = df['Review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df['Review'].head()

##################################################
# 2. Text Visualization
##################################################

###############################
# Terim Frekanslarının Hesaplanması
###############################

tf = df["Review"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

tf.columns = ["words", "tf"]

tf.sort_values("tf", ascending=False)

###############################
# Barplot
###############################

tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()

###############################
# Wordcloud
###############################

text = " ".join(i for i in df['Review'])

wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# wordcloud.to_file("wordcloud.png")
