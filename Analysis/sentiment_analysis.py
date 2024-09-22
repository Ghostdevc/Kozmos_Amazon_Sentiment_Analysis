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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate,train_test_split

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

##################################################
# 3. Sentiment Analysis
##################################################

df['Review'].head()

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

df["Review"][0:10].apply(lambda x: sia.polarity_scores(x))

df["Review"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

df["Review"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["sentiment_label"] = df["Review"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["sentiment_label"] = LabelEncoder().fit_transform(df["sentiment_label"])

##################################################
# 4. ML Preparation
##################################################

y = df["sentiment_label"]
X = df["Review"]

tf_idf_word_vectorizer = TfidfVectorizer()
X_tf_idf_word = tf_idf_word_vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tf_idf_word,
                                                    y,
                                                    test_size=0.20, random_state=17)

