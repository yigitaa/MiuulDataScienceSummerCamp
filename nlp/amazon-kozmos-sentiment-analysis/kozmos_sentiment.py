from warnings import filterwarnings
import matplotlib
matplotlib.use('Qt5Agg')
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
from sklearn.model_selection import train_test_split
filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

##################################################
# 1. Text Preprocessing
##################################################
df = pd.read_excel("Natural Language Processing/Projects/Amazon_Reviews_Sentimental_Analysis/amazon.xlsx")
df.head()

###############################
# Normalizing Case Folding
###############################
df["Review"] = df["Review"].str.lower()

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
sw = stopwords.words('english')
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

###############################
# Rarewords
###############################
temp_df = pd.Series(' '.join(df['Review']).split()).value_counts()
drops = temp_df[temp_df <= 1]
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

drops.head(12)

###############################
# Lemmatization
###############################
df['Review'] = df['Review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

##################################################
# 2. Text Visualization
##################################################

###############################
# Calculating Term Frequencies
###############################
tf = df["Review"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]
tf.sort_values("tf", ascending=False).head(50)

###############################
# Barplot
###############################
tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()

###############################
# Wordcloud
###############################
text = " ".join(i for i in df.Review)

wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud.to_file("wordcloud.png")

##################################################
# 3. Sentiment Analysis
##################################################
df["Review"].head(20)

sia = SentimentIntensityAnalyzer()
df["Review"][:10].apply(lambda x: sia.polarity_scores(x))
df["Review"][:10].apply(lambda x: sia.polarity_scores(x)["compound"])
df["Review"][:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["polarity_score"] = df["Review"].apply(lambda x: sia.polarity_scores(x)["compound"])

##################################################
# 4. Feature Engineering
##################################################

df["sentiment_label"] = df["Review"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")
df["sentiment_label"] = LabelEncoder().fit_transform(df["sentiment_label"])

y = df["sentiment_label"]
X = df["Review"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# TF-IDF word
X_train_tf_idf = TfidfVectorizer().fit_transform(X_train)
X_test_tf_idf = TfidfVectorizer().fit(X_train).transform(X_test)

##############################
# Logistic Regression
###############################

log_model = LogisticRegression().fit(X_train_tf_idf, y_train)
y_pred_train = log_model.predict(X_train_tf_idf)
y_pred_test = log_model.predict(X_test_tf_idf)

from sklearn.metrics import accuracy_score
acc_train = accuracy_score(y_train, y_pred_train)
# 0.922
acc_test = accuracy_score(y_test, y_pred_test)
# 0.897

from sklearn.metrics import classification_report
print(classification_report(y_train, y_pred_train))
# Acc: 0.92, Prec: 0.93, Recall: 0.92, F1: 0.91
print(classification_report(y_test, y_pred_test))
# Acc: 0.90, Prec: 0.90, Recall: 0.90, F1: 0.88

cross_val_score(log_model,
                X_train_tf_idf,
                y_train,
                scoring="accuracy", cv=5).mean()
# 0.887
cross_val_score(log_model,
                X_test_tf_idf,
                y_test,
                scoring="accuracy", cv=5).mean()
# 0.847

# Sample
random_review = pd.Series(df["Review"].sample(1).values)
print(random_review)

from sklearn.feature_extraction.text import CountVectorizer
vectorized_random_review = CountVectorizer().fit(X_train).transform(random_review)

print(vectorized_random_review)
print(log_model.predict(vectorized_random_review))

###############################
# Random Forests
###############################

rf_model = RandomForestClassifier().fit(X_train_tf_idf, y_train)
cross_val_score(rf_model,
                X_train_tf_idf,
                y_train,
                cv=5, scoring="accuracy", n_jobs=-1).mean()
# 0.913
cross_val_score(rf_model,
                X_test_tf_idf,
                y_test,
                cv=5, scoring="accuracy", n_jobs=-1).mean()
# 0.874
