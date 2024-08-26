import pandas as pd
import re
import string


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

#Read csv files
fake_df = pd.read_csv("Data\Fake.csv")
true_df = pd.read_csv("Data\True.csv")

#Assign classes
fake_df["class"]=0
#print(fake_df.head())
true_df["class"]=1
#print(true_df.head())

news_df = pd.concat([fake_df, true_df])

#print(news_df)

news_df = news_df.drop(["subject", "title", "date"], axis = 1)
#print(news_df)

#clean data

#functions
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text