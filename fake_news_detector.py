import pandas as pd
import re
import seaborn as sns
import string


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

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
news_df["text"] = news_df["text"].apply(clean_text)

x = news_df["text"]
y = news_df["class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

vectorisation = TfidfVectorizer()
x_v_train = vectorisation.fit_transform(x_train)
x_v_test = vectorisation.transform(x_test)

lr = LogisticRegression()
lr.fit(x_v_train,y_train)
pred_lr = lr.predict(x_v_test)
lr.score(x_v_test, y_test)
print(classification_report(y_test,pred_lr))