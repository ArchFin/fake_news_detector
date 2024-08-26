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


def detect_fake_news(news_text):
    testing_news = { "text" : [news_text] }
    new_news_df = pd.DataFrame(testing_news)
    new_news_df["text"] = new_news_df["text"].apply(clean_text)
    new_x_test = new_news_df["text"]

    #vectorisation
    new_x_v_test = vectorisation.transform(new_x_test)

    pred_lr = lr.predict(new_x_v_test)
    pred_dt  = dt.predict(new_x_v_test)

    return print("\n\nLR Prediction: {} \nDT Predicton: {}".format(label(pred_lr[0]), label(pred_dt[0])))

def label(n):
    if n==0:
        return "Fake news"
    elif n==1:
        return "Real news"
    else:
        raise "error"



#Read csv files
fake_df = pd.read_csv("Data\Fake.csv")
true_df = pd.read_csv("Data\True.csv")

#Assign classes
fake_df["class"]=0
#print(fake_df.head())
true_df["class"]=1
#print(true_df.head())

news_df = pd.concat([fake_df, true_df])

news_df = news_df.drop(["subject", "title", "date"], axis = 1)
#print(news_df)

#clean data
news_df["text"] = news_df["text"].apply(clean_text)

x = news_df["text"]
y = news_df["class"]

#train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

#vectorisation
vectorisation = TfidfVectorizer()
x_v_train = vectorisation.fit_transform(x_train)
x_v_test = vectorisation.transform(x_test)

#logistic regression
lr = LogisticRegression()
lr.fit(x_v_train,y_train)

#prediction and scoring
pred_lr = lr.predict(x_v_test)
print(lr.score(x_v_test, y_test))
print(classification_report(y_test,pred_lr))

#decision tree classifier
dt = DecisionTreeClassifier()
dt.fit(x_v_train,y_train)

#prediction and scoring linear tree
pred_dt  = dt.predict(x_v_test)
print(dt.score(x_v_test, y_test))
print(classification_report(y_test,pred_dt))

news_input  = input("Please put in news text, cheers")
detect_fake_news(news_input)