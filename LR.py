import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import pickle
#Загрузка данных
df = pd.read_csv("train.csv")
print(df.shape)
print(df.head())
df.Predicted.value_counts()
#Разделение данных на функции и метки
X = df[['url']].copy()
y = df.Predicted.copy()
#Обработка данных
tokenizer = RegexpTokenizer(r'[A-Za-z]+') #[a-zA-Z]обозначает один символ от a до z или от A доZ
stemmer = SnowballStemmer("english")
cv = CountVectorizer()
def prepare_data(X) :
    X['text_tokenized'] = X.url.map(lambda t: tokenizer.tokenize(t)) #Разделение на токены
    X['text_stemmed'] = X.text_tokenized.map(lambda t: [stemmer.stem(word) for word in t])#stemmer приводит слова с одним корнем к одному слову
    X['text_sent'] = X.text_stemmed.map(lambda t: ' '.join(t)) #Объеденяем список в предложение
    features = cv.fit_transform(X.text_sent)
    return X, features
X, features = prepare_data(X)
features.toarray()
#Обучение модели
logreg = LogisticRegression()
trainX, testX, trainY, testY = train_test_split(features, y, test_size=0.3, stratify=y, random_state=42)
logreg.fit(features, y)
#Сохранение обученной модели
with open('myfile.pkl', 'wb') as output:
    pickle.dump(logreg, output)

