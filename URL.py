import streamlit as st
import pickle
import pandas as pd
import numpy as np 
import numpy as np
import nltk as nl 
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer

st.title('Фейковые ссылки')

title = st.text_input('Введите ссылку')

#Загрузка модели
@st.cache_resource
def load_model():
    movies = pickle.load(open('myfile.pkl','rb'))
    model = pickle.load(movies)
    return model

result = st.button('🤗Распознать')


tokenizer = RegexpTokenizer(r'[A-Za-z]+') #[a-zA-Z]обозначает один символ от a до z или от A доZ
stemmer = SnowballStemmer("english")
cv = CountVectorizer()

#Функция для определения СПАМа

def prepare_data(title):
    if not title:
        return None, None
    X = pd.DataFrame({'url': [title]})
    X['text_tokenized'] = X.url.map(lambda t: tokenizer.tokenize(t)) #Разделение на токены
    X['text_stemmed'] = X.text_tokenized.map(lambda t: [stemmer.stem(word) for word in t])#stemmer приводит слова с одним корнем к одному слову
    X['text_sent'] = X.text_stemmed.map(lambda t: ' '.join(t)) #Объеденяем список в предложение
    features = cv.fit_transform(X.text_sent)
    return X, features

X, features = prepare_data(title)

if result and features is not None:
    model = load_model()
    y_pred = model.predict(features)
    if y_pred[0] == 0:
        st.write('Это не спам!')
    else:
        st.write('Это спам!')
