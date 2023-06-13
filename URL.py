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
title = st.text_input('Введите ссылку', value='')

#Загрузка модели
@st.cache(allow_output_mutation=True)
def load_model():
    with open('myfile.pkl','rb') as movies: 
        model = pickle.load(movies)
    return model

# создаем датафрейм с колонкой 'url'
data = {'url': [title]}
df = pd.DataFrame(data)
X = df[['url']].copy()

tokenizer = RegexpTokenizer(r'[A-Za-z]+') #[a-zA-Z]обозначает один символ от a до z или от A доZ
stemmer = SnowballStemmer("english")
cv = CountVectorizer()

#Функция для определения СПАМа

def prepare_data(title, non_optional_func=None):
    if not title:
        return None, None
    X = pd.DataFrame({'url': [title]})
    X['text_tokenized'] = X.url.map(lambda t: tokenizer.tokenize(t)) #Разделение на токены
    X['text_stemmed'] = X.text_tokenized.map(lambda t: [stemmer.stem(word) for word in t])#stemmer приводит слова с одним корнем к одному слову
    X['text_sent'] = X.text_stemmed.map(lambda t: ' '.join(t)) #Объеденяем список в предложение
    features = cv.fit_transform(X.text_sent)
    return X, features


X, features = prepare_data(X)

if result and features is not None:
    model = load_model()
    y_pred = model.predict(features)
    if y_pred[0] == 0:
        st.write('Это не спам!')
    else:
        st.write('Это спам!')

X, features= prepare_data(title)

model = load_model()

if st.button("Проверить"):
    X, features= prepare_data(title)
    pred = model.predict(X, features)
    if pred == 1:
        st.write("Этот URL является безопасным")
    else:
        st.write("Этот URL является вредоносным")
