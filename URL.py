
import streamlit as st
import pickle

st.title('Фейковые ссылки')
title = st.text_input('Введите ссылку', 'вот тут')

#Загрузка модели
@st.cache(allow_output_mutation=True)
def load_model():
    movies = pickle.load(open('myfile.pkl','rb'))
    model = pickle.load(movies)
    return model

result = st.sidebar.button('🤗Распознать')

 