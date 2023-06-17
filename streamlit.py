import streamlit as st
import pickle
import numpy as np
from PIL import  Image

st.header(" Добро пожаловать! 👋")
st.info("Группа 9: Смирнова А., Кожедуб Н., Багаудинов Э., Петраков В.")

display = Image.open('images/display.jpg')
display = np.array(display)
st.image(display)

st.title('Обнаружение фишинговых URL-адресов')
title = st.text_input('Введите ссылку', 'вот тут')
title = [title]
# Загрузка модели
@st.cache_data
def load_model():
    model = pickle.load(open('phish.pkl', 'rb'))
    return model

result = st.button('🤗Распознать')
if result:
    model = load_model()
    y_pred = model.predict(title)
    if y_pred[0] == 0:
        st.write("Этот URL является безопасным")
    else:
         st.write("Этот URL является вредоносным")