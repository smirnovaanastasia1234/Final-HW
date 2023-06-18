import streamlit as st
import pickle
import numpy as np
from PIL import Image
import time

st.header("Добро пожаловать! 👋")
st.info("Группа 9: Смирнова А., Кожедуб Н., Багаудинов Э., Петраков В.")

display = Image.open('images/display.jpg')
display = np.array(display)
st.image(display)

st.title('Обнаружение фишинговых URL-адресов')
title = st.text_input('Введите ссылку', 'вот тут')
title = [title]


# Загрузка модели
@st.cache
def load_model():
    model = pickle.load(open('phish.pkl', 'rb'))
    return model


result = st.button('🤗 Распознать')
progress_bar = st.progress(0)
progress_text = st.empty()
for i in range(101):
    time.sleep(0.1)
    progress_bar.progress(i)
    progress_text.text(f"Progress: {i}%")

if result:
    model = load_model()
    y_pred = model.predict(title)
    if y_pred[0] == 0:
        st.success("Этот URL является безопасным", icon="✅")
        st.image('images/class.jpg')
    else:
        st.warning("Этот URL является вредоносным", icon="⚠️")
        st.image('images/warnings.png')
