#читаю файл
import pandas as pd
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")
test.head(4)
train.head(4)
#смотрю первые четыре строки
#c = f.readlines()
#print(c[:5])
#смотрю количество записей
#print(len(c))