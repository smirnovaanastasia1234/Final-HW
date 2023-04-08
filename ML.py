#читаю файл
import pandas as pd
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")
print(test.head(4))
print(train.head(4))
#если pandas, то
print (test.describe())
print (train.describe())
