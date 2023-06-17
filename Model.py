import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import pickle
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline

#Загрузка данных
df = pd.read_csv("train.csv")
#Разделение данных на функции и метки
X = df[['url']].copy()
y = df.Predicted.copy()

tokenizer = RegexpTokenizer(r'[A-Za-z]+') #[a-zA-Z]обозначает один символ от a до z или от A доZ
stemmer = SnowballStemmer("english")
cv = CountVectorizer()
X['text_tokenized'] = X.url.map(lambda t: tokenizer.tokenize(t)) #Разделение на токены
X['text_stemmed'] = X.text_tokenized.map(lambda t: [stemmer.stem(word) for word in t])#stemmer приводит слова с одним корнем к одному слову
X['text_sent'] = X.text_stemmed.map(lambda t: ' '.join(t)) #Объеденяем список в предложение

pipeline_ls = make_pipeline(CountVectorizer(tokenizer = RegexpTokenizer(r'[A-Za-z]+').tokenize, stop_words='english'),LogisticRegression())
trainX, testX, trainY, testY = train_test_split(df.url, df.Predicted)
pipeline_ls.fit(trainX,trainY)
pipeline_ls.score(testX,testY)
print('Training Accuracy :',pipeline_ls.score(trainX,trainY))
print('Testing Accuracy :',pipeline_ls.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(pipeline_ls.predict(testX), testY),
            columns = ['Прогноз: Плохо', 'Прогноз: Хорошо'],
            index = ['Факт:Плохо', 'Факт:Хорошо'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(pipeline_ls.predict(testX), testY,
                            target_names =['Плохой','Хороший']))

print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")
pickle.dump(pipeline_ls,open('phish.pkl','wb'))

