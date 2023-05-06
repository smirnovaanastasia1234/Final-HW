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
df = pd.read_csv("train.csv")
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

loaded_model = pickle.load(open('phish.pkl', 'rb'))
result = loaded_model.score(testX,testY)
print(result)

predict_good = ['youtube.com']
loaded_model = pickle.load(open('phish.pkl', 'rb'))
result = loaded_model.predict(predict_good)
print(result)

