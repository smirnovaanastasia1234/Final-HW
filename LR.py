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

# Visualization
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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
#Обучение модели
logreg = LogisticRegression(max_iter=1000)
trainX, testX, trainY, testY = train_test_split(features, y, test_size=0.3, stratify=y, random_state=42)
logreg.fit(features, y)

#Сохранение обученной модели
with open('myfile.pkl', 'wb') as output:
    pickle.dump(logreg, output)


#Оценка модели

predict= logreg.predict(testX)

print(metrics.classification_report(predict, testY))

print("\n\nAccuracy Score:", metrics.accuracy_score(testY, predict).round(2)*100, "%")

#Полно

mat = confusion_matrix(ytest, ypred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label');

plt.savefig(confusion_matrix_file)







