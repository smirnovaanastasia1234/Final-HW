#читаю файл
f = open('test.csv', 'r', encoding='utf-8')
print(f)
#смотрю первые четыре строки
c = f.readlines()
print(c[:5])
#смотрю количество записей
print(len(c))