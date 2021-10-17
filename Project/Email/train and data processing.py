# import Libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# dataset
dataset=pd.read_csv("/content/drive/MyDrive/DATA_SET/emails.csv")
print(dataset)

# pre-process, part-1
text=dataset['text']
text=text.map(lambda p:p[9:])
print(text)

spam=dataset['spam']
print(spam)

# pre-process, part-2
# string to numeric
cv = CountVectorizer(dtype='float64',lowercase=True, max_df=.7, max_features=None, min_df=5,
                   ngram_range=(1, 3), token_pattern='(?u)\\b\\w\\w+\\b', stop_words='english',
                   analyzer='word')

text_matrix = cv.fit_transform(list(text)).toarray()
print(text_matrix,'\n\n')

# dump "cv" into pkl model
model_name_cv="cv.pkl"
pickle.dump(cv, open(model_name_cv,'wb'))

spam_matrix=list(spam)
print(spam_matrix,'\n\b')

print(np.array(text_matrix).shape, np.array(spam_matrix).shape)

# split it for train
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(text_matrix, spam_matrix, test_size = 0.002)
print(y_train[:10])

# train model
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
classifier.fit(x_train , y_train)

# dump model
model_name="model.pkl"
pickle.dump(classifier, open(model_name,'wb'))

# predict
y_pred = classifier.predict(x_test)
print('y_pred :- ',y_pred)
print('y_test :- ',y_test)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('cm :- \n',cm)

# accuray score
from sklearn.metrics import accuracy_score
print('\naccuracy_score :- \n',accuracy_score(y_test, y_pred))
print('\naccuracy_score :- \n',str(int((round(accuracy_score(y_test, y_pred),2))*100))+"%")