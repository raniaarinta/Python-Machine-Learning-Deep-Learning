import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import  metrics
import pickle
from sklearn.ensemble import RandomForestClassifier


#Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
data = pd.read_csv("diabetes.csv")
print(data.head())
#data = data[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]]
data = data[["Glucose","BloodPressure","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]]

predict = "Outcome"
# X is an attribute
X=np.array(data.drop([predict], 1))
#Y ia a label
Y=np.array(data[predict])
#split data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
print(x_train,y_test)



#Knn 7
knn7 = KNeighborsClassifier(n_neighbors=7)
knn7.fit(x_train, y_train)
y_pred_kkn7 = knn7.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred_kkn7)
print('accuracy KNN 7 neighbor',acc)

#SVM_rbf
clf = svm.SVC(kernel="rbf")
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print('svm acc:', acc)

#randomforest
rd_model= RandomForestClassifier(max_depth=2, random_state=0)
rd_model.fit(x_train, y_train)
y_pred2 = rd_model.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred2)
print('random forrest', acc)

#save model
knnPickle = open('knn7pickle', 'wb')
randomforestPickle = open('randomforestpickle', 'wb')
svmPickle = open('svmpickle', 'wb')
# source, destination
pickle.dump(knn7, knnPickle)
pickle.dump(rd_model, randomforestPickle)
pickle.dump(clf , svmPickle)






