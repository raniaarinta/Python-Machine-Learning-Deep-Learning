import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model

#Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
data = pd.read_csv("diabetes.csv")
print(data.head())
#data = data[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]]
data = data[["Glucose","BloodPressure","SkinThickness","Insulin","DiabetesPedigreeFunction","Outcome"]]

predict = "Outcome"
# X is an attribute
X=np.array(data.drop([predict], 1))
#Y ia a label
Y=np.array(data[predict])
#split data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
print(x_train,y_test)

#the amount of neigbors
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)
acc = model.score(x_test,y_test)
print(acc)

predicted=model.predict(x_test)
for x in range(len(predicted)):
    print("predicted", predicted[x], "data", x_test[x], "actual", y_test[x])

