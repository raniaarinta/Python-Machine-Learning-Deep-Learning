import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")
print(data.head())
data=data[["G1","G2","G3","absences","freetime","famrel","health","Dalc","Walc"]]
predict ="G3"

# X is an attribute
X=np.array(data.drop([predict], 1))
#Y ia a label
Y=np.array(data[predict])
#spliting 10% into test sample
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(X,Y,test_size=0.1)

linear = linear_model.LinearRegression()
#find the best fit line
linear.fit(x_train, y_train)
#return the accuracy of the model
acc=linear.score(x_test, y_test)
print(acc)

print("coefficient: ", linear.coef_)
print("intercept: ", linear.intercept_)

predictions = linear.predict(x_test)

#print out the prediction before and after
for x in range(len(predictions)):
    print(predictions[x],x_test[x],y_test[x])