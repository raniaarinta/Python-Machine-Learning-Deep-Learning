import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as  pyplot
import pickle
from matplotlib import style

#age,sex,bmi,children,smoker,region,charges
data = pd.read_csv("insurance.csv")
print(data.head())
data = data[["age","bmi","children","charges","smoker"]]
predict ="charges"

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

style.use("ggplot")
p='smoker'
#scatter plot
pyplot.scatter(data[p],data["charges"])
pyplot.xlabel(p)
pyplot.ylabel("charges")
pyplot.show()