import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import  metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()


X = cancer.data
Y = cancer.target

#split data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

classes =['malignant' 'benign']

#SVM classification
clf = svm.SVC(kernel="rbf")
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)