import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report as  cr
from sklearn.metrics import precision_recall_curve as prc


cancer = datasets.load_breast_cancer()


X = cancer.data
Y = cancer.target

#split data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

classes =['malignant' 'benign']

#SVM classification
clf = svm.SVC(kernel="linear", C=2)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
#print accuracy
acc = metrics.accuracy_score(y_test,y_pred)
print("accuracy: \n")
print(acc)
#print confusion matrix
confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
print("confusion matrix: \n")
print(confusion_matrix)
#classification_report
classification_report=cr(y_test,y_pred)
print(classification_report)
#precission recall
precision,recall, threshold= prc(y_test,y_pred)
print("Precision: ",precision)
print("Recall: ",recall)
print("threshold: ",threshold)

