from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from confusion_matrix_graph import plot_confusion_matrix
import numpy as np
import sklearn.model_selection as modelsel
from sklearn import svm, datasets
from sklearn import tree



# Preprocessing the Data
x = np.reshape(np.load('images.npy'), (6500, 784))
y = to_categorical(np.load('labels.npy'))

# Stratified Sampling
testSetSampler = StratifiedShuffleSplit(n_splits=10, test_size=0.25, train_size=0.75)
for x_index, test_index in testSetSampler.split(x, y):
    x_, x_test = x[x_index], x[test_index]
    y_, y_test = y[x_index], y[test_index]

validateSetSampler = StratifiedShuffleSplit(n_splits=10, test_size=0.2, train_size=0.8)
for train_index, validate_index in testSetSampler.split(x_, y_):
    x_train, x_val = x[train_index], x[validate_index]
    y_train, y_val = y[train_index], y[validate_index]

# Model Template
def doFit(xTrain,yTrain,xVal, yVal,  score = 'accuracy'):
    clf = tree.DecisionTreeClassifier(criterion = "entropy", max_depth =6) #max depth
    a = (clf.fit(xTrain,yTrain))
    b = a.tree_
    print("Node Count: " + str(b.node_count))
    with open("cTree.txt", "w") as f:
        f = tree.export_graphviz(a, out_file=f)
    scores = modelsel.cross_val_score(a, xVal, yVal, cv=10, scoring = score)
    print("Cross_Validation Mean Score: " + str(scores.mean()))
    return clf 
# Report Results
clf = doFit(x_train, y_train, x_val, y_val)
y_predict = clf.predict(x_test)
y_predict = np.round(y_predict)
cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_predict.argmax(axis=1))
plot_confusion_matrix(cnf_matrix, classes=['0','1','2','3','4','5','6','7','8','9'],
                      title='Confusion matrix, without normalization')

"""
max_depth   criterion   scores       result
3           entropy     accuracy     0.2256
6           entropy     accuracy     0.6543
6           gini        accuracy     0.5931
10          gini        accuracy     0.6924



"""



"""
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)

cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_predict.argmax(axis=1))
plot_confusion_matrix(cnf_matrix, classes=['0','1','2','3','4','5','6','7','8','9'],
                      title='Confusion matrix, without normalization')
"""
#print(accuracy)