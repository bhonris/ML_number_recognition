from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#from confusion_matrix_graph import plot_confusion_matrix
import numpy as np
import sklearn.model_selection as modelsel
from sklearn import svm, datasets
from sklearn import tree
from matplotlib import pyplot as plt


# Preprocessing the Data
x_raw = np.load('images.npy')
y_raw = np.load('labels.npy')
x = np.reshape(x_raw, (6500, 784))
y = to_categorical(y_raw)

# Stratified Sampling
testSetSampler = StratifiedShuffleSplit(n_splits=10, test_size=0.25, train_size=0.75)
for x_index, test_index in testSetSampler.split(x, y):
    x_, x_test = x[x_index], x[test_index]
    y_, y_test = y[x_index], y[test_index]

validateSetSampler = StratifiedShuffleSplit(n_splits=10, test_size=0.2, train_size=0.8)
for train_index, validate_index in testSetSampler.split(x_, y_):
    x_train, x_val = x[train_index], x[validate_index]
    y_train, y_val = y[train_index], y[validate_index]
#def engineeredTree(matrix_number):

def showImage(x):
    """Plots an image of the given array
        x -> a numpy array
    """
    plt.imshow(x, cmap='gray', interpolation='nearest')
    
def trainDensity(x, y, row = (0,27), column = (0,27)):
    """ Determines the average density of 1's for each number
    x -> an array of 28x28 matrix representing an image
    y -> an array of labels of the number, between 0-9
    row ->tuple subset of the matrix horizontally
    column -> tuple subset of the matrix vertically
    """
    length = len(x)
    numberDensity = [[],[],[],[],[],[],[],[],[],[]]
    for i in range(length):
        curArray = x[i][row[0]:row[1]+1, column[0]:column[1]+1]
        numberDensity[y[i]] += [curArray.mean()]
    #return [x / length for x in numberDensity]
    meanDensity = [np.array(x).mean() for x in numberDensity]
    devDensity = [np.array(x).std() for x in numberDensity]
    return meanDensity, devDensity

def detectCircles(x):
    """Determines if the number contains circles (0,4,6,8,9)
    x-> an array of 28x28 matrix representing the image
    """
    x_top = x[0:14]
    x_bottom = x[14:28]
    top_x, top_y = np.argwhere(x_top>50).sum(0)/len(np.argwhere(x_top>50))
    bot_x, bot_y = np.argwhere(x_bottom>50).sum(0)/len(np.argwhere(x_bottom>50))

    return (top_y, top_x), (bot_y, bot_x)
    #Split the image
    #Find centroid
def checkForEdges(x, i, j, dx, dy, threshold):
    while(x[i][j] < threshold):
        i += dx
        j += dy
        if(not ((0 < i < 27) and (0 < j < 27))):
            return False #edge detected, not in circle
    return True #white detected, may be in circle
def checkForCircles(x, i, j, threshold):
    up = checkForEdges(x,i,j,1,0, threshold);
    if(not up): return False
    down = checkForEdges(x,i,j,-1,0, threshold);
    if(not down): return False
    diag1 = checkForEdges(x,i,j,1,1, threshold);
    if(not diag1): return False
    diag2 = checkForEdges(x,i,j,-1,-1, threshold);
    if(not diag2): return False
    diag3 = checkForEdges(x,i,j,1,-1, threshold);
    if(not diag3): return False
    diag4 = checkForEdges(x,i,j,-1,1, threshold);
    if(not diag4): return False
    return True

    
def detectCircles2(x, threshold = 190):
    
    for e in range(len(x)):
        #find if there is a change from black to white to black
        curCount = 0
        #black to white
        while((curCount < 27) and x[e][curCount] < threshold):
            curCount += 1
        #print("First: " + str(firstChange))
        #white to black
        while((curCount < 27) and x[e][curCount] >= threshold):
            curCount += 1
        firstChange = curCount
        curCount += 1
        #black to white
        while((curCount < 27) and x[e][curCount] < threshold):
            curCount += 1
        secondChange = curCount
        #print("Second: " + str(firstChange))
        if(secondChange >= 27):
            continue
        else:
            middle = int((firstChange + secondChange)/2)
            if(checkForCircles(x, e, middle, threshold)):
                return 1
    return 0

def trainDetectCircles(x, y, row = (0,27), column = (0,27)):
    length = len(x)
    numberDensity = [[],[],[],[],[],[],[],[],[],[]]
    circleSum = 0
    for i in range(length):
        curArray = x[i][row[0]:row[1]+1, column[0]:column[1]+1]
        ans = detectCircles2(curArray)
        numberDensity[y[i]] += [detectCircles2(curArray)]
    #return [x / length for x in numberDensity]
    meanDensity = [np.array(x).mean() for x in numberDensity]
    devDensity = [np.array(x).std() for x in numberDensity]
    accuracy = (len(numberDensity[0]) + len(numberDensity[6]) 
              + len(numberDensity[8]) + len(numberDensity[0]))/length
    return meanDensity, devDensity, accuracy
        
def straightLines(x):
    """ Get the longest straight line
        x -> an array of 28x28 matrix representing an image
    """
    curMax = 0
    for e in range(12,17):
        curCount = 0
        curWhite = 0
        while((curCount < 27) and x[curCount][e] < 50):
            curCount += 1
        while((x[curCount][e] > 50) and (curCount < 27)):
            curCount += 1
            curWhite += 1
        if(curWhite > curMax): curMax = curWhite
    return curMax

def trainStraightLines(x, y, row = (0,27), column = (0,27)):
    """ Determines the average density of 1's for each number
    x -> an array of 28x28 matrix representing an image
    y -> an array of labels of the number, between 0-9
    row ->tuple subset of the matrix horizontally
    column -> tuple subset of the matrix vertically
    """
    length = len(x)
    numberDensity = [[],[],[],[],[],[],[],[],[],[]]
    for i in range(length):
        curArray = x[i][row[0]:row[1]+1, column[0]:column[1]+1]
        numberDensity[y[i]] += [straightLines(curArray)]
    #return [x / length for x in numberDensity]
    meanDensity = [np.array(x).mean() for x in numberDensity]
    devDensity = [np.array(x).std() for x in numberDensity]
    return meanDensity, devDensity

def engineeredTree(xTrain,yTrain,xVal, yVal):
    """
    1: low density overall (19), small density when looking at left and right, 
        no circles (0.01), long (14),
    2: 
    """
    return 0

# Model Template
def baseLineModel(xTrain,yTrain,xVal, yVal,  score = 'accuracy', max_depth = None):
    clf = tree.DecisionTreeClassifier(max_depth = max_depth) #max depth
    a = (clf.fit(xTrain,yTrain))
    b = a.tree_
    print("Node Count: " + str(b.node_count))
    with open("cTree.txt", "w") as f:
        f = tree.export_graphviz(a, out_file=f)
    scores = modelsel.cross_val_score(clf, xVal, yVal, cv=10, scoring = score)
    print(scores)
    print("Cross_Validation Mean Score: " + str(scores.mean()))

    return clf

# Report Results
##############################################################################
"""Uncomment this portion out to test code"""
#clf = baseLineModel(x_train, y_train, x_val, y_val)
#y_predict = clf.predict(x_test)
#y_predict = np.round(y_predict)
#cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_predict.argmax(axis=1))
#plot_confusion_matrix(cnf_matrix, classes=['0','1','2','3','4','5','6','7','8','9'],
#                      title='Confusion matrix for BaseLine Model')
#scores = accuracy_score(y_test, y_predict)
#print("Cross_Validation Mean Score: " + str(scores.mean()))
##############################################################################



"""
max_depth       nodes   criterion   scores       cross_val
None            895     gini        90.3         0.73
10              681     gini        83.2         0.71
8               375     gini        77.7         0.70

max_features    nodes   criterion   scores       cross_val
auto            1461    gini        88.1         0.66
log2            1971    gini        84.8         0.61
sqrt            1481    gini        87.1         0.65

max_leaf_nodes  nodes   criterion   score        cross_val
100             199     gini        87.3         0.70
300             599     gini        86.2         0.70   
400             799     gini        88.4         0.70 

min_impurity_decrease (MID)
MID            nodes   criterion   score        cross_val
0.00003        875     gini        90.6         0.70
0.0001         583     gini        87.1         0.69
0.0003         195     gini        76.6         0.68

presort        nodes   criterion   score        cross_val
True           901     gini        90.0         0.67
min_impurity_split (MIS)

"""



"""
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)

cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_predict.argmax(axis=1))
plot_confusion_matrix(cnf_matrix, classes=['0','1','2','3','4','5','6','7','8','9'],
                      title='Confusion matrix, without normalization')
"""
#print(accuracy)