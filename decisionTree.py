from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, fbeta_score, average_precision_score, recall_score
from sklearn.metrics import accuracy_score
from confusion_matrix_graph import plot_confusion_matrix
import numpy as np
import sklearn.model_selection as modelsel
from sklearn import svm, datasets
from sklearn import tree
from matplotlib import pyplot as plt
from collections import Counter
import pandas as pd


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
    x_train, x_val = x_[train_index], x_[validate_index]
    y_train, y_val = y_[train_index], y_[validate_index]
#def engineeredTree(matrix_number):

def showImage(x):
    """Plots an image of the given array
        x -> a numpy array
    """
    plt.imshow(x, cmap='gray', interpolation='nearest')
    
def trainDensity(x, y, row = (0,27), column = (0,27), stdDev = False):
    """ Determines the average density of 1's for each number
    x -> an array of 28x28 matrix representing an image
    y -> an array of labels of the number, between 0-9
    row ->tuple subset of the matrix horizontally
    column -> tuple subset of the matrix vertically
    """
    length = len(x)
    print("Length: " + length)
    numberDensity = [[],[],[],[],[],[],[],[],[],[]]
    for i in range(length):
        curArray = x[i][row[0]:row[1]+1, column[0]:column[1]+1]
        numberDensity[y[i]] += [curArray.mean()]
    #return [x / length for x in numberDensity]
    meanDensity = [np.array(x).mean() for x in numberDensity]
    devDensity = [np.array(x).std() for x in numberDensity]
    if(stdDev):
        return meanDensity, devDensity
    else:
        return meanDensity

def getDensity(x, y, row = (0,27), column = (0,27)):
    """ Determines the average density of 1's for each number
    x -> an array of 28x28 matrix representing an image
    y -> an array of labels of the number, between 0-9
    row ->tuple subset of the matrix horizontally
    column -> tuple subset of the matrix vertically
    """
    length = len(x)
    retArray = []
    for i in range(length):
        curArray = x[i][row[0]:row[1]+1, column[0]:column[1]+1]
        retArray += [curArray.mean()]
    #return [x / length for x in numberDensity]
    return retArray

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

def trainDetectCircles(x, y, row = (0,27), column = (0,27), stdDev = False):
    length = len(x)
    numberDensity = [[],[],[],[],[],[],[],[],[],[]]
    for i in range(length):
        curArray = x[i][row[0]:row[1]+1, column[0]:column[1]+1]
        ans = detectCircles2(curArray)
        numberDensity[y[i]] += [detectCircles2(curArray)]
    #return [x / length for x in numberDensity]
    meanDensity = [Counter(x) for x in numberDensity]
    devDensity = [np.array(x).std() for x in numberDensity]
    accuracy = (len(numberDensity[0]) + len(numberDensity[6]) 
              + len(numberDensity[8]) + len(numberDensity[0]))/length
    
    if(stdDev):
        return meanDensity, devDensity
    else:
        return meanDensity
        
def getCircles(x, y, row = (0,27), column = (0,27)):
    length = len(x)
    retArray = []
    for i in range(length):
        curArray = x[i][row[0]:row[1]+1, column[0]:column[1]+1]
        retArray += [detectCircles2(curArray)]
    return retArray

    
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

def straightLinesHorizontal(x):
    """ Get the longest straight line
        x -> an array of 28x28 matrix representing an image
    """
    curMax = 0
    for e in range(6,22):
        curCount = 0
        curWhite = 0
        while((curCount < 27) and x[e][curCount] < 50):
            curCount += 1
        while((x[e][curCount] > 50) and (curCount < 27)):
            curCount += 1
            curWhite += 1
        if(curWhite > curMax): curMax = curWhite
    return curMax

def trainStraightLines(x, y, row = (0,27), column = (0,27), stdDev = False):
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
    if(stdDev):
        return meanDensity, devDensity
    else:
        return meanDensity

def getStraightLines(x, y, row = (0,27), column = (0,27)):
    """ Determines the average density of 1's for each number
    x -> an array of 28x28 matrix representing an image
    y -> an array of labels of the number, between 0-9
    row ->tuple subset of the matrix horizontally
    column -> tuple subset of the matrix vertically
    """
    length = len(x)
    retArray = []
    for i in range(length):
        curArray = x[i][row[0]:row[1]+1, column[0]:column[1]+1]
        retArray+= [straightLines(curArray)]
    return retArray
        

def trainStraightHorizontalLines(x, y, row = (0,27), column = (0,27), stdDev = False):
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
        numberDensity[y[i]] += [straightLinesHorizontal(curArray)]
    #return [x / length for x in numberDensity]
    meanDensity = [np.array(x).mean() for x in numberDensity]
    devDensity = [np.array(x).std() for x in numberDensity]
    if(stdDev):
        return meanDensity, devDensity
    else:
        return meanDensity

def getStraightHorizontalLines(x, y, row = (0,27), column = (0,27)):
    """ Determines the average density of 1's for each number
    x -> an array of 28x28 matrix representing an image
    y -> an array of labels of the number, between 0-9
    row ->tuple subset of the matrix horizontally
    column -> tuple subset of the matrix vertically
    """
    length = len(x)
    retArray = []
    for i in range(length):
        curArray = x[i][row[0]:row[1]+1, column[0]:column[1]+1]
        retArray += [straightLinesHorizontal(curArray)]
    return retArray

def centroid(x):
    spread = 0
    lengthSum = 0
    for i in range(len(x)):
        for j in range(len(x[i])):
            if(x[i][j] > 10):
                curLength = np.linalg.norm([(i-14),(j-14)])
                spread += x[i][j]
                lengthSum += curLength
    return spread/lengthSum
            
def trainCentroid(x, y, row = (0,27), column = (0,27), stdDev = False):
    length = len(x)
    numberDensity = [[],[],[],[],[],[],[],[],[],[]]
    for i in range(length):
        curArray = x[i][row[0]:row[1]+1, column[0]:column[1]+1]
        numberDensity[y[i]] += [centroid(curArray)]
    #return [x / length for x in numberDensity]
    meanDensity = [np.array(x).mean() for x in numberDensity]
    if(stdDev):
        devDensity = [np.array(x).std() for x in numberDensity]
        return meanDensity, devDensity
    else:
        return meanDensity
    
def getCentroid(x, y, row = (0,27), column = (0,27)):
    length = len(x)
    numberDensity = []
    for i in range(length):
        curArray = x[i][row[0]:row[1]+1, column[0]:column[1]+1]
        numberDensity += [centroid(curArray)]
    #return [x / length for x in numberDensity]
#    meanDensity = [np.array(x).mean() for x in numberDensity]
    return numberDensity
    


# Model Template
def baseLineModel(xTrain,yTrain,xVal, yVal,  
                  criterion='gini', splitter = 'best', max_depth = None,
                  max_leaf_nodes=None, min_impurity_decrease = 0.0,
                  max_features = None):
    clf = tree.DecisionTreeClassifier() #max depth
    #b = a.tree_
    #print("Node Count: " + str(b.node_count))
    #with open("cTree.txt", "w") as f:
    #    f = tree.export_graphviz(a, out_file=f)
#    scores = modelsel.cross_val_score(clf, xTrain, yTrain, cv=10, scoring = score)
#    print(scores)
#    print("Cross_Validation Mean Score: " + str(scores.mean()))
    return clf

# Report Results
###############################################################################
"""Uncomment this portion out to test code"""
def doEvaluation(xTrain, yTrain, xTest, yTest, criterion='gini', splitter = 'best', max_depth = None,
                  max_leaf_nodes=None, min_impurity_decrease = 0.0,
                  max_features = None, title = "Confusion Matrix", presort = False):
    clf = tree.DecisionTreeClassifier(criterion = criterion, splitter = splitter,
                                      max_depth = max_depth, max_leaf_nodes = max_leaf_nodes,
                                      min_impurity_decrease = min_impurity_decrease,
                                      max_features = max_features, presort = presort)
    clf.fit(xTrain, yTrain)
    y_predict = clf.predict(xTest)
    y_predict = np.round(y_predict)
    b = clf.tree_
    with open("cTree.txt", "w") as f:
        f = tree.export_graphviz(clf, out_file=f)
    return fullEvaluation(clf, xTrain, yTrain, yTest, y_predict, title)
    

###############################################################################
def fullEvaluation(clf, xTrain, yTrain, y_test, predTest, title):
    """
    ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'log_loss', 
    'mean_absolute_error', 'mean_squared_error', 'precision', 'r2', 'recall', 
    'roc_auc']
    """
    normalScore = []
    crossValScore = []
    
    print("Accuracy on test set:")
    acc_score = accuracy_score(y_test, predTest)
    normalScore += [acc_score]
    print(acc_score)
#    acc_score_xval = modelsel.cross_val_score(clf, xTrain, yTrain, cv=10, scoring = 'accuracy').mean()
#    crossValScore += [acc_score_xval]
#    print(acc_score_xval)
    
    print("Precision score:")
 #   print(predTest)
    average_precision_knn = average_precision_score(y_test, predTest)
    normalScore += [average_precision_knn]
#    print(average_precision_knn)
#    average_precision_knn_xval = modelsel.cross_val_score(clf, xTrain, yTrain, cv=10, scoring = 'precision').mean()
#    crossValScore += [average_precision_knn_xval]
#    print(average_precision_knn_xval)

    print("Recall score:")
    recall_knn = recall_score(y_test, predTest, average='micro')
    normalScore += [recall_knn]
    print(recall_knn)
#    theOneWhoKnocks = modelsel.cross_val_score(clf, xTrain, yTrain, cv=10, scoring = 'recall').mean()
#    crossValScore += [theOneWhoKnocks]
#    print(theOneWhoKnocks)

    
    print("F-score:")
    fscore_knn = fbeta_score(y_test, predTest, beta=1, average='micro')
    normalScore += [fscore_knn]
    print(fscore_knn)
#    fscore_knn_xval = modelsel.cross_val_score(clf, xTrain, yTrain, cv=10, scoring = 'f1').mean()
#    crossValScore += [fscore_knn_xval]
#    print(fscore_knn_xval)

   # cnf_matrix = confusion_matrix(y_test, predTest)
    #plot_confusion_matrix(cnf_matrix, classes=['0','1','2','3','4','5','6','7','8','9'],
     #                 title=title)
    return normalScore
#Feature Extraction Trees
###############################################################################
def featureExtraction(x_val,y_val):
    print("Starting feature extraction")
    circleSet = getCircles(x_val,y_val)
    print("Finished Circle Set")
    fullDensitySet = getDensity(x_val, y_val)
    print("Finished Full Density")
    density1 = getDensity(x_val, y_val, row=(0,14))
    print("Finished density1")
    density2 = getDensity(x_val, y_val, row=(14,27))
    print("Finished density2")
    vertical = getStraightLines(x_val, y_val)
    print("Finished vertical")
    horizontal = getStraightHorizontalLines(x_val, y_val)
    print("Finished horizontal")
#    centroid = getCentroid(x_val,y_val)
    density3 = getDensity(x_val, y_val, row=(0,14),column=(0,14))
    density4 = getDensity(x_val, y_val, row=(0,14),column=(14,27))
    density5 = getDensity(x_val, y_val, row=(14,27),column=(0,14))
    density6 = getDensity(x_val, y_val, row=(14,27),column=(14,27))

    df = pd.DataFrame()
    df['isCircle'] = circleSet
    df['full_density'] = fullDensitySet
    df['densityTop'] = density1
    df['densityBottom'] = density2
    df['vertical'] = vertical
    df['horizontal'] = horizontal
    df['density1'] = density3
    df['density2'] = density4
    df['density3'] = density5
    df['density4'] = density6
#    df['centroid'] = centroid
    return df

###############################################################################
normalDF = pd.DataFrame(columns = ['accuracy', 'precision', 'recall', 'f-score'])
#crossValDF = pd.DataFrame(columns = ['accuracy', 'precision', 'recall', 'f-score'])

print("Default Tree")
x = doEvaluation(x_train, y_train, x_test, y_test, title = "Default Tree")
normalDF.loc[0] = x

print("Entropy")
y_predict = doEvaluation(x_train, y_train, x_test, y_test, criterion = 'entropy', title = "Using Entropy")
normalDF.loc[1] = x
#crossValDF.loc[1]= y

print("Max Depth: 8")
x = doEvaluation(x_train, y_train,x_test, y_test, criterion = 'entropy', max_depth = 8, title = "Max Depth 8 ")
normalDF.loc[2] = x
#crossValDF.loc[2]= y

print("Max Depth: 10")
x = doEvaluation(x_train, y_train, x_test, y_test,criterion = 'entropy', max_depth = 10, title = "Max Depth 10")
normalDF.loc[3] = x
#crossValDF.loc[3]= y

print("Max Depth: 15")
x = doEvaluation(x_train, y_train, x_test, y_test, criterion = 'entropy', max_depth = 15, title = "Max Depth 15")
normalDF.loc[4] = x
#crossValDF.loc[4]= y

print("Max Features: auto")
x = doEvaluation(x_train, y_train, x_test, y_test,criterion = 'entropy', max_features="auto", title = "Max Features auto")
normalDF.loc[5] = x

print("Max Features: log2")
x = doEvaluation(x_train, y_train, x_test, y_test,criterion = 'entropy', max_features="log2", title = "Max Features log2")
normalDF.loc[6] = x

print("Max Features: sqrt")
x = doEvaluation(x_train, y_train, x_test, y_test,criterion = 'entropy', max_features="sqrt", title = "Max Features sqrt")
normalDF.loc[7] = x

print("Max Leaf Nodes: 100")
x = doEvaluation(x_train, y_train, x_test, y_test,criterion = 'entropy', max_leaf_nodes=100, title = "Max Leaf Nodes 100")
normalDF.loc[8] = x

print("Max Leaf Nodes: 200")
x = doEvaluation(x_train, y_train,x_test, y_test,criterion = 'entropy', max_leaf_nodes=100, title = "Max Leaf Nodes 200")
normalDF.loc[9] = x

print("Max Leaf Nodes: 300")
x = doEvaluation(x_train, y_train, x_test, y_test,criterion = 'entropy', max_leaf_nodes=100, title = "Max Leaf Nodes 300")
normalDF.loc[10] = x

print("Max Leaf Nodes: 400")
x = doEvaluation(x_train, y_train, x_test, y_test,criterion = 'entropy', max_leaf_nodes=100, title = "Max Leaf Nodes 400")
normalDF.loc[11] = x

print("Min Impurity Decrease:  0.00003")
x = doEvaluation(x_train, y_train, x_test, y_test,criterion = 'entropy', min_impurity_decrease=0.00003, title = "Min Impurity Decrease 0.00003")
normalDF.loc[12] = x

print("Min Impurity Decrease:  0.00010")
x = doEvaluation(x_train, y_train, x_test, y_test,criterion = 'entropy', min_impurity_decrease=0.00010, title = "Min Impurity Decrease 0.00010")
normalDF.loc[13] = x

print("Min Impurity Decrease:  0.00030")
x = doEvaluation(x_train, y_train, x_test, y_test,criterion = 'entropy', min_impurity_decrease=0.00030, title = "Min Impurity Decrease 0.00030")
normalDF.loc[14] = x

print("PreSort: True")
x = doEvaluation(x_train, y_train, x_test, y_test,criterion = 'entropy', presort= True, title = "Presort")
normalDF.loc[15] = x
normalDF.index = ["Default Tree", "Entropy", 
                  "Max Depth: 8", "Max Depth: 10", "Max Depth: 15", 
                  "Max Features: auto", "Max Features: log2", "Max Features: sqrt",
                  "Max Leaf Nodes: 100", "Max Leaf Nodes: 200", "Max Leaf Nodes: 300", "Max Leaf Nodes: 400",
                  "Min Impurity Decrease:  0.00003", "Min Impurity Decrease:  0.0001", "Min Impurity Decrease:  0.0003",
                  "Presort"]


###############################################################################



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

"""



"""
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)

cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_predict.argmax(axis=1))
plot_confusion_matrix(cnf_matrix, classes=['0','1','2','3','4','5','6','7','8','9'],
                      title='Confusion matrix, without normalization')
"""
#print(accuracy)

# Stratified Sampling

#testSetSampler = StratifiedShuffleSplit(n_splits=10, test_size=0.25, train_size=0.75)
#for x_index, test_index in testSetSampler.split(x_raw, y_raw):
#    x_, x_test = x_raw[x_index], x_raw[test_index]
#    y_, y_test = y_raw[x_index], y_raw[test_index]
#
#validateSetSampler = StratifiedShuffleSplit(n_splits=10, test_size=0.2, train_size=0.8)
#for train_index, validate_index in testSetSampler.split(x_, y_):
#    x_train, x_val = x_[train_index], x_[validate_index]
#    y_train, y_val = y_[train_index], y_[validate_index]
#
#df_train    = featureExtraction(x_train, y_train)
#df_validate = featureExtraction(x_val  , y_val)
#df_test     = featureExtraction(x_test , y_test)
#
#y_predict = doEvaluation(df_train,y_train, df_test, y_test, criterion = 'entropy')

#errorList = []
#for e in range(len(y_predict)):
#    if(y_predict[e] !=  y_test[e]):
#        errorList += [e]
#y_predict = clf.predict(df_test)
#y_predict = np.round(y_predict)
#cnf_matrix = confusion_matrix(y_test, y_predict)
#plot_confusion_matrix(cnf_matrix, classes=['0','1','2','3','4','5','6','7','8','9'],
#                      title='Confusion matrix for Feature')
#scores = accuracy_score(y_test, y_predict)

