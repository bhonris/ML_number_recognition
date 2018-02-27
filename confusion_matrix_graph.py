import matplotlib.pyplot as plt
import itertools
import numpy as np

def plot_confusion_matrix(cMat, classes,
                          title='Confusion matrix'):

    plt.imshow(cMat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cMat.max() / 2.
    for i, j in itertools.product(range(cMat.shape[0]), range(cMat.shape[1])):
        plt.text(j, i, format(cMat[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cMat[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
