import matplotlib
import recall as recall
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, fbeta_score, average_precision_score, recall_score
from sklearn.metrics import accuracy_score
from confusion_matrix_graph import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from keras import initializers
from sklearn import svm, datasets

# Artificial Neural Networks

# Preprocessing the Data
x = np.reshape(np.load('images.npy'), (6500, 784))
y = to_categorical(np.load('labels.npy'))

# Stratified Sampling
testSetSampler = StratifiedShuffleSplit(n_splits=10, test_size=0.25, train_size=0.75, random_state=0)
for x_index, test_index in testSetSampler.split(x, y):
    x_, x_test = x[x_index], x[test_index]
    y_, y_test = y[x_index], y[test_index]

validateSetSampler = StratifiedShuffleSplit(n_splits=10, test_size=0.2, train_size=0.8, random_state=0)
for train_index, validate_index in testSetSampler.split(x_, y_):
    x_train, x_val = x_[train_index], x_[validate_index]
    y_train, y_val = y_[train_index], y_[validate_index]

# Model Template
model = Sequential() # declare model
model.add(Dense(50, input_shape=(28*28, ), kernel_initializer=initializers.lecun_uniform(seed=None)))  # first layer
model.add(Activation('relu'))

model.add(Dense(30, activation='tanh'))
model.add(Dense(100, activation='relu', use_bias=True, bias_initializer='zeros'))
model.add(Dense(200, activation='relu', use_bias=True, bias_initializer='zeros'))
model.add(Dense(60, activation='tanh'))
model.add(Dense(100, activation='relu', use_bias=True, bias_initializer='zeros'))
model.add(Dense(30, activation='tanh'))

model.add(Dense(10, kernel_initializer='he_normal')) # last layer
model.add(Activation('softmax'))

# Compile Model
model.compile(optimizer='sgd',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

# Train Model
history = model.fit(x_train, y_train,
                    validation_data = (x_val, y_val),
                    epochs=1100,
                    batch_size=400)


# Report Results
print(history.history)

# Graph for accuracy history
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Graph for loss history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Predict the test set
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)

# Misclassified samples
misclassified = np.where(y_predict != y_test)
print("Misclassified:")
print(misclassified)
print("Misclassified no duplicates:")
misclassified = list(set(misclassified[0]))
print(misclassified)

# Display image
print(y_predict.argmax(axis=1)[misclassified[0]])
plt.imshow(np.reshape(x_test[misclassified[0]], (28, 28)))
plt.show()
print(y_predict.argmax(axis=1)[misclassified[1]])
plt.imshow(np.reshape(x_test[misclassified[1]], (28, 28)))
plt.show()
print(y_predict.argmax(axis=1)[misclassified[2]])
plt.imshow(np.reshape(x_test[misclassified[2]], (28, 28)))
plt.show()

# Metrics
print("Metrics for ANNs")
cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_predict.argmax(axis=1))
plot_confusion_matrix(cnf_matrix, classes=['0','1','2','3','4','5','6','7','8','9'],
                      title='Confusion matrix, without normalization')
print("Accuracy score:")
print(accuracy_score(y_test, y_predict))

print("Precision score:")
average_precision = average_precision_score(y_test, y_predict)
print(average_precision)

recall = recall_score(y_test, y_predict, average='micro')
print("Recall score:")
print(recall)

fscore = fbeta_score(y_test, y_predict, beta=1, average='micro')
print("F-score:")
print(fscore)