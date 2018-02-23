from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
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
    x_train, x_val = x[train_index], x[validate_index]
    y_train, y_val = y[train_index], y[validate_index]

# Model Template
model = Sequential() # declare model
model.add(Dense(60, input_shape=(28*28, ), kernel_initializer=initializers.glorot_uniform(40)))  # first layer
model.add(Activation('relu'))

model.add(Dense(60, activation='selu', kernel_initializer=initializers.lecun_normal()))
model.add(Dense(60, activation='tanh', use_bias=True, kernel_initializer=initializers.glorot_uniform(40), bias_initializer='zeros'))
model.add(Dense(60, activation='tanh', use_bias=True, kernel_initializer=initializers.glorot_uniform(40), bias_initializer='zeros'))
model.add(Dense(60, activation='relu', use_bias=True, kernel_initializer=initializers.glorot_uniform(40), bias_initializer='zeros'))
model.add(Dense(60, activation='selu', kernel_initializer=initializers.lecun_normal()))

model.add(Dense(10, kernel_initializer='he_normal')) # last layer
model.add(Activation('softmax'))

# Compile Model
model.compile(optimizer='sgd',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

# Train Model
history = model.fit(x_train, y_train,
                    validation_data = (x_val, y_val), 
                    epochs=1400,
                    batch_size=512)

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

# Metrics
cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_predict.argmax(axis=1))
plot_confusion_matrix(cnf_matrix, classes=['0','1','2','3','4','5','6','7','8','9'],
                      title='Confusion matrix, without normalization')

print(accuracy_score(y_test, y_predict))


# K-Nearest Neighbor Classification
neigh = KNeighborsClassifier(n_neighbors=3)

#Fitting the model
historyNeigh = neigh.fit(x_train, y_train) # fit (X, y) fits the model using X as training data and y as target value

# Predict the response on validation set
pred = neigh.predict(x_val)
pred = np.round(pred)

# Evaluate Accuracy on validation set
print("Accuracy Validation")
print(accuracy_score(y_val, pred))

# Predict the response on test set
predTest = neigh.predict(x_test)
predTest = np.round(predTest)

# Evaluate Accuracy on test set
print("Accuracy Test")
print(accuracy_score(y_test, predTest))

# Confusion matrix and other metrics
print("Confusion matrix")
cnf_matrix_KN = confusion_matrix(y_test.argmax(axis=1), predTest.argmax(axis=1))
plot_confusion_matrix(cnf_matrix_KN, classes=['0','1','2','3','4','5','6','7','8','9'],
                      title='Confusion matrix, without normalization')

