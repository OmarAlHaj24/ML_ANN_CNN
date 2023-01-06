
import os
import cv2
import random

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
from tensorflow import keras
from keras import backend as back
from sklearn import svm, __all__
from sklearn.model_selection import GridSearchCV, train_test_split


def recall_m(y_true, y_pred):
    true_positives = back.sum(back.round(back.clip(y_true * y_pred, 0, 1)))
    possible_positives = back.sum(back.round(back.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + back.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = back.sum(back.round(back.clip(y_true * y_pred, 0, 1)))
    predicted_positives = back.sum(back.round(back.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + back.epsilon())
    return precision


# Name of the directory containing the dataset

DATADIR = "C:/Users/Rocker/Desktop/4thYear/ML/Project/Dataset/"

# Loop over DATADIR
imgSet = []
labelsSet = []
counter = 0

for folder in os.listdir(DATADIR):
    path = os.path.join(DATADIR, folder)
    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (100, 100))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray/255.0
        #[[100*100], [100*100], [100*100]]
        imgSet.append(gray)
        labelsSet.append(counter)
    counter+=1

#Shuffle imSet and labelSet with same seed
random.seed(42)
random.shuffle(imgSet)
random.seed(42)
random.shuffle(labelsSet)


#Split imgSet and labels into train and test
print(len(imgSet))
X_train = imgSet[:int(len(imgSet)*0.8)]
X_train = np.array(X_train)
X_test = imgSet[int(len(imgSet)*0.8):]
X_test = np.array(X_test)
y_train = labelsSet[:int(len(labelsSet)*0.8)]
y_train = np.array(y_train)
y_test = labelsSet[int(len(labelsSet)*0.8):]
y_test = np.array(y_test)



model1 = keras.Sequential([
    keras.layers.Flatten(input_shape=(100, 100)),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model1.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy", recall_m, precision_m])


print("********************** MODEL 1 **********************")
# Train using cross validation
from sklearn.model_selection import KFold, train_test_split

kf = KFold(n_splits=5, random_state=None, shuffle=False)
for train_index, test_index in kf.split(imgSet):
    X_train2, X_test2 = np.array(imgSet)[train_index], np.array(imgSet)[test_index]
    y_train2, y_test2 = np.array(labelsSet)[train_index], np.array(labelsSet)[test_index]
    model1.fit(X_train2, y_train2, epochs=5)
    test_loss, test_acc, recall, precision = model1.evaluate(X_test2, y_test2)
    print("Accuracy: " + str(test_acc))






print("********************** MODEL 2 **********************")

model2 = keras.Sequential([
    keras.layers.Flatten(input_shape=(100, 100)),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model2.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy", recall_m, precision_m])

#Train using cross validation
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, random_state=None, shuffle=False)
for train_index, test_index in kf.split(imgSet):
    X_train2, X_test2 = np.array(imgSet)[train_index], np.array(imgSet)[test_index]
    y_train2, y_test2 = np.array(labelsSet)[train_index], np.array(labelsSet)[test_index]
    model2.fit(X_train2, y_train2, epochs=5)
    test_loss, test_acc, recall, precision = model2.evaluate(X_test2, y_test2)
    print("Accuracy: " + str(test_acc))





# ***************************** New Model *****************************
imgSet = []
labelsSet = []
counter = 0

for folder in os.listdir(DATADIR):
    path = os.path.join(DATADIR, folder)
    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (100, 100))
        imgSet.append(img.flatten())
        labelsSet.append(counter)
    counter+=1

npImgSet=np.array(imgSet)
npLabelsSet=np.array(labelsSet)
# ***************************** Process Images *****************************
dataF = pd.DataFrame(npImgSet)
dataFY = pd.DataFrame(npLabelsSet)


# ***************************** Shuffle and Split *****************************
x_train,x_test,y_train,y_test = train_test_split(dataF, dataFY, test_size=0.2)
# ***************************** Train & Evaluate Model *****************************
from sklearn import tree
clf = tree.DecisionTreeClassifier()
print("Fitting Model")
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
print(accuracy)