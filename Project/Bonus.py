import os
import cv2
import random

import matplotlib.pyplot
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow import keras

#Normalize dataset using the following properties
#calculate average for all images,
#subtract this averages from each image.
#Divide each image by 255
DATADIR = "C:/Users/Rocker/Desktop/4thYear/ML/Project/Dataset/"
imgSet = []
labelsSet = []
counter = 0

for folder in os.listdir(DATADIR):
    # Path to the folder
    path = os.path.join(DATADIR, folder)
    # Loop over the folder and get average of all images inside of it
    sum = 0
    for file in os.listdir(path):
        # Path to the image
        img_path = os.path.join(path, file)
        # Read the image
        img = cv2.imread(img_path)
        #resize the image to 100x100
        img = cv2.resize(img, (100, 100))
        sum += img
    average = sum/len(os.listdir(path))
    # Loop over the folder again and subtract average from each image
    for file in os.listdir(path):
        # Path to the image
        img_path = os.path.join(path, file)
        # Read the image
        img = cv2.imread(img_path)
        #resize the image to 100x100
        img = cv2.resize(img, (100, 100))
        img = img - average
        img = img/255.0
        # Append the image to the list
        #[[100*100], [100*100], [100*100]]
        imgSet.append(img)
        labelsSet.append(counter)
    counter+=1



random.seed(42)
random.shuffle(imgSet)
random.seed(42)
random.shuffle(labelsSet)


#Split imgSet and labels into train and test
X_train = imgSet[:int(len(imgSet)*0.8)]
X_train = np.array(X_train)
X_test = imgSet[int(len(imgSet)*0.8):]
X_test = np.array(X_test)
y_train = labelsSet[:int(len(labelsSet)*0.8)]
y_train = np.array(y_train)
y_test = labelsSet[int(len(labelsSet)*0.8):]
y_test = np.array(y_test)


# build the convolutional neural network model
model = tf.keras.Sequential(
    [
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation="relu",input_shape=(100, 100, 3)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),

    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
    ]
    )

model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])


from sklearn.model_selection import KFold
kf = KFold(n_splits=5, random_state=None, shuffle=False)
for train_index, test_index in kf.split(imgSet):
    X_train2, X_test2 = np.array(imgSet)[train_index], np.array(imgSet)[test_index]
    y_train2, y_test2 = np.array(labelsSet)[train_index], np.array(labelsSet)[test_index]
    model.fit(X_train2, y_train2, epochs=5)
    test_loss, test_acc = model.evaluate(X_test2, y_test2)
    print("Accuracy: " + str(test_acc))