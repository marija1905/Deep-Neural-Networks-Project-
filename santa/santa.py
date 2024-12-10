# This is a sample Python script.
import os

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cv2

main_path = './is that santa/'
img_size = (64, 64)
batch_size = 64

from keras.utils import image_dataset_from_directory

classes = ['santa', 'not a santa']
y = [len(os.listdir('./is that santa/santa/')), len(os.listdir('./is that santa/nota-a-santa/'))]

fig, ax = plt.subplots()
plt.xlabel('klase')
plt.ylabel('broj odbiraka')
ax.bar(classes, y)

print(f' --> Broj odbiraka klasa je: [ Santa = {y[0]}, NotASanta = {y[1]} ]\n --> Klase su balansirane :)')

Xtrain = image_dataset_from_directory(main_path,
                                       subset = 'training',
                                       validation_split=0.2,
                                       image_size=img_size,
                                       batch_size=batch_size,
                                       seed=123)



Xval = image_dataset_from_directory(main_path,
                                    subset='validation',
                                    validation_split=0.2,
                                    image_size=img_size,
                                    batch_size=batch_size,
                                    seed=123)



classes = Xtrain.class_names
print(classes)

N = 10
plt.figure()
for img, lab in Xtrain.take(1):
    for i in range(N):
        plt.subplot(2, int(N/2), i+1)
        plt.imshow(img[i].numpy().astype('uint8'))
        plt.title(classes[lab[i]])
        plt.axis('off')


plt.show()

from keras import layers
from keras import Sequential

data_augmentation = Sequential(
 [
     layers.RandomFlip("horizontal", input_shape=(img_size[0], img_size[1], 3)),
     layers.RandomRotation(0.25),
     layers.RandomZoom(0.1),
 ]
)

N = 10
plt.figure()
for img, lab in Xtrain.take(1):
     plt.title(classes[lab[0]])
     for i in range(N):
         aug_img = data_augmentation(img)
         plt.subplot(2, int(N/2), i+1)
         plt.imshow(aug_img[0].numpy().astype('uint8'))
         plt.axis('off')
plt.show()

from keras import Sequential
from keras import layers
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
num_classes = len(classes)


model = Sequential([
     data_augmentation,
     layers.Rescaling(1./255, input_shape=(64, 64, 3)),
     layers.Conv2D(16, 3, padding='same', activation='relu'),
     layers.MaxPooling2D(),
     layers.Conv2D(32, 3, padding='same', activation='relu'),
     layers.MaxPooling2D(),
     layers.Conv2D(64, 3, padding='same', activation='relu'),
     layers.MaxPooling2D(),
     layers.Dropout(0.2),
     layers.Flatten(),
     layers.Dense(128, activation='relu'),
     layers.Dense(num_classes, activation='softmax')
])
model.summary()
model.compile(Adam(learning_rate=0.001),
     loss=SparseCategoricalCrossentropy(),
     metrics=['accuracy'])

from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_accuracy', patience=20, verbose=0)
history = model.fit(Xtrain, epochs=50, validation_data=Xval, callbacks=[es], verbose=1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.subplot(121)
plt.plot(acc)
plt.plot(val_acc)
plt.title('Accuracy')
plt.subplot(122)
plt.plot(loss)
plt.plot(val_loss)
plt.title('Loss')
plt.show()


labels_val = np.array([])
pred_val = np.array([])

labels_train = np.array([])
pred_train = np.array([])

#matrica konfuzije za test skup
bad_pred=[]
good_pred=[]
i = 0
for img, lab in Xval:
     labels_val = np.append(labels_val, lab)
     pred_val = np.append(pred_val, np.argmax(model.predict(img, verbose=0), axis=1))
     if(labels_val[i] == pred_val[i]): good_pred.append(img)
     else:
         bad_pred.append(img)

     i = i + 1



if (bad_pred == []):
    print("nema lose pred")
else:
    plt.figure()
    plt.imshow((bad_pred[0])[0].numpy().astype('uint8'))
    plt.title("losa predikcija")
    plt.axis("off")
    plt.show()

plt.figure()
plt.imshow((good_pred[0])[0].numpy().astype('uint8'))
plt.title("dobra predikcija")
plt.axis("off")
plt.show()




from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(labels_val, pred_val, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
cmDisplay.plot()
plt.title('Matrica konfuzije test skupa')
plt.show()


#matrica konfuzije za train skup
for img, lab in Xtrain:
     labels_train = np.append(labels_train, lab)
     pred_train = np.append(pred_train, np.argmax(model.predict(img, verbose=0), axis=1))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(labels_train, pred_train, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
cmDisplay.plot()
plt.title('Matrica konfuzije train skupa')
plt.show()

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


def make_model(hp):
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255, input_shape=(64, 64, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(Adam(learning_rate=lr),
                  loss=SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model

import keras_tuner as kt

tuner = kt.RandomSearch(make_model, objective='val_accuracy', overwrite = True, max_trials=10)
tuner.search(Xtrain, epochs=20, validation_data= Xval, callbacks=[es])

best_model = tuner.get_best_models()
best_hyperparam = tuner.get_best_hyperparameters(num_trials=1)[0]
print('Optimalna konstanta obučavanja: ', best_hyperparam['learning_rate'])

