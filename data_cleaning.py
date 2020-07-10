#import libraries

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import seaborn as sns
import pickle
from sklearn.utils import shuffle
from tensorflow.keras import layers, models, datasets
#import dataset
with open("traffic-signs-data/train.p", mode='rb') as training_dataset:
    train = pickle.load(training_dataset)
with open("traffic-signs-data/test.p", mode='rb') as testing_dataset:
    test = pickle.load(testing_dataset)
with open("traffic-signs-data/valid.p", mode='rb') as validation_dataset:
    validation = pickle.load(validation_dataset)

#split dataset
X_train, y_train = train['features'], train['labels']
X_validation, y_validation = validation['features'], validation['labels']
X_test, y_test = test['features'], test['labels']
#print(X_train.shape)

#data preparation
X_train, y_train = shuffle(X_train, y_train)

X_train_gray = np.sum(X_train/3, axis=3, keepdims=True)
X_test_gray = np.sum(X_test/3, axis=3, keepdims=True)
X_validation_gray = np.sum(X_validation/3, axis=3, keepdims=True)

#data normalizing
X_train_gray_norm = (X_train_gray-128)/128
X_test_gray_norm = (X_test_gray-128)/128
X_validation_gray_norm = (X_validation_gray-128)/128



#build model
model = tf.keras.models.Sequential()
model.add(layers.Conv2D(16, (5,5), activation='relu', input_shape=(32, 32, 1)))
model.add(layers.AveragePooling2D())

model.add(layers.Conv2D(32, (5,5), activation='relu'))
model.add(layers.AveragePooling2D())

model.add(layers.Flatten())
model.add(layers.Dense(120, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(80, activation='relu'))
model.add(layers.Dense(43, activation='softmax'))

#compile model
model.compile(optimizer = 'Adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#model summary
model.summary()

#model fit
history = model.fit(X_train_gray_norm, y_train, batch_size = 500, nb_epoch=50, verbose = 2, validation_data = (X_validation_gray_norm, y_validation))

#plot model accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(len(acc))
model.save('model.h5')

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Loss')
plt.legend(loc=0)
plt.figure()
plt.show()

#predict model
predicted_classes = model.predict_classes(X_test_gray_norm)
y_true = y_test

L = 7
W = 7
fig, axes = plt.subplots(L, W, figsize = (12, 12))
axes = axes.ravel()

for i in np.arange(0, L*W):
    axes[i].imshow(X_test[i])
    axes[i].set_title('Prediction: {}, True: {} '.format(predicted_classes[i], y_true[i]))
plt.subplots_adjust(wspace=1)

pickle_out= open("model_trained.p","wb")  # wb = WRITE BYTE
pickle.dump(model,pickle_out)
pickle_out.close()