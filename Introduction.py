# import kerea mnist dataset
import os
from keras.datasets import mnist
from keras.models import model_from_json

# load the data

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape the data

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# convert the data to float

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# normalize the data

x_train =x_train/ 255
x_test =x_test/ 255

# one hot encoding

from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# define the model

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200)

# evaluate the model

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# save the model

model_structure= model.to_json()
file=open("model_structure.json","w")
file.write(model_structure)
file.close()

model.save_weights("model_weights.h5")

# load the model

file=open("model_structure.json")
model_structure=file.read()
model=model_from_json(model_structure)
model.load_weights("model_weights.h5")

# predict

import numpy as np

class_labels=[
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    
]
list_of_images=np.expand_dims(x_test,axis=0)
results=model.predict(list_of_images)
single_result=results[0]
most_likely_class_index=int(np.argmax(single_result))
class_likelihood=single_result[most_likely_class_index]
class_label=class_labels[most_likely_class_index]
print("Image = {}-Likelihood: {:2f}".format(class_label, class_likelihood))