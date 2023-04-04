from __future__ import absolute_import,division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib
from tensorflow.python.tpu import feature_column_v2 as fc
import tensorflow as tf

CSV_COLUMN_NAMES = ['SepalLength','SepalWidth','PetalLength','PetalWidth','Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train = pd.read_csv("./datasets/iris_training.csv",names=CSV_COLUMN_NAMES,header=0)
test = pd.read_csv("./datasets/iris_test.csv",names=CSV_COLUMN_NAMES,header=0)
# print(train.head())

train_y=train.pop('Species')
test_y=test.pop('Species')

def input_fn(features,labels, training=True,batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features),labels))
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)

my_feature_column = []
for key in train.keys():
    my_feature_column.append(tf.feature_column.numeric_column(key=key))
# print(my_feature_column)

classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_column,
    hidden_units=[30,10],
    n_classes=3
)
classifier.train(
    input_fn=lambda: input_fn(train,train_y,training=True),
    steps=5000
)
eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(test,test_y,training=False)
)

import os
os.system('cls' if os.name == 'nt' else 'clear')

print('\nTest set accuracy: ',(eval_result['accuracy']*100))


def input_fn(features, batch_size=256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)
features=['SepalLength','SepalWidth','PetalLength','PetalWidth']
predict={}
print("Please type numeric values as prompted: ")
for feature in features:
    valid=True
    while valid:
        val = input(feature + ": ")
        if not val.isdigit(): valid = False
    predict[feature] = [float(val)]

predictions = classifier.predict(
    input_fn=lambda: input_fn(predict)
)
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    print('Prediction is "{}" ({:.1f}%)'.format(
        SPECIES[class_id],100*probability
    ))