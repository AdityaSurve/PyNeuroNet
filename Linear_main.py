from __future__ import absolute_import,division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib
from tensorflow.python.tpu import feature_column_v2 as fc
import tensorflow as tf

dftrain = pd.read_csv('datasets/train.csv')
dfeval = pd.read_csv('datasets/eval.csv')
print(dftrain.head())
print(dfeval.head())
y_train=dftrain.pop('survived')
y_eval=dfeval.pop('survived')
print(dftrain.head())
print(dfeval.head())
print(y_train.head())
print(y_eval.head())

dftrain.age.hist(bins=20)
dftrain.sex.value_counts().plot(kind='barh')
dftrain['class'].value_counts().plot(kind='pie')
pd.concat([dftrain,y_train],axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
plt.show()

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df)) 
    if shuffle:
      ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).repeat(num_epochs) 
    return ds 
  return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn) 

import os
os.system('cls' if os.name == 'nt' else 'clear')
print(result )
print(result['accuracy'])  
result = list(linear_est.predict(eval_input_fn))

print(dfeval.loc[5])
print(y_eval.loc[5])
print(result[5]['probabilities'][1]*100," %")