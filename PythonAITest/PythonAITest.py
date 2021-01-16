import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow._api.v2.compat.v2.feature_column as fc

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')


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
train_input_fn()

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)
#print(result['accuracy'])
#print(result)
result = list(linear_est.predict(eval_input_fn))
print(dfeval.loc[0])
print(result[1]['probabilities'][1])








#rank1_tensor = tf.Variable(["Test"], tf.string)
#rank2_tensor = tf.Variable([["test", "ok"], ["test", "yes"]], tf.string)
#print(tf.rank(rank1_tensor))
#print(rank2_tensor.shape)


#t = tf.zeros([5,5,5,5])
#t = tf.reshape(t, [125,-1])
#print(t)


#x = [1, 2, 2.5, 3, 4]
#y = [1, 4, 7, 9, 15]
#plt.plot(x, y, 'ro')
#plt.axis([0, 6, 0, 20])
#plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
#plt.show()
###
#print(dftrain.head())

###
###

#print(dftrain["age"], y_train.loc[0])
#print(dftrain.shape)
