import argparse
import logging
import json
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import datetime

from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from numpy.random import seed

import tensorflow as tf
tf.random.set_seed(221)
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

logging.getLogger().setLevel(logging.INFO)

def make_datasets_unbatched():
  df = pd.read_csv("https://raw.githubusercontent.com/charlesa101/KubeflowUseCases/draft/Hotel%20bookings%20demand/hotel_bookings.csv?token=AQEY3DFJCQCART4U4QXWS6TA6HBYM")
  df.head()

  # Examine the columns with missing values
  df_null = df.isnull().sum()
  df_null[df_null.values > 0].sort_values(ascending=False)

  # drop missing values
  df = df.drop(['company', 'agent'], axis=1)
  df = df.dropna(subset=['country', 'children'], axis=0)
  df = df.reset_index(drop=True)

  # Converting wrong datatype columns to correct type (object to datetime)
  df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])

  # Converting string month to numerical one (Dec = 12, Jan = 1, etc.)
  datetime_object = df['arrival_date_month'].str[0:3]
  month_number = np.zeros(len(datetime_object))

  # Creating a new column based on numerical representation of the months
  for i in range(0, len(datetime_object)):
    datetime_object[i] = datetime.datetime.strptime(datetime_object[i], "%b")
    month_number[i] = datetime_object[i].month

  # Float to integer conversion
  month_number = pd.DataFrame(month_number).astype(int)

  # 3 columns merged into one
  df['arrival_date'] = df['arrival_date_year'].map(str) + '-' + month_number[0].map(str) + '-' \
                       + df['arrival_date_day_of_month'].map(str)
  # Dropping already used columns
  df = df.drop(['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month',
                  'arrival_date_week_number'], axis=1)
  # convert the newly created arrival_date feature to datetime type
  df['arrival_date'] = pd.to_datetime(df['arrival_date'])

  # Calculating total guests by combining adults, children and babies columns
  df['total guests'] = df['adults'] + df['children'] + df['babies']

  # drop data points that include zero Total Guests
  df = df[df['total guests'] != 0]

  # Total Number of Days Stayed
  df['total stays'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']

  dataNoCancel = df[df['is_canceled'] == 0]
  dataNoCancel = dataNoCancel.reset_index(drop=True)

  df = df.drop(['adults', 'children', 'babies', 'stays_in_weekend_nights', 'stays_in_week_nights', 'arrival_date', 'reservation_status_date'], axis=1)

  # Categorical variables preprocessing with label encoding
  list_1 = list(df.columns)
  cate_list=[]
  for i in list_1:
    if df[i].dtype=='object':
      cate_list.append(i)
  # transform the categorical variables with label encoder
  le = LabelEncoder()
  for i in cate_list:
    df[i] = le.fit_transform(df[i])
    
  # split the data into dependent variables and independent variable
  X = df.drop(['hotel'],axis=1)
  y = df.hotel

  # split the data into training and test set
  X_train,X_test,y_train,y_test = tts(X,y,random_state=36,test_size=0.3)

  # scale the data
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.fit_transform(X_test)

  train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
  test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
  train = train_dataset.cache().shuffle(2000).repeat()
  return train, test_dataset

def model(args):
  seed(1)
  model = Sequential()
  model.add(Dense(10, activation='relu', input_dim=21))
  #model.add(BatchNormalization())
  model.add(Dense(10, activation='relu'))
  #model.add(Dropout(0.2))
  model.add(Dense(1, activation='sigmoid'))
    
  model.summary()
  opt = args.optimizer
  model.compile(optimizer=opt,
                loss='binary_crossentropy',
                metrics=['accuracy'])
  tf.keras.backend.set_value(model.optimizer.learning_rate, args.learning_rate)
  return model

def main(args):
  # MultiWorkerMirroredStrategy creates copies of all variables in the model's
  # layers on each device across all workers

  strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
  communication=tf.distribute.experimental.CollectiveCommunication.AUTO)
  logging.debug(f"num_replicas_in_sync: {strategy.num_replicas_in_sync}")
  BATCH_SIZE_PER_REPLICA = args.batch_size
  BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    
  # Datasets need to be created after instantiation of `MultiWorkerMirroredStrategy`
  train_dataset, test_dataset = make_datasets_unbatched()
  train_dataset = train_dataset.batch(batch_size=BATCH_SIZE)
  test_dataset = test_dataset.batch(batch_size=BATCH_SIZE)

  # See: https://www.tensorflow.org/api_docs/python/tf/data/experimental/DistributeOptions
  options = tf.data.Options()
  options.experimental_distribute.auto_shard_policy = \
  tf.data.experimental.AutoShardPolicy.DATA
    
  train_datasets_sharded = train_dataset.with_options(options)
  test_dataset_sharded = test_dataset.with_options(options)

  with strategy.scope():
    # Model building/compiling need to be within `strategy.scope()`.
    multi_worker_model = model(args)

    # Keras' `model.fit()` trains the model with specified number of epochs and
    # number of steps per epoch. 
    multi_worker_model.fit(train_datasets_sharded,
                         epochs=100,
                         steps_per_epoch=30)

    eval_loss, eval_acc = multi_worker_model.evaluate(test_dataset_sharded, 
                                                    verbose=0, steps=10)
    # Log metrics for Katib
    logging.info("loss={:.4f}".format(eval_loss))
    logging.info("accuracy={:.4f}".format(eval_acc))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch_size",
                      type=int,
                      default=32,
                      metavar="N",
                      help="Batch size for training (default: 128)")
  parser.add_argument("--learning_rate", 
                      type=float,  
                      default=0.1,
                      metavar="N",
                      help='Initial learning rate')
  parser.add_argument("--optimizer", 
                      type=str, 
                      default='adam',
                      metavar="N",
                      help='optimizer')
  parsed_args, _ = parser.parse_known_args()
  main(parsed_args)
