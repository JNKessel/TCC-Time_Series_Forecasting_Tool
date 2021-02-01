import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from WindowGenerator import WindowGenerator
from Baseline import Baseline

class BaselineModel:
  def __init__(self, request_data):
    self.request_data = request_data
    self.train_percentage = request_data['datasetPercentages'][0]
    self.val_percentage = request_data['datasetPercentages'][1]
    self.test_percentage = request_data['datasetPercentages'][2]
    return

  def predict(self):
    request_data_df = pd.DataFrame(self.request_data['seriesData'], columns=['Column1'])
    print('\n\nrequest_data_df', request_data_df)

    ########## SPLIT THE DATA ##########
    # We'll use a (70%, 20%, 10%) split for the training, validation, and test sets. 
    # Note the data is not being randomly shuffled before splitting. 
    column_indices = {name: i for i, name in enumerate(request_data_df.columns)}
    print('\n\ncolumn_indices', column_indices)

    n = len(request_data_df)
    train_df = request_data_df[0:int(n * self.train_percentage)]
    val_df = request_data_df[int(n * self.train_percentage):int(n * (self.train_percentage + self.val_percentage))]
    test_df = request_data_df[int(n * (self.train_percentage + self.val_percentage)):]

    num_features = request_data_df.shape[1]

    ########## NORMALIZE TEH DATA ##########
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    print('\n\nlen(train_df)')
    print(len(train_df))
    print('\nlen(val_df)')
    print(len(val_df))
    print('\nlen(test_df)')
    print(len(test_df))

    ################## SINGLE STEPS MODELS ##################

    #### Configure a WindowGenerator object to produce these single-step (input, label) pairs:
    single_step_window = WindowGenerator(
        input_width=1, label_width=1, shift=1,
        label_columns=['Column1'], train_df=train_df, val_df=val_df, test_df=test_df)
    print('\n\nConfigure a WindowGenerator object to produce these single-step (input, label) pairs: single_step_window')
    print(single_step_window)
    print('\n\n')

    ########## BASELINE MODEL ##########
    ########## BASELINE MODEL ##########
    ########## BASELINE MODEL ##########
    ########## BASELINE MODEL ##########
    baseline = Baseline(label_index=column_indices['Column1'])

    baseline.compile(loss=tf.losses.MeanSquaredError(),
                    metrics=[tf.metrics.MeanAbsoluteError()])


    result = {'training_data_predictions': [], 'validation_data_predictions': [], 'test_data_predictions': []}

    predictions = baseline.predict(single_step_window.train)
    for train in predictions:
      prediction = (train[0][0]*train_std[0])+train_mean[0]
      result['training_data_predictions'].append(prediction)

    predictions = baseline.predict(single_step_window.val)
    for val in predictions:
      prediction = (val[0][0]*train_std[0])+train_mean[0]
      result['validation_data_predictions'].append(prediction)

    predictions = baseline.predict(single_step_window.test)
    print('\n\n\npredictions - test')
    ii = 0 
    for test in predictions:
      prediction = (test[0][0]*train_std[0])+train_mean[0]
      result['test_data_predictions'].append(prediction)

    print('\n\nResult: ', result)

    return result