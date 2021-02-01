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

class RecurrentLSTMModel:
  def __init__(self, request_data):
    self.request_data = request_data
    self.train_percentage = request_data['datasetPercentages'][0]
    self.val_percentage = request_data['datasetPercentages'][1]
    self.test_percentage = request_data['datasetPercentages'][2]
    self.number_of_epochs = request_data['numberOfEpochs']
    self.window_size = request_data['specificContect']['WindowSize']
    return

  def compile_and_fit(self, model, window):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=self.number_of_epochs,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanAbsoluteError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=self.number_of_epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history

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

    inputWidth = self.window_size
    wide_window_RNN = WindowGenerator(
        input_width=inputWidth, label_width=inputWidth, shift=1,
        label_columns=['Column1'], train_df=train_df, val_df=val_df, test_df=test_df)
    print('\n\nwide_window_RNN')
    print(wide_window_RNN)
    print('\n\n')

    

    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences=True),
        # Shape => [batch, time, featuremv s]
        tf.keras.layers.Dense(units=1)
    ])

    history = self.compile_and_fit(lstm_model, wide_window_RNN)

    print('\nInput shape:', wide_window_RNN.example[0].shape)
    print('Output shape:', lstm_model(wide_window_RNN.example[0]).shape)

    print('\n\nsummary: ')
    lstm_model.summary()


    result = {'training_data_predictions': [], 'validation_data_predictions': [], 'test_data_predictions': []}

    predictions = lstm_model.predict(wide_window_RNN.train)
    for train in predictions:
      prediction = (train[0][0]*train_std[0])+train_mean[0]
      result['training_data_predictions'].append(prediction)

    predictions = lstm_model.predict(wide_window_RNN.val)
    for val in predictions:
      prediction = (val[0][0]*train_std[0])+train_mean[0]
      result['validation_data_predictions'].append(prediction)

    predictions = lstm_model.predict(wide_window_RNN.test)
    for test in predictions:
      prediction = (test[0][0]*train_std[0])+train_mean[0]
      result['test_data_predictions'].append(prediction)

    print('\n\nResult: ', result)

    return result