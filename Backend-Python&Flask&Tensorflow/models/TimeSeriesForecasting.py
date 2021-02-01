#This tutorial is an introduction to time series forecasting using TensorFlow. 
# It builds a few different styles of models including Convolutional and Recurrent Neural Networks (CNNs and RNNs).

########## IMPORTS ##########
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

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

########## GET DATA ##########
df = pd.read_excel("dados sensação térmica -  microclimas de bairro.xlsx") 

tempList = []
for i in range(len(df)):
    date = '01.' + str(df.iloc[[i]]['Month'].values)[1:-1] + '.' + str(df.iloc[[i]]['Year'].values)[1:-1]  + ' 01:00:00'
    tempList.append(date)
df['Date Time'] = tempList
df = df[:144]
print(df.head())
print(df.tail())

date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
print(df.head())

df.pop('Nome da Região')
df.pop('Year')
df.pop('Month')
print(df.head())


########## PLOTTING A FEW FEATURES OVER TIME ##########
print('\n\n\n\nPLOTTING A FEW FEATURES OVER TIME\n\n')
plot_cols = ['ST_']
plot_features = df[plot_cols]
plot_features.index = date_time
plot_features.plot(subplots=True)
plt.show()

plot_features = df[plot_cols][:480]
plot_features.index = date_time[:480]
plot_features.plot(subplots=True)
plt.show()

########## Next look at the statistics of the dataset: ##########
print('\n\nNext look at the statistics of the dataset:')
print(df.describe().transpose())

########## TIME ##########

# Similarly the Date Time column is very useful, but not in this string form. Start by converting it to seconds:
timestamp_s = date_time.map(datetime.datetime.timestamp)
print('\ntimestamp_s')
print(timestamp_s)

# Similar to the wind direction the time in seconds is not a useful model input. 
# Being weather data it has clear daily and yearly periodicity. There are many ways you could deal with periodicity.

# A simple approach to convert it to a usable signal is to use sin and cos to convert the time to clear 
# "Time of day" and "Time of year" signals:
day = 24*60*60
year = (365.2425)*day
month = year/12

df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

########## PLOTTING THGE PERIODICALLY WAVES ##########
plt.plot(np.array(df['Year sin'])[:13])
plt.plot(np.array(df['Year cos'])[:13])
plt.xlabel('Time [Year]')
plt.title('Time of day signal')
plt.show()

# This gives the model access to the most important frequency features. 
# In this case we knew ahead of time which frequencies were important.

# If you didn't know, you can determine which frequencies are important using an fft. 
# To check our assumptions, here is the tf.signal.rfft of the temperature over time. 
# Note the obvious peaks at frequencies near 1/year and 1/day:
fft = tf.signal.rfft(df['ST_'])
f_per_dataset = np.arange(0, len(fft))

n_samples_h = len(df['ST_'])
months_per_year = 12
years_per_dataset = n_samples_h/(months_per_year)
print('\nyears_per_dataset')
print(years_per_dataset)

print('\nIf you didnt know, you can determine which frequencies are important using an fft:')
f_per_year = f_per_dataset/years_per_dataset
plt.step(f_per_year, np.abs(fft))
plt.xscale('log')
plt.ylim(0, 400)
plt.xlim([0.1, max(plt.xlim())])
plt.xticks([1, 30, 365.2524], labels=['1/Year', '1/Month', '1/day'])
plt.xlabel('Frequency (log scale)')
plt.show()

########## SPLIT THE DATA ##########
# We'll use a (70%, 20%, 10%) split for the training, validation, and test sets. 
# Note the data is not being randomly shuffled before splitting. 
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]


########## NORMALIZE TEH DATA ##########
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

# Now peek at the distribution of the features. Some features do have long tails, 
# but there are no obvious errors like the -9999 wind velocity value.
print('\n\nNow peek at the distribution of the features:')
df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)
plt.show()


########## DATA WINDOWING ##########

# This section focuses on implementing the data windowing so that it can be reused for all of those models.

# The rest of this section defines a WindowGenerator class. This class can:
# 1 - Handle the indexes and offsets as shown in the diagrams above.
# 2 - Split windows of features into a (features, labels) pairs.
# 3 - Plot the content of the resulting windows.
# 4 - Efficiently generate batches of these windows from the training, evaluation, and test data, using tf.data.Datasets.

# Here is code to create the 2 windows shown in the diagrams at the start of this section:
w1 = WindowGenerator(input_width=24, label_width=1, shift=24,
                     label_columns=['ST_'], train_df=train_df, val_df=val_df, test_df=test_df)
print('\n\nCreate the 2 windows shown in the diagrams at the start of this section:')
print(w1)

print('\n')

w2 = WindowGenerator(input_width=6, label_width=1, shift=1,
                     label_columns=['ST_'], train_df=train_df, val_df=val_df, test_df=test_df)
print(w2)

# Try splitting the window
# Stack three slices, the length of the total window:
example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                           np.array(train_df[30:30+w2.total_window_size]),
                           np.array(train_df[60:60+w2.total_window_size])])


example_inputs, example_labels = w2.split_window(example_window)

print('\n\nTry splitting the window:')
print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'labels shape: {example_labels.shape}')

# Typically data in TensorFlow is packed into arrays where the outermost index is across examples 
# (the "batch" dimension). The middle indices are the "time" or "space" (width, height) dimension(s). 
# The innermost indices are the features.

# The code above took a batch of 3, 7-timestep windows, with 19 features at each time step. 
# It split them into a batch of 6-timestep, 19 feature inputs, and a 1-timestep 1-feature label. 
# The label only has one feature because the WindowGenerator was initialized with label_columns=['T (degC)']. 
# Initially this tutorial will build models that predict single output labels.

w2.plot()

# You can plot the other columns, but the example window w2 configuration only has labels for the T (degC) column.
print('\nYou can plot the other columns, but the example window w2 configuration only has labels for the T (degC) column:')
w2.plot(plot_col='Year sin')

# Finally this make_dataset method will take a time series DataFrame and convert it to a tf.data.Dataset of 
# (input_window, label_window) pairs using the preprocessing.timeseries_dataset_from_array function.

# The Dataset.element_spec property tells you the structure, dtypes and shapes of the dataset elements.

# Each element is an (inputs, label) pair
print('\n\nEach element is an (inputs, label) pair:')
w2.train.element_spec

# Iterating over a Dataset yields concrete batches:
for example_inputs, example_labels in w2.train.take(1):
    print(f'EXAMPLE Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'EXAMPLE Labels shape (batch, time, features): {example_labels.shape}')




################## SINGLE STEPS MODELS ##################

# The simplest model you can build on this sort of data is one that predicts a single feature's value, 
# 1 timestep (1h) in the future based only on the current conditions.

# So start by building models to predict the T (degC) value 1h into the future.

#### Configure a WindowGenerator object to produce these single-step (input, label) pairs:
single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,
    label_columns=['ST_'], train_df=train_df, val_df=val_df, test_df=test_df)
print('\n\nConfigure a WindowGenerator object to produce these single-step (input, label) pairs:')
print(single_step_window)

# The window object creates tf.data.Datasets from the training, validation, and test sets, 
# allowing you to easily iterate over batches of data.
for example_inputs, example_labels in single_step_window.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')



########## BASELINE MODEL ##########
########## BASELINE MODEL ##########
########## BASELINE MODEL ##########
########## BASELINE MODEL ##########

# Before building a trainable model it would be good to have a performance baseline as a point for 
# comparison with the later more complicated models.

# This first task is to predict temperature 1h in the future given the current value of all features. 
# The current values include the current temperature.

# So start with a model that just returns the current temperature as the prediction, 
# predicting "No change". This is a reasonable baseline since temperature changes slowly. 
# Of course, this baseline will work less well if you make a prediction further in the future.
baseline = Baseline(label_index=column_indices['ST_'])

baseline.compile(loss=tf.losses.MeanSquaredError(),
                 metrics=[tf.metrics.MeanAbsoluteError()])

val_performance = {}
performance = {}
val_performance_MAPE = {}
performance_MAPE = {}
val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
performance['Baseline'] = baseline.evaluate(single_step_window.test)


def calcMAPE(model, testData, valData, name):
    predictions = model.predict(testData)
    print('\n\n\nTEST - predictions - ' + name)
    i = 1
    for x, y in testData:
        if i == 1:
            newY = y
        i = i + 1

    i = 0
    predVector = []
    newYVector = []
    for el in predictions:
        print(str(el[0][0]) + '  -  ' + str(newY.numpy()[i][0][0]))
        i = i + 1
    print('\n\n')

    i = 0
    train_std_new = train_std['ST_']
    train_meam_new = train_mean['ST_']
    performance_value = 0
    for el in predictions:
        predNewValue = ( el[0][0] * train_std_new ) + train_meam_new
        newYNewValue = ( newY.numpy()[i][0][0] * train_std_new ) + train_meam_new
        predVector.append(predNewValue)
        newYVector.append(newYNewValue)
        print(str(predNewValue) + '  -  ' + str(newYNewValue))
        performance_value = performance_value + abs( (newYNewValue-predNewValue)/newYNewValue )
        i = i + 1
    performance_value = performance_value / i
    performance_MAPE[name] = performance_value
    print('\n\nperformance_MAPE[' + name + ']')
    print(performance_MAPE[name])
    print('\n\n\n')

    predictions = model.predict(valData)
    print('\n\n\nVAL - predictions - ' + name)
    i = 1
    for x, y in valData:
        if i == 1:
            newY = y
        i = i + 1

    i = 0
    predVector = []
    newYVector = []
    for el in predictions:
        print(str(el[0][0]) + '  -  ' + str(newY.numpy()[i][0][0]))
        i = i + 1
    print('\n\n')

    i = 0
    train_std_new = train_std['ST_']
    train_meam_new = train_mean['ST_']
    performance_value = 0
    for el in predictions:
        predNewValue = ( el[0][0] * train_std_new ) + train_meam_new
        newYNewValue = ( newY.numpy()[i][0][0] * train_std_new ) + train_meam_new
        predVector.append(predNewValue)
        newYVector.append(newYNewValue)
        print(str(predNewValue) + '  -  ' + str(newYNewValue))
        performance_value = performance_value + abs( (newYNewValue-predNewValue)/newYNewValue )
        i = i + 1
    performance_value = performance_value / i
    val_performance_MAPE[name] = performance_value
    print('\n\nval_performance_MAPE[' + name + ']')
    print(val_performance_MAPE[name])
    print('\n\n\n')


calcMAPE(baseline, single_step_window.test, single_step_window.val, 'Baseline')



predictions = baseline.predict(single_step_window.val)
print('\n\n\npredictions - val')
print(predictions)
print(single_step_window.val)
i = 1
for x, y in single_step_window.val:
   print(i)
   print(i)
   print(i)
   print(i)
   print(y)
   i = i + 1
print('\n\n\n')


# The WindowGenerator has a plot method, but the plots won't be very interesting with only a single sample. 
# So, create a wider WindowGenerator that generates windows 24h of consecutive inputs and labels at a time.

# The wide_window doesn't change the way the model operates. 
# The model still makes predictions 1h into the future based on a single input time step. 
# Here the time axis acts like the batch axis: Each prediction is made independently with no interaction 
# between time steps.
print('\n\nThe model still makes predictions 1h into the future based on a single input time step. Here the time axis acts like the batch axis: Each prediction is made independently with no interaction between time steps:')

wide_window = WindowGenerator(
    input_width=24, label_width=24, shift=1,
    label_columns=['ST_'], train_df=train_df, val_df=val_df, test_df=test_df)
print(wide_window)

# This expanded window can be passed directly to the same baseline model without any code changes. 
# This is possible because the inputs and labels have the same number of timesteps, and the baseline just 
# forwards the input to the output:

print('\nInput shape:', single_step_window.example[0].shape)
print('Output shape:', baseline(single_step_window.example[0]).shape)

# Plotting the baseline model's predictions you can see that it is simply the labels, shifted right by 1h.
print('\n\nPlotting the baseline models predictions you can see that it is simply the labels, shifted right by 1h:')
wide_window.plot(baseline)


# In the above plots of three examples the single step model is run over the course of 24h. This deserves some explaination:

# The blue "Inputs" line shows the input temperature at each time step. The model recieves all features, this plot only shows the temperature.
# The green "Labels" dots show the target prediction value. These dots are shown at the prediction time, not the input time. That is why the range of labels is shifted 1 step relative to the inputs.
# The orange "Predictions" crosses are the model's prediction's for each output time step. If the model were predicting perfectly the predictions would land directly on the "labels".


########## LINEAR MODEL ##########
########## LINEAR MODEL ##########
########## LINEAR MODEL ##########
########## LINEAR MODEL ##########

# The simplest trainable model you can apply to this task is to insert linear transformation between the input and output. In this case the output from a time step only depends on that step:

print('\n\nThe simplest trainable model you can apply to this task is to insert linear transformation between the input and output. In this case the output from a time step only depends on that step:')

# A layers.Dense with no activation set is a linear model. The layer only transforms the last axis of the 
# data from (batch, time, inputs) to (batch, time, units), it is applied independently to every item across the 
# batch and time axes.

linear = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
])
print('Input shape:', single_step_window.example[0].shape)
print('Output shape:', linear(single_step_window.example[0]).shape)


MAX_EPOCHS = 300
# This tutorial trains many models, so package the training procedure into a function:
def compile_and_fit(model, window, patience=10):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history


# Train the model and evaluate its performance:
print('\n\nTrain the model and evaluate its performance:')
history = compile_and_fit(linear, single_step_window)

val_performance['Linear'] = linear.evaluate(single_step_window.val)
performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0)

calcMAPE(linear, single_step_window.test, single_step_window.val, 'Linear')

# Like the baseline model, the linear model can be called on batches of wide windows. 
# Used this way the model makes a set of independent predictions on consecuitive time steps. 
# The time axis acts like another batch axis. There are no interactions between the precictions at each time step.

print('\nInput shape:', wide_window.example[0].shape)
print('Output shape:', baseline(wide_window.example[0]).shape)

# Here is the plot of its example predictions on the wide_window, note how in many cases the prediction is clearly better than just returning the input temperature, but in a few cases it's worse:
print('\nHere is the plot of its example predictions on the wide_window, note how in many cases the prediction is clearly better than just returning the input temperature, but in a few cases its worse:')
wide_window.plot(linear)

# One advantage to linear models is that they're relatively simple to interpret. 
# You can pull out the layer's weights, and see the weight assigned to each input:

print('\n\nOne advantage to linear models is that theyre relatively simple to interpret. You can pull out the layers weights, and see the weight assigned to each input:')
plt.bar(x = range(len(train_df.columns)),
        height=linear.layers[0].kernel[:,0].numpy())
axis = plt.gca()
axis.set_xticks(range(len(train_df.columns)))
_ = axis.set_xticklabels(train_df.columns, rotation=90)
plt.show()

# Sometimes the model doesn't even place the most weight on the input T (degC). 
# This is one of the risks of random initialization.


########## DENSE MODEL ##########
########## DENSE MODEL ##########
########## DENSE MODEL ##########
########## DENSE MODEL ##########

# Here's a model similar to the linear model, except it stacks several a few Dense layers between 
# the input and the output:
print('\n\n\nHeres a model similar to the linear model, except it stacks several a few Dense layers between the input and the output:')

dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

history = compile_and_fit(dense, single_step_window)

val_performance['Dense'] = dense.evaluate(single_step_window.val)
performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)

calcMAPE(dense, single_step_window.test, single_step_window.val, 'Dense')

wide_window.plot(dense)



########## MULTI_STEP DENSE MODEL ##########
########## MULTI_STEP DENSE MODEL ##########
########## MULTI_STEP DENSE MODEL ##########
########## MULTI_STEP DENSE MODEL ##########

# A single-time-step model has no context for the current values of its inputs. 
# It can't see how the input features are changing over time. 
# To address this issue the model needs access to multiple time steps when making predictions:

print('\nThe baseline, linear and dense models handled each time step independently. Here the model will take multiple time steps as input to produce a single output:')

# Create a WindowGenerator that will produce batches of the 3h of inputs and, 1h of labels:
# Note that the Window's shift parameter is relative to the end of the two windows.
CONV_WIDTH = 3
conv_window = WindowGenerator(
    input_width=CONV_WIDTH,
    label_width=1,
    shift=1,
    label_columns=['ST_'], train_df=train_df, val_df=val_df, test_df=test_df)

print(conv_window)

print('Given 3h as input, predict 1h into the future.')
conv_window.plot()

# You could train a dense model on a multiple-input-step window by adding a layers.Flatten as the first layer of the model:
multi_step_dense = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1]),
])

print('\nInput shape:', conv_window.example[0].shape)
print('Output shape:', multi_step_dense(conv_window.example[0]).shape)

history = compile_and_fit(multi_step_dense, conv_window)

IPython.display.clear_output()
val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val)
performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.test, verbose=0)

calcMAPE(multi_step_dense, conv_window.test, conv_window.val, 'Multi step dense')

print('\nYou could train a dense model on a multiple-input-step window by adding a layers.Flatten as the first layer of the model:')
conv_window.plot(multi_step_dense)

# The main down-side of this approach is that the resulting model can only be executed on input wndows 
# of exactly this shape.
print('\n\nThe main down-side of this approach is that the resulting model can only be executed on input wndows of exactly this shape.')
print('Input shape:', wide_window.example[0].shape)
try:
  print('Output shape:', multi_step_dense(wide_window.example[0]).shape)
except Exception as e:
  print(f'\n{type(e).__name__}:{e}')

# The convolutional models in the next section fix this problem.
print('\nThe convolutional models in the next section fix this problem:')



########## CONVOLUTIONAL MODEL ##########
########## CONVOLUTIONAL MODEL ##########
########## CONVOLUTIONAL MODEL ##########
########## CONVOLUTIONAL MODEL ##########

# A convolution layer (layers.Conv1D) also takes multiple time steps as input to each prediction.
# Below is the same model as multi_step_dense, re-written with a convolution.

# The layers.Flatten and the first layers.Dense are replaced by a layers.Conv1D.
# The layers.Reshape is no longer necessary since the convolution keeps the time axis in its output.
conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(CONV_WIDTH,),
                           activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
])
print("Conv model on `conv_window`")
print('Input shape:', conv_window.example[0].shape)
print('Output shape:', conv_model(conv_window.example[0]).shape)

# Train and evaluate it on the conv_window and it should give performance similar to the multi_step_dense model.
print('\n\nTrain and evaluate it on the conv_window and it should give performance similar to the multi_step_dense model:')

history = compile_and_fit(conv_model, conv_window)

IPython.display.clear_output()
val_performance['Conv'] = conv_model.evaluate(conv_window.val)
performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)

calcMAPE(conv_model, conv_window.test, conv_window.val, 'Conv')

# The difference between this conv_model and the multi_step_dense model is that the conv_model can be run on 
# inputs on inputs of any length. The convolutional layer is applied to a sliding window of inputs:
print('\nThe difference between this conv_model and the multi_step_dense model is that the conv_model can be run on inputs on inputs of any length. The convolutional layer is applied to a sliding window of inputs:')

# If you run it on wider input, it produces wider output:
print('\nIf you run it on wider input, it produces wider output:')

print("\nWide window")
print('Input shape:', wide_window.example[0].shape)
print('Labels shape:', wide_window.example[1].shape)
print('Output shape:', conv_model(wide_window.example[0]).shape)


# Note that the output is shorter than the input. To make training or plotting work, you need the labels, 
# and prediction to have the same length. So build a WindowGenerator to produce wide windows with a few 
# extra input time steps so the label and prediction lengths match:
LABEL_WIDTH = 24
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
wide_conv_window = WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1,
    label_columns=['ST_'], train_df=train_df, val_df=val_df, test_df=test_df)
print('\n\nwide_conv_window:')
print(wide_conv_window)

print("\nWide conv window")
print('Input shape:', wide_conv_window.example[0].shape)
print('Labels shape:', wide_conv_window.example[1].shape)
print('Output shape:', conv_model(wide_conv_window.example[0]).shape)

# Now you can plot the model's predictions on a wider window. Note the 3 input time steps before the first prediction. Every prediction here is based on the 3 preceding timesteps:
print('\n\nNow you can plot the models predictions on a wider window. Note the 3 input time steps before the first prediction. Every prediction here is based on the 3 preceding timesteps:')
wide_conv_window.plot(conv_model)



########## RECURRENT NEURAL NETWORK MODEL ##########
########## RECURRENT NEURAL NETWORK MODEL ##########
########## RECURRENT NEURAL NETWORK MODEL ##########
########## RECURRENT NEURAL NETWORK MODEL ##########

wide_window_RNN = WindowGenerator(
    input_width=12, label_width=12, shift=1,
    label_columns=['ST_'], train_df=train_df, val_df=val_df, test_df=test_df)
print(wide_window_RNN)


print('\n\n\n\nRECURRENT NEURAL NETWORK:')

lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

# With return_sequences=True the model can be trained on 24h of data at a time.
# Note: This will give a pessimistic view of the model's performance. On the first timestep the model has no 
# access to previous steps, and so can't do any better than the simple linear and dense models shown earlier.

print('\nInput shape:', wide_window_RNN.example[0].shape)
print('Output shape:', lstm_model(wide_window_RNN.example[0]).shape)

history = compile_and_fit(lstm_model, wide_window_RNN)

IPython.display.clear_output()
val_performance['LSTM'] = lstm_model.evaluate(wide_window_RNN.val)
performance['LSTM'] = lstm_model.evaluate(wide_window_RNN.test, verbose=0)

calcMAPE(lstm_model, wide_window_RNN.test, wide_window_RNN.val, 'LSTM')

wide_window.plot(lstm_model)


########## PERFORMANCE ##########
########## PERFORMANCE ##########
########## PERFORMANCE ##########
########## PERFORMANCE ##########

print('\n\nPERFORMANCE')
print('\nWith this dataset typically each of the models does slightly better than the one before it:')

x = np.arange(len(performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')
print('\n\nmetric_index')
print(metric_index)

print('\n\nval_performance.values()')
print(val_performance.values())
print('\n\nperformance.values()')
print(performance.values())

val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

plt.ylabel('mean_absolute_error [ST_, normalized]')
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=performance.keys(),
           rotation=45)
_ = plt.legend()
plt.show()

for name, value in performance.items():
    print(f'{name:12s}: {value[1]:0.4f}')




x = np.arange(len(performance_MAPE))
width = 0.3

print('\n\nval_performance_MAPE.values()')
print(val_performance_MAPE.values())
print('\n\nperformance_MAPE.values()')
print(performance_MAPE.values())

val_mae = [v for v in val_performance_MAPE.values()]
test_mae = [v for v in performance_MAPE.values()]

plt.ylabel('mean_absolute_PERCENTAGE_error [ST_, normalized]')
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=performance_MAPE.keys(),
           rotation=45)
_ = plt.legend()
plt.show()

for name, value in performance_MAPE.items():
    print(f'{name:12s}: {value[0]:0.4f}')
