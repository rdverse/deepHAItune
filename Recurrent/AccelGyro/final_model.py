from tensorflow.keras import layers
from keras import backend as k
import IPython

import Data
import tensorflow as tf
from sklearn.model_selection import train_test_split

import model
from model import build_model_CNN

from plotter import hist_plotter

import os

from tensorflow_docs.modeling import EpochDots

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    print('Could not initialize the tensorflow gpu')
    pass


def R_Square(y_true, y_pred):

    Num = k.sum(k.square(y_true - y_pred))
    Denom = k.sum(k.square(y_true - k.mean(y_true)))
    print(Num)
    R = 1 - Num / (Denom + k.epsilon())
    return R


def build_model_CNN():

    min_conv1 = 27
    max_conv1 = 36
    min_conv2 = 36
    max_conv2 = 45
    convLSTMfils = 45
    input_shape = (3, 150, 1)

    inputA = layers.Input(shape=input_shape, name='Accelerometer')
    modelA = inputA
    modelA = layers.Reshape((1, 3, 150, 1))(modelA)
    modelA = layers.ConvLSTM2D(convLSTMfils, (3, 3),
                               padding='same',
                               activation='relu')(modelA)
    modelA = layers.MaxPooling2D()(modelA)
    modelA = layers.GlobalAveragePooling2D()(modelA)

    inputG = layers.Input(shape=input_shape, name='Gyroscope')
    modelG = inputG
    modelG = layers.Reshape((1, 3, 150, 1))(modelG)
    modelG = layers.ConvLSTM2D(convLSTMfils, (3, 3),
                               padding='same',
                               activation='relu')(modelG)
    modelG = layers.MaxPooling2D()(modelG)
    modelG = layers.GlobalAveragePooling2D()(modelG)

    model = layers.Concatenate()([modelA, modelG])
    model = layers.Dense(180, activation='relu')(model)
    model = layers.Dropout(0.4)(model)
    model = layers.Dense(30, activation='relu')(model)
    model = layers.Dropout(0.4)(model)
    output = layers.Dense(1)(model)

    model = tf.keras.Model(inputs=[inputA, inputG], outputs=output)

    optimizer = tf.keras.optimizers.RMSprop(1e-3)
    optimizer = tf.keras.optimizers.Adam(1e-3)

    loss = tf.keras.losses.Huber()
    loss = tf.keras.losses.LogCosh()
    loss = tf.keras.losses.MeanSquaredError(reduction="auto",
                                            name="mean_squared_error")

    loss = tf.keras.losses.MeanAbsolutePercentageError()

    loss = tf.keras.losses.MeanAbsoluteError(reduction="auto",
                                             name="mean_absolute_error")

    model.compile(
        loss=loss,  #'mean_absolute_error',
        optimizer=optimizer,
        metrics=['mae', 'mape', R_Square])

    # tf.keras.utils.plot_model(model,
    #                           to_file='/home/redev/Pictures/Model.png',
    #                          show_shapes=True)
    return model


data_attr = [150, 50]
Features_TrainA, Labels_TrainA, Features_TestA, Labels_TestA = Data.dataset_main(
    data_attr[0], data_attr[1], Accel='Yes')

Features_TrainG, Labels_TrainG, Features_TestG, Labels_TestG = Data.dataset_main(
    data_attr[0], data_attr[1], Accel='No')

Features_TrainA, Features_ValA, Labels_TrainA, Labels_ValA = train_test_split(
    Features_TrainA,
    Labels_TrainA,
    shuffle=True,
    test_size=0.2,
    random_state=42)

Features_TrainG, Features_ValG, Labels_TrainG, Labels_ValG = train_test_split(
    Features_TrainG,
    Labels_TrainG,
    shuffle=True,
    test_size=0.2,
    random_state=42)


class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)


model = build_model_CNN()

earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 patience=50)

reduceLRplateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                       min_lr=1e-6,
                                                       patience=30,
                                                       factor=0.9,
                                                       verbose=1)

if os.path.isdir('logdir'):
    pass
else:
    os.mkdir('logdir')

logdir = 'logdir'
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

print(model.summary())

hist = model.fit(
    [Features_TrainA, Features_TrainG],
    Labels_TrainA,
    validation_data=([Features_ValA, Features_ValG], Labels_ValG),
    epochs=2000,
    verbose=3,
    callbacks=[
        #   earlystopping,
        #   reduceLRplateau,
        tensorboard_callback,
        EpochDots(),
        ClearTrainingOutput()
    ])

hist_plotter(hist.history)

print("Results of the model training")
print(model.evaluate([Features_TestA, Features_TestG], Labels_TestA))
