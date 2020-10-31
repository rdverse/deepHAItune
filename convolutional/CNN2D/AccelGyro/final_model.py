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
    R = 1 - Num / (Denom + k.epsilon())
    return R


def build_model_CNN():
    inputA = layers.Input(shape=(3, 150, 1))
    modelA = inputA
    min_conv1 = 27
    max_conv1 = 36
    min_conv2 = 36
    max_conv2 = 45
    modelA = layers.Conv2D(27,
                           kernel_size=(3, 3),
                           padding='same',
                           activation='relu')(inputA)

    modelA = layers.Conv2D(45,
                           kernel_size=(3, 3),
                           padding='same',
                           activation='relu')(modelA)

    #modelA = layers.Flatten()(modelA)
    modelA = layers.GlobalMaxPool2D()(modelA)
    inputG = layers.Input(shape=(3, 150, 1))

    modelG = inputG

    modelG = layers.Conv2D(27,
                           kernel_size=(3, 3),
                           padding='same',
                           activation='relu')(inputG)

    modelG = layers.Conv2D(45,
                           kernel_size=(3, 3),
                           padding='same',
                           activation='relu')(modelG)

    # model = layers.Dropout(0.4)(model)
    #modelG = layers.Flatten()(modelG)
    modelG = layers.GlobalMaxPool2D()(modelG)
    model = layers.Concatenate()([modelA, modelG])
    #   layers.concatenate(modelA,modelG)

    #model = layers.Dropout(0.4)(model)

    model = layers.Dense(120, activation='relu')(model)

    model = layers.Dropout(0.4)(model)

    model = layers.Dense(30, activation='relu')(model)

    model = layers.Dropout(0.4)(model)

    output = layers.Dense(1)(model)

    model = tf.keras.Model(inputs=[inputA, inputG], outputs=output)

    optimizer = tf.keras.optimizers.RMSprop(1e-3)
    # optimizer = tf.keras.optimizers.Adam(1e-3)

    loss = tf.keras.losses.Huber()
    loss = tf.keras.losses.LogCosh()
    loss = tf.keras.losses.MeanAbsolutePercentageError()
    loss = tf.keras.losses.MeanSquaredError(reduction="auto",
                                            name="mean_squared_error")
    loss = tf.keras.losses.MeanAbsoluteError(name="mean_absolute_error")
    model.compile(
        loss=loss,  #'mean_absolute_error',
        optimizer=optimizer,
        metrics=['mae', 'mape', R_Square])

    tf.keras.utils.plot_model(model,
                              to_file='/home/redev/Pictures/Model.png',
                              show_shapes=True)
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
                                                       patience=10,
                                                       factor=0.9,
                                                       verbose=1)

if os.path.isdir('logdir'):
    pass
else:
    os.mkdir('logdir')

logdir = 'logdir'
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

hist = model.fit(
    [Features_TrainA, Features_TrainG],
    Labels_TrainA,
    validation_data=([Features_ValA, Features_ValG], Labels_ValA),
    epochs=200,
    verbose=3,
    callbacks=[
        #        earlystopping,
        #reduceLRplateau,
        tensorboard_callback,
        ClearTrainingOutput()
    ])

hist_plotter(hist.history)

print("Results of the model training")
print(model.evaluate([Features_TestA, Features_TestG], Labels_TestA))
