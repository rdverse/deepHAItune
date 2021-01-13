from tensorflow.keras import layers
from keras import backend as k
import IPython
import Data
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import model
from model import build_model_CNN
from plotter import hist_plotter
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

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

    modelA = layers.Conv2D(27,
                           kernel_size=(3, 3),
                           padding='same',
                           activation='relu')(inputA)

    modelA = layers.Conv2D(45,
                           kernel_size=(3, 3),
                           padding='same',
                           activation='relu')(modelA)

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

    modelG = layers.GlobalMaxPool2D()(modelG)
    model = layers.Concatenate()([modelA, modelG])

    model = layers.Dense(120, activation='relu')(model)
    model = layers.Dropout(0.4)(model)
    model = layers.Dense(30, activation='relu')(model)
    model = layers.Dropout(0.4)(model)
    output = layers.Dense(15)(model)

    model = tf.keras.Model(inputs=[inputA, inputG], outputs=output)

    optimizer = tf.keras.optimizers.Adam(1e-3)

    #loss = tf.keras.losses.MeanAbsoluteError(name="mean_absolute_error")

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(
        loss=loss,  #'mean_absolute_error',
        optimizer=optimizer,
        metrics=['accuracy'])

    return model


data_attr = [150, 50]
Features_A, Labels, pIDs = Data.dataset_main(150, 50, Accel='Yes')
Features_G, Labels, pIDs = Data.dataset_main(150, 50, Accel='No')

pIDsUnique = np.unique(pIDs)
pIDsUnique
pIDsInts = np.arange(len(pIDsUnique))
pIDsDict = dict()
for i, ID in enumerate(pIDsUnique):
    pIDsDict[ID] = pIDsInts[i]

pIDsVals = np.array([pIDsDict[ID] for ID in pIDs]).reshape(-1, 1)

oneHot = OneHotEncoder(sparse=False)
pIDsEnc = oneHot.fit_transform(pIDsVals)

Features_TrainA, Features_TestA, Labels_TrainA, Labels_TestA = train_test_split(
    Features_A, pIDsVals, shuffle=True, test_size=0.2, random_state=42)
Features_TrainG, Features_TestG, Labels_TrainG, Labels_TestG = train_test_split(
    Features_G, pIDsVals, shuffle=True, test_size=0.2, random_state=42)
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
    epochs=300,
    verbose=3
    #,
    #   callbacks=[
    #        earlystopping,
    #reduceLRplateau,
    #       tensorboard_callback
    #    ]
)

hist_plotter(hist.history)

print("Results of the model training")
print(model.evaluate([Features_TestA, Features_TestG], Labels_TestA))
