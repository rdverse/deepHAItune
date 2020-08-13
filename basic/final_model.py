from tensorflow.keras import layers
from keras import backend as k
import IPython

import Data
import tensorflow as tf
from sklearn.model_selection import train_test_split
import kerastuner as kt
import model
from model import build_model_CNN


def R_Square(y_true, y_pred):

    Num = k.sum(k.square(y_true - y_pred))
    Denom = k.sum(k.square(y_true - k.mean(y_true)))
    R = 1 - Num / (Denom + k.epsilon())
    return R


def build_model_CNN(hp):
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

    modelA = layers.Flatten()(modelA)

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
    modelG = layers.Flatten()(modelG)

    model = layers.Concatenate()([modelA, modelG])
    #   layers.concatenate(modelA,modelG)

    model = layers.Dropout(0.4)(model)

    model = layers.Dense(120, activation='relu')(model)

    model = layers.Dropout(0.4)(model)

    model = layers.Dense(30, activation='relu')(model)

    model = layers.Dropout(0.4)(model)

    output = layers.Dense(1)(model)

    model = tf.keras.Model(inputs=[inputA, inputG], outputs=output)

    optimizer = tf.keras.optimizers.RMSprop(1e-3)

    model.compile(loss='mean_absolute_error',
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

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

model.fit([Features_TrainA, Features_TrainG],
          Labels_TrainA,
          validation_data=([Features_ValA, Features_ValG], Labels_ValA),
          epochs=200,
          verbose=3,
          callbacks=[callback, ClearTrainingOutput()])

print("Results of the model training")
print(model.evaluate([Features_TestA, Features_TestG], Labels_TestA))
