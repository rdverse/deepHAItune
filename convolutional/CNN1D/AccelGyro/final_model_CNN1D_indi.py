######### Import Libraries
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
import numpy as np

##########Setup GPU
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    print('Could not initialize the tensorflow gpu')
    pass

#####Some Global declarations
data_attr = [50, 50]
logdir = 'logdir'

######Define callbacks for the model training

earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 patience=50)

reduceLRplateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                       min_lr=1e-6,
                                                       patience=15,
                                                       factor=0.9,
                                                       verbose=1)

tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)


class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)


#######################Utility Functions############


def make_inexistent(path):
    '''
    create a directory if it is inexistent
    '''
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)


def R_Square(y_true, y_pred):
    '''
    Custom keras callback R^2 function
    '''
    Num = k.sum(k.square(y_true - y_pred))
    Denom = k.sum(k.square(y_true - k.mean(y_true)))
    R = 1 - Num / (Denom + k.epsilon())
    return R


def reshaping(array):
    '''
    Reshape the arrays to make the 1D convolutions fit the case
    '''
    size_arr = len(array)
    #    print(array.shape)
    array = array.swapaxes(1, 2)
    #   print(array.shape)
    array = array.ravel().reshape(size_arr, data_attr[0], 3)
    #  print(array.shape)

    return (array)


##############Model Architecture
def grow_branch(input):
    # Takes input
    # Create branch
    # Sadly every branch has same number of convolutions
    fil = 45
    ksize = 5
    pad = ''
    use_bias = True
    kernel_initializer = 'glorot_normal'

    model = layers.Conv1D(
        fil,
        kernel_size=ksize,
        padding='valid',
        #activation='relu',
        #        name='conv1D_1',
        use_bias=use_bias,
        kernel_initializer=kernel_initializer)(input)

    model = layers.BatchNormalization()(model)
    #modelA = layers.Activation('relu')(modelA)
    model = layers.LeakyReLU()(model)

    model = layers.Conv1D(
        fil,
        kernel_size=ksize,
        padding='same',
        #activation='relu',
        #       name='conv1D_2',
        use_bias=use_bias,
        kernel_initializer=kernel_initializer)(model)

    model = layers.BatchNormalization()(model)
    #    modelA = layers.Add()([modelA, inputA])
    model = layers.Activation('relu')(model)

    model_copy = model

    model = layers.GlobalMaxPool1D()(model)
    model = layers.Add()([model_copy, model])
    #modelA = layers.Activation('relu')(modelA)
    model = layers.LeakyReLU()(model)
    model = layers.GlobalMaxPool1D()(model)

    #Create a feature  vector of size framesize/3 approximately.
    return model


def build_model_CNN():

    A_Xi = layers.Input(shape=(data_attr[0], 1))
    A_X = grow_branch(A_Xi)

    A_Yi = layers.Input(shape=(data_attr[0], 1))
    A_Y = grow_branch(A_Yi)

    A_Zi = layers.Input(shape=(data_attr[0], 1))
    A_Z = grow_branch(A_Zi)

    G_Xi = layers.Input(shape=(data_attr[0], 1))
    G_X = grow_branch(G_Xi)

    G_Yi = layers.Input(shape=(data_attr[0], 1))
    G_Y = grow_branch(G_Yi)

    G_Zi = layers.Input(shape=(data_attr[0], 1))
    G_Z = grow_branch(G_Zi)

    model = layers.Concatenate()([A_X, A_Y, A_Z, G_X, G_Y, G_Z])

    #    model = modelG

    #   layers.concatenate(modelA,modelG)

    #model = layers.Dropout(0.4)(model)

    model = layers.Dense(270,
                         activation='relu',
                         kernel_initializer='he_uniform')(model)

    model = layers.Dropout(0.4)(model)

    model = layers.Dense(30,
                         activation='relu',
                         kernel_initializer='he_uniform')(model)

    model = layers.Dropout(0.4)(model)

    output = layers.Dense(1)(model)

    model = tf.keras.Model(inputs=[A_Xi, A_Yi, A_Zi, G_Xi, G_Yi, G_Zi],
                           outputs=output)

    optimizer = tf.keras.optimizers.RMSprop(1e-3)
    optimizer = tf.keras.optimizers.Adam(1e-3)

    loss = tf.keras.losses.MeanAbsoluteError(reduction="auto",
                                             name="mean_absolute_error")
    model.compile(
        loss=loss,  #'mean_absolute_error',
        optimizer=optimizer,
        metrics=['mae', 'mape', R_Square])

    tf.keras.utils.plot_model(model,
                              to_file='/home/redev/Pictures/Model.png',
                              show_shapes=True)
    return model


def make_data(data_attr):

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

    Features_TrainA = reshaping(Features_TrainA)
    Features_TrainG = reshaping(Features_TrainG)
    Features_TestA = reshaping(Features_TestA)
    Features_TestG = reshaping(Features_TestG)
    Features_ValA = reshaping(Features_ValA)
    Features_ValG = reshaping(Features_ValG)

    print(Features_TrainA.shape)

    #Distill each axis
    A_XT, A_YT, A_ZT = np.array_split(Features_TrainA, 3, axis=-1)
    G_XT, G_YT, G_ZT = np.array_split(Features_TrainG, 3, axis=-1)

    A_XV, A_YV, A_ZV = np.array_split(Features_ValA, 3, axis=-1)
    G_XV, G_YV, G_ZV = np.array_split(Features_ValG, 3, axis=-1)

    A_XTe, A_YTe, A_ZTe = np.array_split(Features_TestA, 3, axis=-1)
    G_XTe, G_YTe, G_ZTe = np.array_split(Features_TestG, 3, axis=-1)

    L_T = Labels_TrainA
    L_V = Labels_ValA
    L_Te = Labels_TestA

    return (A_XT, A_YT, A_ZT, G_XT, G_YT, G_ZT, A_XV, A_YV, A_ZV, G_XV, G_YV,
            G_ZV, A_XTe, A_YTe, A_ZTe, G_XTe, G_YTe, G_ZTe, L_T, L_V, L_Te)


A_XT, A_YT, A_ZT, G_XT, G_YT, G_ZT, A_XV, A_YV, A_ZV, G_XV, G_YV, G_ZV, A_XTe, A_YTe, A_ZTe, G_XTe, G_YTe, G_ZTe, L_T, L_V, L_Te = make_data(
    data_attr)

model = build_model_CNN()

print(model.summary())

hist = model.fit(
    [A_XT, A_YT, A_ZT, G_XT, G_YT, G_ZT],
    L_T,
    validation_data=([A_XV, A_YV, A_ZV, G_XV, G_YV, G_ZV], L_V),
    epochs=1000,
    verbose=3,
    callbacks=[
        #        earlystopping,
        #reduceLRplateau,
        tensorboard_callback,
        ClearTrainingOutput()
    ])

hist_plotter(hist.history)

print("Results of the model training")
print(model.evaluate([A_XTe, A_YTe, A_ZTe, G_XTe, G_YTe, G_ZTe], L_Te))
