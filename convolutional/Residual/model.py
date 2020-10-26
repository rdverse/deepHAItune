import tensorflow as tf
from tensorflow.keras import layers
from keras import backend as k
import os
import datetime

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


def build_model_CNN(hp):
    min_conv1 = 9
    max_conv1 = 45
    min_conv2 = 9
    max_conv2 = 54
    ksize_min = 3
    ksize_max = 3

    #######Branch for accelerometer

    inputA = layers.Input(shape=(3, 50, 1))
    modelA = inputA

    for i in range(hp.Int(name='num_layers_A', min_value=1, max_value=2)):
        modelA = layers.Conv2D(filters=hp.Int(name='layer{}_CNN_A'.format(i),
                                              min_value=min_conv1,
                                              max_value=max_conv1,
                                              step=9,
                                              default=27),
                               padding='same',
                               activation='relu',
                               kernel_size=(3, 3))(modelA)

        for j in range(hp.Int(name='num_BN', min_value=0, max_value=1)):
            modelA = layers.BatchNormalization(
                name='BN_CNN1D_A{}'.format(i))(modelA)

    modelA = layers.GlobalMaxPool2D()(modelA)

    #########Branch for Gyroscope

    inputG = layers.Input(shape=(3, 50, 1))
    modelG = inputG

    for i in range(hp.Int(name='num_layers_G', min_value=1, max_value=2)):
        modelG = layers.Conv2D(filters=hp.Int(name='layer{}_CNN_G'.format(i),
                                              min_value=min_conv1,
                                              max_value=max_conv1,
                                              step=9,
                                              default=27),
                               kernel_size=(3, 3),
                               padding='same',
                               activation='relu')(modelG)

        for j in range(hp.Int(name='num_BN', min_value=0, max_value=1)):
            modelG = layers.BatchNormalization(
                name='BN_CNN1D_G{}'.format(i))(modelG)

    modelG = layers.GlobalMaxPool2D()(modelG)

    ### Concatenate Branches
    model = layers.Concatenate()([modelA, modelG])

    #####Add optional dense layers

    for i in range(hp.Int(name='num_layers', min_value=0, max_value=2)):

        model = layers.Dense(hp.Int(name='hidden_size_{}'.format(i),
                                    min_value=30,
                                    max_value=180,
                                    step=30,
                                    default=60),
                             activation='relu')(model)

        model = layers.Dropout(0.4)(model)

    output = layers.Dense(1)(model)

    ## define model and compile

    model = tf.keras.Model(inputs=[inputA, inputG], outputs=output)

    optimizer = tf.keras.optimizers.RMSprop(
        hp.Choice(name='learning_rate', values=[1e-3]))

    model.compile(loss='mean_absolute_error',
                  optimizer=optimizer,
                  metrics=['mae', 'mape', R_Square])

    tf.keras.utils.plot_model(
        model,
        to_file='logdir_CNN_hp/Model_{}.png'.format(
            #os.environ.get('USER'),
            datetime.datetime.now()),
        show_shapes=True)
    return model
