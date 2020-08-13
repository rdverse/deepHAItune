import tensorflow as tf
from tensorflow.keras import layers
from keras import backend as k


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
    modelA = layers.Conv2D(filters=hp.Int('layer1_CNN_A',
                                          min_value=min_conv1,
                                          max_value=max_conv1,
                                          step=9,
                                          default=27),
                           kernel_size=(3, 3),
                           padding='same',
                           activation='relu')(inputA)

    modelA = layers.Conv2D(filters=hp.Int('layer2_CNN_A',
                                          min_value=min_conv2,
                                          max_value=max_conv2,
                                          step=9,
                                          default=45),
                           kernel_size=(3, 3),
                           padding='same',
                           activation='relu')(modelA)

    modelA = layers.Flatten()(modelA)

    inputG = layers.Input(shape=(3, 150, 1))

    modelG = inputG
    modelG = layers.Conv2D(filters=hp.Int('layer1_CNN_G',
                                          min_value=min_conv1,
                                          max_value=max_conv1,
                                          step=9,
                                          default=27),
                           kernel_size=(3, 3),
                           padding='same',
                           activation='relu')(inputG)
    modelG = layers.Conv2D(filters=hp.Int('layer2_CNN_G',
                                          min_value=min_conv2,
                                          max_value=max_conv2,
                                          step=9,
                                          default=45),
                           kernel_size=(3, 3),
                           padding='same',
                           activation='relu')(modelG)
    # model = layers.Dropout(0.4)(model)
    modelG = layers.Flatten()(modelG)

    model = layers.Concatenate()([modelA, modelG])
    #   layers.concatenate(modelA,modelG)

    model = layers.Dropout(
        hp.Float('Dropout_val0_', min_value=0.2, max_value=0.4,
                 step=0.1))(model)

    model = layers.Dense(hp.Int('hidden_size0_',
                                min_value=90,
                                max_value=120,
                                step=30,
                                default=180),
                         activation='relu')(model)

    model = layers.Dropout(
        hp.Float('Dropout_val1_', min_value=0.3, max_value=0.4,
                 step=0.1))(model)

    model = layers.Dense(hp.Int('hidden_size1_',
                                min_value=15,
                                max_value=30,
                                step=15,
                                default=30),
                         activation='relu')(model)

    model = layers.Dropout(
        hp.Float('Dropout_val2_', min_value=0.3, max_value=0.4,
                 step=0.1))(model)

    output = layers.Dense(1)(model)

    model = tf.keras.Model(inputs=[inputA, inputG], outputs=output)

    optimizer = tf.keras.optimizers.RMSprop(
        hp.Choice('learning_rate', values=[1e-3]))

    model.compile(loss='mean_absolute_error',
                  optimizer=optimizer,
                  metrics=['mae', 'mape', R_Square])

    tf.keras.utils.plot_model(model,
                              to_file='/home/devesh/Pictures/Model.png',
                              show_shapes=True)
    return model
