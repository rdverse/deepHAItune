import Data
import tensorflow as tf
from sklearn.model_selection import train_test_split
import kerastuner as kt
import model
from model import build_model_CNN

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

input_dim = 3
batch_size = 64
units = 100
dense_units = 120
output_size = 1  # labels are from 0 to 9
#store in model

#model = build_model(allow_cudnn_kernel=True)
import IPython


class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)


tuner = kt.BayesianOptimization(
    build_model_CNN,
    objective='val_mae',
    #    max_epochs=333,
    max_trials=500,
    overwrite=True
    #min_epochs = 10,
    #hyperband_iterations=1,
)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
tuner.search([Features_TrainA, Features_TrainG],
             Labels_TrainA,
             validation_data=([Features_ValA, Features_ValG], Labels_ValA),
             epochs=500,
             verbose=3,
             callbacks=[callback, ClearTrainingOutput()])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

model = tuner.hypermodel.build(best_hps)
model.fit([Features_TrainA, Features_TrainG],
          Labels_TrainA,
          epochs=300,
          validation_data=([Features_ValA, Features_ValG], Labels_ValA),
          callbacks=[callback],
          verbose=1)

print(model.evaluate([Features_TestA, Features_TestG], Labels_TestA))
