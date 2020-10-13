import shutil
import Data
import tensorflow as tf
from sklearn.model_selection import train_test_split
import kerastuner as kt
import model
from model import build_model_CNN
import pdb
import os
from plotter import hist_plotter
from kerastuner import HyperParameters

data_attr = [50, 0]


def reshaping(array):
    size_arr = len(array)
    #    print(array.shape)
    array = array.swapaxes(1, 2)
    #   print(array.shape)
    array = array.ravel().reshape(size_arr, data_attr[0], 3)
    #  print(array.shape)

    return (array)


physical_devices = tf.config.list_physical_devices('GPU')

try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    print('Could not initialize the tensorflow gpu')
    pass

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

print('Shape Features_A {}'.format(Features_TrainA.shape))
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


hp = HyperParameters()

folPath = 'logdir_CNN_hp'

if os.path.isdir(folPath):
    shutil.rmtree(folPath)
else:
    os.mkdir(folPath)

logdir = folPath

tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

tuner = kt.Hyperband(build_model_CNN,
                     objective='val_mae',
                     max_epochs=50,
                     factor=5,
                     hyperband_iterations=1,
                     seed=20,
                     tune_new_entries=True,
                     allow_new_entries=True)

print(tuner.search_space_summary())

tuner.search([Features_TrainA, Features_TrainG],
             Labels_TrainA,
             validation_data=([Features_ValA, Features_ValG], Labels_ValA),
             verbose=3,
             callbacks=[tensorboard_callback,
                        ClearTrainingOutput()])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)
hist = model.fit([Features_TrainA, Features_TrainG],
                 Labels_TrainA,
                 epochs=200,
                 validation_data=([Features_ValA, Features_ValG], Labels_ValA),
                 callbacks=[tensorboard_callback,
                            ClearTrainingOutput()],
                 verbose=3)

print("Found The Best Model")
print(model.evaluate([Features_TestA, Features_TestG], Labels_TestA))
print(best_hps)
hist_plotter(hist.history)
