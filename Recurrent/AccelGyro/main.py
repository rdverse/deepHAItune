import IPython
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

folPath = 'logdir_CNN_hp'

if os.path.isdir(folPath):
    shutil.rmtree(folPath)
    os.mkdir(folPath)

else:
    os.mkdir(folPath)

logdir = folPath

pysical_devices = tf.config.list_physical_devices('GPU')

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

print('Shape Features_A {}'.format(Features_TrainA.shape))
input_dim = 3
batch_size = 64
units = 100
dense_units = 120
output_size = 1  # labels are from 0 to 9
#store in model

#model = build_model(allow_cudnn_kernel=True)


class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)


hp = HyperParameters()

tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

tuner = kt.Hyperband(build_model_CNN,
                     objective='val_mae',
                     max_epochs=200,
                     factor=2,
                     hyperband_iterations=1,
                     seed=20,
                     tune_new_entries=True,
                     allow_new_entries=True)

print(tuner.search_space_summary())
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 patience=50)

reduceLRplateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                       min_lr=1e-6,
                                                       patience=25,
                                                       factor=0.9,
                                                       verbose=1)

tuner.search([Features_TrainA, Features_TrainG],
             Labels_TrainA,
             validation_data=([Features_ValA, Features_ValG], Labels_ValA),
             verbose=3,
             callbacks=[
                 tensorboard_callback, earlystopping, reduceLRplateau,
                 ClearTrainingOutput()
             ])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)
hist = model.fit([Features_TrainA, Features_TrainG],
                 Labels_TrainA,
                 epochs=300,
                 validation_data=([Features_ValA, Features_ValG], Labels_ValA),
                 callbacks=[tensorboard_callback,
                            ClearTrainingOutput()],
                 verbose=3)

print("Found The Best Model")
print(model.evaluate([Features_TestA, Features_TestG], Labels_TestA))
print(best_hps.values)
hist_plotter(hist.history)
