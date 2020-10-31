from sklearn.metrics import accuracy_score
import pandas as pd
import csv
import numpy as np
import os


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(mape)
    return (mape)


class Evaluator():
    def __init__(self, algoName, data_attr, grid, configData, confData, split):
        self.algoName = algoName
        self.data_attr = data_attr
        self.grid = grid
        self.configData = configData
        self.split = split

        self.X_Train = confData[0]
        self.y_Train = self.__alter_labels(confData[1].ravel())
        self.X_Test = confData[2]
        self.y_Test = self.__alter_labels(confData[3].ravel())

        self.model = None
        self.preds = None

        print('Shape of the input Data is {}'.format(self.X_Train.shape))
        self.__check_files()

    def __alter_labels(self, labels):
        labels = [0 if label <= self.split else 1 for label in labels]
        return np.array(labels)

    def __check_files(self):
        PATH = 'Results/' + self.algoName + '.csv'

        if not os.path.isdir('Results'):
            os.mkdir('Results')

    def train_model(self):
        self.model = self.grid.fit(self.X_Train, self.y_Train)
        self.preds = self.model.predict(self.X_Test)

    def _results(self):
        accuracy = accuracy_score(self.preds, self.y_Test)
        return accuracy

    def _gethps(self):
        return (self.model.best_params_)

    def store(self):
        PATH = 'Results/' + self.algoName + '.csv'
        print(PATH)
        if not os.path.exists(PATH):
            with open(PATH, 'w') as csvFile:
                print('writing csv file')
                csvWriter = csv.writer(csvFile)
                columns = [
                    'algoName', 'Frame', 'hyperparameters', 'configData',
                    'accuracy', 'split'
                ]
                csvWriter.writerow(columns)

        accuracy = self._results()
        print(self._results())
        ##Save list
        save_list = [
            self.algoName,
            str(self.data_attr),
            str(self._gethps()), self.configData, accuracy, self.split
        ]

        with open(PATH, 'a') as csvFile:
            csvWriter = csv.writer(csvFile)
            csvWriter.writerow(save_list)
