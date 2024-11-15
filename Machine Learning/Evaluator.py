from sklearn.metrics import mean_absolute_error, r2_score
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
    def __init__(self, algoName, data_attr, grid, configData, confData):
        self.algoName = algoName
        self.data_attr = data_attr
        self.grid = grid
        self.configData = configData
        self.X_Train = confData[0]
        self.y_Train = confData[1].ravel()
        self.X_Test = confData[2]
        self.y_Test = confData[3].ravel()

        self.model = None
        self.preds = None

        self.__check_files()

    def __check_files(self):
        PATH = 'Results/' + self.algoName + '.csv'

        if not os.path.isdir('Results'):
            os.mkdir('Results')

        # if os.path.exists(PATH):
        #     os.remove(PATH)

    def train_model(self):
        self.model = self.grid.fit(self.X_Train, self.y_Train)
        self.preds = self.model.predict(self.X_Test)

    def evaluate(self):
        mae = mean_absolute_error(self.y_Test, self.preds)
        mape = mean_absolute_percentage_error(self.y_Test, self.preds)
        r2 = r2_score(self.y_Test, self.preds)
        self.results = [mae, mape, r2]

    def _results(self):
        mae = mean_absolute_error(self.y_Test, self.preds)
        mape = mean_absolute_percentage__error(self.y_Test, self.preds)
        r2 = r2_score(self.y_Test, self.preds)
        return mae, mape, r2

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
                    'mae', 'mape', 'r2'
                ]
                csvWriter.writerow(columns)

        mae, mape, r2 = self._results()
        print(self._results())
        ##Save list
        save_list = [
            self.algoName,
            str(self.data_attr),
            str(self._gethps()), self.configData, mae, mape, r2
        ]

        with open(PATH, 'a') as csvFile:
            csvWriter = csv.writer(csvFile)
            csvWriter.writerow(save_list)
