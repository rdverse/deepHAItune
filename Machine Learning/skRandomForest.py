from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from hyperparameters import hpts
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import itertools
import os
import unittest
import ast


class skRandomForest():
    model = None
    name = 'RandomForest'

    def __init__(self):
        self.name = 'RandomForest'
        self.hp = hpts()
        self.hyper = self.hp.hyper
        self.model = self.build_model()

    def build_model(self):

        scale = StandardScaler()
        pca = PCA()
        rf = RandomForestRegressor(n_jobs=-1)

        pipe = Pipeline([('scale', scale), ('pca', pca), ('rf', rf)])

        param_grid = {
            'pca__n_components': self.hyper['pca__n_components'],
            'rf__max_depth': self.hyper['rf__max_depth'],
            'rf__min_samples_leaf': self.hyper['rf__min_samples_leaf'],
            'rf__min_samples_split': self.hyper['rf__min_samples_split'],
            'rf__n_estimators': self.hyper['rf__n_estimators']
        }

        grid = GridSearchCV(pipe, param_grid, verbose=3, n_jobs=-1, cv=3)

        return grid

    #Visualize parameters and next important thing is feature importance of the algorithms.

    def best_model(self, metric):

        path = os.path.join('Results', self.name + '.csv')
        res = pd.read_csv(path)

        if metric not in res.columns:
            raise ValueError(
                'Metric not found! Available metrics are {}'.format(
                    res.columns))

        if metric in ['mae', 'mape']:
            bestParmsIndex = res[[metric]].idxmin().item()
        else:
            bestParmsIndex = res[[metric]].idxmax().item()

        bestParams = res[['hyperparameters']].loc[bestParmsIndex].item()

        print('These are the best parameters {}'.format(bestParams))

        bestParams = ast.literal_eval(bestParams)

        checkPCA = [param.split('_') for param in bestParams]

        checkPCA = list(itertools.chain(*checkPCA))

        for key, val in bestParams.items():
            print(bestParams[key])
            print([val])
            bestParams[key] = [val]

        if 'pca' in checkPCA:
            pca = PCA()
            rf = RandomForestRegressor(n_jobs=-1)
            pipe = Pipeline(steps=[('pca', pca), ('rf', rf)])
            grid = GridSearchCV(pipe, bestParams, verbose=3, n_jobs=-1, cv=3)

        else:
            rf = RandomForestRegressor(n_jobs=-1)
            model = Pipeline(steps=[('rf', rf)])
            pipe = Pipeline(steps=[('rf', rf)])
            grid = GridSearchCV(pipe, bestParams, verbose=3, n_jobs=-1, cv=3)

        return (grid)
