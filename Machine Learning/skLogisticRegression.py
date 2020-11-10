from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from hyperparameters import hpts
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


class skLogisticRegression():
    model = None
    name = 'LogisticRegression'

    def __init__(self, choice):
        self.name = 'SVM'
        self.hp = hpts()
        self.hyper = self.hp.hyper
        self.applyPCA = choice
        self.model = self.build_model()

    def build_model(self):

        scale = StandardScaler()
        pca = PCA()

        logreg = LogisticRegression(verbose=1, max_iter=3000)

        param_grid = {
            'logreg__penalty': self.hyper['logreg__penalty'],
            'logreg__solver': self.hyper['logreg__solver'],
        }

        if self.applyPCA:
            param_grid['pca__n_components'] = self.hyper['pca__n_components']
            pipe = Pipeline([('scale', scale), ('pca', pca),
                             ('logreg', logreg)])

        else:
            print(self.applyPCA)
            print(type(self.applyPCA))
            pipe = Pipeline([('scale', scale), ('logreg', logreg)])

        grid = GridSearchCV(pipe, param_grid, verbose=3, n_jobs=-1, cv=3)
        return grid
