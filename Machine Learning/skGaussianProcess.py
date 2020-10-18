from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from hyperparameters import hpts
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.preprocessing import StandardScaler


class skGaussianProcess():
    model = None
    name = 'GaussianProcess'

    def __init__(self):
        self.name = 'GaussianProcess'
        self.hp = hpts()
        self.hyper = self.hp.hyper
        self.model = self.build_model()

    def build_model(self):

        scale = StandardScaler()
        pca = PCA()
        kernel = DotProduct() + WhiteKernel()
        gp = GaussianProcessRegressor(
            kernel=kernel,
            random_state=29,
        )

        pipe = Pipeline([
            ('scale', scale),
            #('pca', pca),
            ('gp', gp)
        ])

        param_grid = {
            #   'pca__n_components': self.hyper['pca__n_components'],
            'gp__n_restarts_optimizer': [0, 2]
            #'rf__max_depth': self.hyper['rf__max_depth'],
            #'rf__min_samples_leaf': self.hyper['rf__min_samples_leaf'],
            #'rf__min_samples_split': self.hyper['rf__min_samples_split'],
            #'rf__n_estimators': self.hyper['rf__n_estimators']
        }

        grid = GridSearchCV(pipe, param_grid, verbose=3, n_jobs=-1, cv=3)

        return grid
