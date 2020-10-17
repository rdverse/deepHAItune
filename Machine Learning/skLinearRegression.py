from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA, NMF
from hyperparameters import hpts
from sklearn.model_selection import GridSearchCV


class skLinearRegression():
    model = None

    def __init__(self):
        self.name = 'LinearRegression'
        self.hp = hpts()
        self.model = self.build_model()

    def build_model(self):

        pca = PCA()
        reg = LinearRegression()

        pipe = Pipeline([('pca', pca), ('reg', reg)])

        param_grid = {
            'pca__n_components': self.hp.hyper['pca__n_components'],
        }

        grid = GridSearchCV(pipe, param_grid, n_jobs=-1)

        return grid
