from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from hyperparameters import hpts
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class skRandomForest():
    model = None

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

        grid = GridSearchCV(pipe, param_grid, verbose=3, n_jobs=-1)

        return grid
