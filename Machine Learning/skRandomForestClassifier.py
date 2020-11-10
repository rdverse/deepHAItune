from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from hyperparameters import hpts
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class skRandomForestClassifier():
    model = None
    name = 'RandomForest'

    def __init__(self, applyPCA=False):
        self.name = 'RandomForestClassifier'
        self.hp = hpts()
        self.hyper = self.hp.hyper
        self.applyPCA = applyPCA
        self.model = self.build_model()

    def build_model(self):

        scale = StandardScaler()
        pca = PCA()
        rf = RandomForestClassifier(n_jobs=-1, verbose=1)

        param_grid = {
            'rf__max_depth': self.hyper['rf__max_depth'],
            'rf__min_samples_leaf': self.hyper['rf__min_samples_leaf'],
            'rf__min_samples_split': self.hyper['rf__min_samples_split'],
            'rf__n_estimators': self.hyper['rf__n_estimators']
        }

        if self.applyPCA:
            param_grid['pca__n_components'] = self.hyper['pca__n_components']
            pipe = Pipeline([('scale', scale), ('pca', pca), ('rf', rf)])

        else:
            pipe = Pipeline([('scale', scale), ('pca', pca), ('rf', rf)])

        grid = GridSearchCV(pipe, param_grid, verbose=1, n_jobs=-1, cv=3)

        return grid

    #Visualize parameters and next important thing is feature importance of the algorithms.
