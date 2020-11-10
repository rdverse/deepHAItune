from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from hyperparameters import hpts
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class skKNeighborsClassifier():
    model = None
    name = 'KNeighborsClassifier'

    def __init__(self, choice):
        self.name = 'kNeighborsClassifier'
        self.hp = hpts()
        self.hyper = self.hp.hyper
        self.applyPCA = choice
        self.model = self.build_model()

    def build_model(self):

        scale = StandardScaler()
        pca = PCA()

        KM = KMeans()

        param_grid = {
            'kNNC__n_neighbors': self.hyper['kNNC__n_neighbors'],
            'kNNC__leaf_size': self.hyper['kNNC__leaf_size'],
            'kNNC__weights': self.hyper['kNNC__weights'],
            'kNNC__algorithm': self.hyper['kNNC__algorithm'],
        }

        if self.applyPCA:
            pipe = Pipeline([('scale', scale), ('pca', pca), ('kNNC', kNNC)])
            param_grid['pca__n_components'] = self.hyper['pca__n_components']

        else:
            pipe = Pipeline([('scale', scale), ('kNNC', kNNC)])

        grid = GridSearchCV(pipe, param_grid, verbose=3, n_jobs=-1, cv=3)
        return grid
