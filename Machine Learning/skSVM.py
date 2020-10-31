from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from hyperparameters import hpts
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class skSVM():
    model = None
    name = 'SupportVectorMachine'

    def __init__(self):
        self.name = 'SVM'
        self.hp = hpts()
        self.hyper = self.hp.hyper
        self.model = self.build_model()

    def build_model(self):

        scale = StandardScaler()
        pca = PCA()

        svm = SVC(verbose=2, max_iter=3000)

        pipe = Pipeline([('scale', scale), ('pca', pca), ('svm', svm)])

        param_grid = {
            'pca__n_components': self.hyper['pca__n_components'],
            'svm__C': self.hyper['svm__C'],
            'svm__gamma': self.hyper['svm__gamma'],
            'svm__kernel': self.hyper['svm__kernel']
        }

        grid = GridSearchCV(pipe, param_grid, verbose=3, n_jobs=-1, cv=3)
        return grid
