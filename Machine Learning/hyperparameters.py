class hpts():
    hyper = dict()

    def __init__(self):
        #For pca
        self.hyper = {
            'pca__n_components': [5, 10, 25, 30],
            'rf__max_depth': [60, 80, 100, 120],
            'rf__min_samples_leaf': [3, 4, 5],
            'rf__min_samples_split': [8, 10, 12],
            'rf__n_estimators': [50, 100, 200, 300, 1000]
        }
