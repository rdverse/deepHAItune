class hpts():
    hyper = dict()

    def __init__(self):
        #For pca
        self.hyper = {
            'pca__n_components': [5, 25],
            'rf__max_depth': [60, 100, 120],
            'rf__min_samples_leaf': [3, 4, 5],
            'rf__min_samples_split': [8, 10, 15],
            'rf__n_estimators': [50, 125, 250, 900],

            ######## For Classifiers
            'svm__C': [0.001, 0.1, 1, 5, 10, 100],
            'svm__gamma': [0.1, 0.001, 1, 5, 10, 100],
            'svm__kernel': ['linear', 'rbf', 'sigmoid']
        }
