class hpts():
    hyper = dict()

    def __init__(self):
        #For pca
        self.hyper = {'pca__n_components': [5, 10, 25, 30]}
