import numpy as np
class StandardScaler:
    def __init__(self):
        self.train_mean = {'x':np.nan ,'y':np.nan, 'z':np.nan}
        self.train_std = {'x':np.nan ,'y':np.nan, 'z':np.nan}
        
    def fit(self,train_Features):
        self.train_mean['x'] = np.mean([f[0] for f in train_Features])
        self.train_mean['y'] = np.mean([f[1] for f in train_Features])
        self.train_mean['z'] = np.mean([f[2] for f in train_Features])
        
        self.train_std['x'] = np.std([f[0] for f in train_Features])
        self.train_std['y'] = np.std([f[1] for f in train_Features])
        self.train_std['z'] = np.std([f[2] for f in train_Features])
        
    def transform(self, Features):
        F_X = ([f[0] for f in Features] - self.train_mean['x'])/self.train_std['x']
        F_Y = ([f[1] for f in Features] - self.train_mean['y'])/self.train_std['y']
        F_Z = ([f[2] for f in Features] - self.train_mean['z'])/self.train_std['z']
        
        Features = np.array(list(zip(F_X,F_Y,F_Z)))
        return Features

