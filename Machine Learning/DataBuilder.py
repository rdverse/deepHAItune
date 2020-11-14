from MLData import *
from tqdm import tqdm
from IPython.display import clear_output


class DataBuilder():
    def __init__(self, heuristic, applyFilter):
        print('building data')

        self.heu = heuristic
        self.applyFilter = applyFilter
        self.data_attr = {
            'data1': [50, 0],
            'data2': [100, 0],
            'data3': [150, 0],
        }

        if self.heu:
            del self.data_attr['data1']
            del self.data_attr['data2']

        self.best_results = {
            'conf': ['no', 'values'],
            'data': ['no', 'values'],
            'params': '',
            'results': '',
        }

        self.confs = dict()

        for key, val in tqdm(self.data_attr.items()):

            self.confs[key] = self.get_data(val)
            clear_output(wait=True)

    def get_data(self, da):
        FTrain_A, LTrain_A, FTest_A, LTest_A = dataset_main(
            da[0],
            da[1],
            'Yes',
            Resultant='no',
            heuristic=self.heu,
            applyFilter=self.applyFilter)

        FTrain_G, LTrain_G, FTest_G, LTest_G = dataset_main(
            da[0],
            da[1],
            'No',
            Resultant='no',
            heuristic=self.heu,
            applyFilter=self.applyFilter)

        FTrain_RA, LTrain_RA, FTest_RA, LTest_RA = dataset_main(
            da[0],
            da[1],
            Accel='Yes',
            Resultant='Yes',
            heuristic=self.heu,
            applyFilter=self.applyFilter)

        FTrain_RG, LTrain_RG, FTest_RG, LTest_RG = dataset_main(
            da[0],
            da[1],
            Accel='No',
            Resultant='Yes',
            heuristic=self.heu,
            applyFilter=self.applyFilter)

        # check the shape of this guy
        print(FTrain_A.shape)

        data = {
            'conf1': [FTrain_A, LTrain_A, FTest_A, LTest_A],
            'conf2': [FTrain_G, LTrain_G, FTest_G, LTest_G],
            'conf3': [FTrain_RA, LTrain_RA, FTest_RA, LTest_RA],
            'conf4': [FTrain_RG, LTrain_RG, FTest_RG, LTest_RG],
            'conf5': [
                np.hstack((FTrain_A, FTrain_G)), LTrain_G,
                np.hstack((FTest_A, FTest_G)), LTest_G
            ],
            'conf6': [
                np.hstack((FTrain_RA, FTrain_RG)), LTrain_RG,
                np.hstack((FTest_RA, FTest_RG)), LTest_RG
            ]
        }

        return data
