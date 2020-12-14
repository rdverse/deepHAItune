import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import re
import math
from entropy import spectral_entropy
import Heuristics
from scipy.signal import sosfilt, butter


def filter_butter(arr):
    but = butter(N=4, Wn=(0.5, 6), btype='bandpass', fs=51.1, output='sos')
    filtsig = sosfilt(but, arr)
    return (filtsig)


''' 

                                 ##### Make_Dataset #####

@args : None
@Returns (Features and Labels) of dimensions specified in next cell

Features : X Y Z axis concatenated together

Labels : Speed of each sample

Inputs : 1) Frame Size : size of the sliding window
         2) Overlap Percentage : percentage of overlap desired in each sliding frame
        
In function variables : 1) instances : Total no of frames that can be achieved
                        2) start_index : defines start of each sliding frame
                        3) end_index : defines end of each sliding frame
                        
Calculation :   1) instances = (Total Length(1535) - Frame_Length) / (Frame_Length(1 - Overlap_percent))
                
                2) For each Frame,
                   start_index = end_index - (Frame_size*overlap_percent)
                   end_index = start_index + Frame_size
                
                   Feature = X + Y + Z

'''


def Make_Dataset(Frame_size,
                 overlap_percent,
                 Accel,
                 Resultant,
                 heuristic=False,
                 applyFilter=False):
    Features = list()
    Labels = list()
    height_df = pd.read_csv('Cardio_Data/DataCollection.csv')
    count = 0

    if Accel.lower() == 'yes':
        print('Accel')
        attriutes = ['Accel_LN_X_CAL', 'Accel_LN_Y_CAL', 'Accel_LN_Z_CAL']
    else:
        print('Gyro')
        attriutes = ['Gyro_X_CAL', 'Gyro_Y_CAL', 'Gyro_Z_CAL']

    instances = int(
        math.floor(
            (1535 - Frame_size) / (Frame_size * (1 - overlap_percent / 100))))
    print(instances)
    PATH = 'Cardio_Data/Cleaned_data'
    profiles = os.listdir(PATH)

    for profile in profiles:
        speeds = os.listdir(PATH + '/' + profile)
        speeds = [s for s in speeds if float(s[:-4]) < 7.1]
        speeds = [s for s in speeds if float(s[:-4]) > 2.9]

        #speeds = [speed for speed in speeds if float(speed[0])<6]
        for speed in speeds:

            #Read the csv file using pandas
            df = pd.read_csv(PATH + '/' + profile + '/' + speed)

            #Get label for this speed
            Label = float(re.sub('\.csv$', '', speed))

            start_index = 0
            end_index = Frame_size
            #instances = 18
            for i in range(instances):

                feat_x = np.array(
                    df[attriutes[0]][start_index:end_index]).reshape(-1, 1)
                feat_y = np.array(
                    df[attriutes[1]][start_index:end_index]).reshape(-1, 1)
                feat_z = np.array(
                    df[attriutes[2]][start_index:end_index]).reshape(-1, 1)

                if applyFilter:
                    feat_y = filter_butter(feat_y)
                    feat_z = filter_butter(feat_z)
                    feat_x = filter_butter(feat_x)

                feat_r = np.sqrt(
                    np.square(feat_x) + np.square(feat_y) + np.square(feat_z))

                start_index = end_index - int(
                    Frame_size * overlap_percent / 100)
                end_index = start_index + Frame_size

                # Build array of features
                # print('Person : {} , Speed : {} , Start_index : {} , End_index : {}'.format(profile, speed[0:3],start_index,end_index))

                feat_x = np.array(feat_x)
                feat_y = np.array(feat_y)
                feat_z = np.array(feat_z)

                if heuristic:
                    feat_x = Heuristics.HeuristicBuilder(feat_x)
                    feat_y = Heuristics.HeuristicBuilder(feat_y)
                    feat_z = Heuristics.HeuristicBuilder(feat_z)
                    feat_r = Heuristics.HeuristicBuilder(feat_r)

                if Resultant.lower() == 'yes':
                    Feature = np.sqrt(
                        np.square(feat_x) + np.square(feat_y) +
                        np.square(feat_z)).flatten()
                else:
                    Feature = np.array([feat_x, feat_y, feat_z,
                                        feat_r]).flatten()
                Features.append((Feature))
                Labels.append(Label)

    return Features, Labels


'''
Train_Test_split for the data , 
default, test_size = 0.2
'''


def Split_Data(Features, Labels):

    #Train Test Split on data
    Features_Train, Features_Test, Labels_Train, Labels_Test = train_test_split(
        Features, Labels, shuffle=True, random_state=42, test_size=0.3)
    return (Features_Train, Features_Test, Labels_Train, Labels_Test)


def dataset_main(Frame_size, overlap_percent, Accel, Resultant, heuristic,
                 applyFilter):
    #Frame_size = int(input(" Enter the Frame Size for the dataset, range(50, 1500) : "))
    #overlap_percent = float(input("Enter the percentage of overlap desired for the dataset range(0,100): "))

    Features, Labels = Make_Dataset(Frame_size, overlap_percent, Accel,
                                    Resultant, heuristic, applyFilter)
    Features = np.array(Features)

    Features_Train, Features_Test, Labels_Train, Labels_Test = Split_Data(
        Features, Labels)
    #scaler = StandardScaler()
    #scaler.fit(Features_Train)
    #Features = scaler.transform(Features)

    Features_Train, Features_Test, Labels_Train, Labels_Test = Split_Data(
        Features, Labels)

    Labels_Train = np.array(Labels_Train).reshape(len(Labels_Train), 1)
    print('Train Labels shape   : {}'.format(Labels_Train.shape))
    Features_Train = np.array(Features_Train)
    # print(Features_Train[0])
    print('Train Features shape : {}'.format(Features_Train.shape))

    Labels_Test = np.array(Labels_Test).reshape(len(Labels_Test), 1)
    print('Test labels shape    : {}'.format(Labels_Test.shape))
    Features_Test = np.array(Features_Test)
    print('Test Features shape  : {}'.format(Features_Test.shape))
    return Features_Train, Labels_Train, Features_Test, Labels_Test


class StandardScaler:
    def __init__(self):
        self.train_mean = {'x': np.nan, 'y': np.nan, 'z': np.nan}
        self.train_std = {'x': np.nan, 'y': np.nan, 'z': np.nan}

    def fit(self, train_Features):
        #print(train_Features)
        self.train_mean['x'] = np.mean([f[0] for f in train_Features])
        self.train_mean['y'] = np.mean([f[1] for f in train_Features])
        self.train_mean['z'] = np.mean([f[2] for f in train_Features])

        self.train_std['x'] = np.std([f[0] for f in train_Features])
        self.train_std['y'] = np.std([f[1] for f in train_Features])
        self.train_std['z'] = np.std([f[2] for f in train_Features])

    def transform(self, Features):
        F_X = ([f[0] for f in Features] -
               self.train_mean['x']) / self.train_std['x']
        F_Y = ([f[1] for f in Features] -
               self.train_mean['y']) / self.train_std['y']
        F_Z = ([f[2] for f in Features] -
               self.train_mean['z']) / self.train_std['z']

        Features = np.array(list(zip(F_X, F_Y, F_Z)))
        return Features


''' 

                                 ##### Leave one Person out #####

'''


def Dataset_Leave_Person(Frame_size, overlap_percent, test_profile, Accel):
    Features_Train = list()
    Labels_Train = list()
    Features_Test = list()
    Labels_Test = list()

    count = 0

    instances = int(
        math.floor(
            (1535 - Frame_size) / (Frame_size * (1 - overlap_percent / 100))))
    print(instances)
    PATH = 'Cardio_Data/Cleaned_data'
    profiles = os.listdir(PATH)
    for profile in profiles:
        speeds = os.listdir(PATH + '/' + profile)
        speeds = [s for s in speeds if float(s[:-4]) < 7.1]
        speeds = [s for s in speeds if float(s[:-4]) > 2.9]

        #print(speeds)
        #speeds = [speed for speed in speeds if float(speed[0])<6]
        if Accel == 'Yes':
            print('Accel')
            attriutes = ['Accel_LN_X_CAL', 'Accel_LN_Y_CAL', 'Accel_LN_Z_CAL']
        else:
            print('Gyro')
            attriutes = ['Gyro_X_CAL', 'Gyro_Y_CAL', 'Gyro_Z_CAL']

        for speed in speeds:

            #Read the csv file using pandas
            df = pd.read_csv(PATH + '/' + profile + '/' + speed)

            #Get label for this speed
            Label = float(re.sub('\.csv$', '', speed))

            start_index = 0
            end_index = Frame_size
            #instances = 18
            for i in range(instances):

                feat_x = np.array(
                    df[attriutes[0]][start_index:end_index]).reshape(-1, 1)
                feat_y = np.array(
                    df[attriutes[1]][start_index:end_index]).reshape(-1, 1)
                feat_z = np.array(
                    df[attriutes[2]][start_index:end_index]).reshape(-1, 1)

                start_index = end_index - int(
                    Frame_size * overlap_percent / 100)
                end_index = start_index + Frame_size
                # Build array of features
                # print('Person : {} , Speed : {} , Start_index : {} , End_index : {}'.format(profile, speed[0:3],start_index,end_index))
                Feature = np.array([feat_x, feat_y, feat_z]).flatten()

                if profile == test_profile:
                    Features_Test.append((Feature))
                    Labels_Test.append(Label)

                else:
                    Features_Train.append((Feature))
                    Labels_Train.append(Label)


#     scaler = StandardScaler()
#     scaler.fit(Features_Train)
#     Features_Train = scaler.transform(Features_Train)
#     Features_Test = scaler.transform(Features_Test)

    Labels_Train = np.array(Labels_Train).reshape(len(Labels_Train), 1)
    print('Train')
    print('Train Labels shape   : {}'.format(Labels_Train.shape))
    Features_Train = np.array(Features_Train)
    print('Train Features shape : {}'.format(Features_Train.shape))
    print('Test')
    Labels_Test = np.array(Labels_Test).reshape(len(Labels_Test), 1)
    print('Test labels shape    : {}'.format(Labels_Test.shape))
    Features_Test = np.array(Features_Test)
    print('Test Features shape  : {}'.format(Features_Test.shape))

    return Features_Train, Labels_Train, Features_Test, Labels_Test
