#There are two kinds of data - leave one out and all include
#this program should return list of features
# choices - dataset type
# sensor type

#All person included

import math
import os
import re

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from StandardScaler import StandardScaler


def get_data(Frame_size, overlap_percent, Accel):
    #Frame_size = int(input(" Enter the Frame Size for the dataset, range(50, 1500) : "))
    #overlap_percent = float(input("Enter the percentage of overlap desired for the dataset range(0,100): "))

    Features, Labels = Make_Dataset(Frame_size, overlap_percent, Accel)
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


def Make_Dataset(Frame_size, overlap_percent, Accel):
    Features = list()
    Labels = list()
    Ids = list()

    count = 0
    if Accel == 'Yes':
        print('Accel')
        attriutes = ['Accel_LN_X_CAL', 'Accel_LN_Y_CAL', 'Accel_LN_Z_CAL']
    else:
        print('Gyro')
        attriutes = ['Gyro_X_CAL', 'Gyro_Y_CAL', 'Gyro_Z_CAL']

    print('new Data')
    PATH = 'Cardio_Data/Cleaned_data'
    profiles = os.listdir(PATH)

    samplePath = PATH + '/' + profiles[0]
    samplePath = samplePath + '/' + os.listdir(samplePath)[0]
    dataLen = len(pd.read_csv(samplePath))

    instances = int(
        math.floor((dataLen - Frame_size) / (Frame_size *
                                             (1 - overlap_percent / 100))))
    print(instances)

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

                start_index = end_index - int(
                    Frame_size * overlap_percent / 100)
                end_index = start_index + Frame_size
                # Build array of features
                # print('Person : {} , Speed : {} , Start_index : {} , End_index : {}'.format(profile, speed[0:3],start_index,end_index))
                Feature = np.array([feat_x, feat_y, feat_z])

                Features.append((Feature))
                Labels.append(Label)
                Ids.append(profile)

    return np.array(Features), np.array(Labels), np.array(Ids)


'''
Train_Test_split for the data , 
default, test_size = 0.2

'''
# def Split_Data(Features,Labels):
#     import numpy as np
#     from sklearn.model_selection import StratifiedKFold
#     X = np.zeros(Features.shape[0])
#     y = Labels
#     y_binned = np.digitize(y, bins)
#     skf = StratifiedKFold(n_splits=2)
#     skf.get_n_splits(X, y_binned)

#     print(skf)

#     for train_index, test_index in skf.split(X, y):
#         print("TRAIN:", train_index, "TEST:", test_index)
#         Features_Train, Features_Test = X[train_index], X[test_index]
#         Labels_Train, Labels_Test = y[train_index], y[test_index]
#         break
#     return(Features_Train, Features_Test, Labels_Train, Labels_Test)


def Split_Data(Features, Labels):

    #Train Test Split on data
    Features_Train, Features_Test, Labels_Train, Labels_Test = train_test_split(
        Features, Labels, shuffle=True, random_state=42, test_size=0.2)
    return (Features_Train, Features_Test, Labels_Train, Labels_Test)
