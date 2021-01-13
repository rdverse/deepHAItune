import math
import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from StandardScaler import StandardScaler


def dataset_main(Frame_size, overlap_percent, Accel, split=False):
    #Frame_size = int(input(" Enter the Frame Size for the dataset, range(50, 1500) : "))
    #overlap_percent = float(input("Enter the percentage of overlap desired for the dataset range(0,100): "))

    Features, Labels, pIDs = Make_Dataset(Frame_size, overlap_percent, Accel)
    Features = np.array(Features)
    Labels = np.array(Labels).reshape(-1, 1)
    pIDs = np.array(pIDs)
    
    print(Features.shape)
    print(Labels.shape)
    print(pIDs.shape)

    print(Features[:2])
    print(Labels[:3])
    print(pIDs[:3])

    return Features, Labels, pIDs


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
    Features, Labels, pIDs = list(), list(), list()

    attributes = get_attributes(Accel)
    PATH = 'Cardio_Data/Cleaned_data'

    dataLen = get_dataLen(PATH)

    instances = int(
        math.floor((dataLen - Frame_size) / (Frame_size *
                                             (1 - overlap_percent / 100))))
    print(instances)

    for root, dirs, files in os.walk(PATH):
        for file in files:

            #Get label for this speed
            Label = float(re.sub('\.csv$', '', file))

            if (Label > 7.1 or Label < 2.9):
                print('skipping this file : {}'.format(Label))
                continue

            pID = root.split('/')[-1]

            filePath = os.path.join(root, file)

            #Read the csv file using pandas
            df = pd.read_csv(filePath)

            start_index = 0
            end_index = Frame_size

            for i in range(instances):

                feat_x = np.array(
                    df[attributes[0]][start_index:end_index]).reshape(-1, 1)
                feat_y = np.array(
                    df[attributes[1]][start_index:end_index]).reshape(-1, 1)
                feat_z = np.array(
                    df[attributes[2]][start_index:end_index]).reshape(-1, 1)

                Feature = np.array([feat_x, feat_y, feat_z])

                start_index = end_index - int(
                    Frame_size * overlap_percent / 100)

                end_index = start_index + Frame_size

                Features.append((Feature))
                Labels.append(Label)
                pIDs.append(pID)

    return Features, Labels, pIDs


def get_dataLen(PATH):
    profiles = os.listdir(PATH)
    samplePath = PATH + '/' + profiles[0]
    samplePath = samplePath + '/' + os.listdir(samplePath)[0]
    dataLen = len(pd.read_csv(samplePath))
    return (dataLen)


def get_attributes(Accel):
    if Accel == 'Yes':
        print('Accel')
        attributes = ['Accel_LN_X_CAL', 'Accel_LN_Y_CAL', 'Accel_LN_Z_CAL']
    else:
        print('Gyro')
        attributes = ['Gyro_X_CAL', 'Gyro_Y_CAL', 'Gyro_Z_CAL']

    return (attributes)


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
