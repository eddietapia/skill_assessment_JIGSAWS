#!/usr/bin/env python
__author__ = "Ziheng Wang"
__email__ = "zihengwang@utdallas.edu"

import os
import glob
import numpy as np

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical


class TimeSeriesData(object):
    def __init__(self, windowSize, classNum):
        self.windowSize = windowSize
        self.stepSize = windowSize/6
        self.classNum = classNum
        self.metaData = {}

    def getNormalized(self, x1, x2):
        """" standardize data """
        mean = np.mean (x1, axis=0) # training data x1
        std = np.std (x1, axis=0)
        # 1e-9 avoid dividing by zero
        X1 = (x1 - mean) / (std + 1e-9)
        X2 = (x2 - mean) / (std + 1e-9)
        return X1, X2


    def getMetaData(self):
        temp_metaData = {}
        tempData = {}
        avgGRS = {}
        old_score = []
        old_str = []

        count = 0
        for file in glob.glob ('./DATA_LOSO/' + 'meta_file_' + '*' + '.txt'):
            for line in open (file, 'r'):
                line = line.strip ()
                if len (line) == 0:
                    break
                b = line.split ()
                surgery_name = b[0]
                skill_level = b[1]
                b = b[2:]
                scores = [int (e) for e in b]
                temp_metaData[surgery_name] = (skill_level, scores)

                str_trial = surgery_name.split ("_", 2)[-1]  # 'H004'
                str_sub = str_trial.split ("0", 1)[0]  # 'H'
                str_task = surgery_name.split (str_trial, 2)[0]
                str_tasksub = str_task + str_sub

                count = count + 1
                old_score = [sum (i) for i in zip (old_score, scores)]
                tempData[str_tasksub] = (skill_level, old_score, count)

                if str_tasksub != old_str:
                    count = 1
                    old_score = scores
                    old_str = str_tasksub

        for line in tempData.keys ():
            avgGRS[line] = [float ("{0:.2f}".format (a / (tempData[line][2]))) for a in tempData[line][1]]

        for file in glob.glob ('./DATA_LOSO/' + 'meta_file_' + '*' + '.txt'):
            for line in open (file, 'r'):
                line = line.strip ()
                if len (line) == 0:
                    break
                b = line.split ()
                surgery_name = b[0]
                skill_level = b[1]

                str_trial = surgery_name.split ("_", 2)[-1]  # 'H004'
                str_sub = str_trial.split ("0", 1)[0]  # 'H'
                str_task = surgery_name.split (str_trial, 2)[0]
                str_tasksub = str_task + str_sub

                self.metaData[surgery_name] = (skill_level, avgGRS[str_tasksub])


    def getSkillLevel(self, surgery_name):
        if self.metaData.__contains__ (surgery_name):
            if self.classNum == 3:
                score_grs = self.metaData[surgery_name][1][0]

                if surgery_name.__contains__ ('Knot_Tying'):
                    if score_grs <= 15:
                        y = 0
                    elif score_grs > 15 and score_grs < 20:
                        y = 1
                    else:
                        y = 2
                if surgery_name.__contains__ ('Suturing'):
                    if score_grs <= 19:
                        y = 0
                    elif score_grs > 19 and score_grs <= 24:
                        y = 1
                    else:
                        y = 2
                elif surgery_name.__contains__ ('Needle_Passing'):
                    if score_grs <= 15:
                        y = 0
                    elif score_grs > 15 and score_grs < 20:
                        y = 1
                    else:
                        y = 2
                return y
        return None

    # import raw data and get window slides
    def getKinematicData(self, url):
        dataX = np.zeros ((0, self.windowSize, 38))
        dataY = np.zeros ((0, 1))

        print ("loading data from url:\t", str (url))
        filelist = glob.glob (url + "*.txt")  # return a list of all txt files in the directory
        for file in filelist:
            file_name = os.path.basename(file)  # 'Needle_Passing_H004.txt'
            surgery_name = os.path.splitext(file_name)[0]  # 'Needle_Passing_H004'
            y = self.getSkillLevel(surgery_name)
            if y is None:
                continue

            # reading kinematic data from a file
            x = np.genfromtxt (file, delimiter='', dtype=np.float32)
            x = np.vstack ((x[:, 0:38], x[:, 38:76]))

            n = x.shape[0]  # get sequence length

            # sliding windows and get sub-sequences as individual samples.
            X = np.array ([x[i:i + self.windowSize] for i in np.arange (0, (n - self.windowSize), self.stepSize, dtype=int)])
            Y = np.array ([y for i in range (X.shape[0])])

            dataX = np.vstack ((dataX, X))
            dataY = np.vstack ((dataY, Y[:, None]))

        return dataX, dataY


    def train_val_test_split(self, url):
        _path = os.path.join (url, "train/")  # url = "./DATA/NP/"
        train, trainY = self.getKinematicData(_path)  # import training data and get sub-sequences as individual samples
        print ("load traing set DONE! (size:{})\n".format(train.shape))
        _path = os.path.join (url, "test/")
        test, testY = self.getKinematicData (_path)  # import testing data and get sub-sequences as individual samples
        print ("load testing set DONE! (size:{})\n".format(test.shape))

        # normalize data
        trainX, testX = self.getNormalized(train, test)  # Normalize train/test data
        assert trainX.shape[2] == testX.shape[2], "Mismatch in train/test channels!"

        # Train/Validation/Test split
        X_tr, X_val, lab_tr, lab_val = train_test_split (trainX, trainY, test_size=0.2, stratify=trainY, random_state=123)
        X_test = testX

        Y_tr = to_categorical (lab_tr.astype (int), self.classNum)  # One hot encoding
        Y_val = to_categorical (lab_val.astype (int), self.classNum)  # One hot encoding
        Y_test = to_categorical (testY.astype (int), self.classNum)  # One hot encoding

        return X_tr, X_val, X_test, Y_tr, Y_val, Y_test