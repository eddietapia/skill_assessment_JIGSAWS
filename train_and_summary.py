#!/usr/bin/env python
__author__ = "Ziheng Wang"
__email__ = "zihengwang@utdallas.edu"

import time
import numpy as np
import pandas as pd
from keras import models, layers, optimizers
from load_and_import import *
from metric_and_output import plot_accuracy, plot_loss, plot_confusion_matrix

global LOSO_MODE

def model_cnn(input_size, output_size, learning_rate):
    model = models.Sequential ()

    model.add (layers.Conv1D (filters=38, kernel_size=2, strides=1,
                              padding='same', kernel_initializer='glorot_normal', activation='relu',
                              input_shape= input_size))
    model.add (layers.MaxPool1D (pool_size=2, strides=2, padding='same'))
    model.add (layers.Dropout (rate=0.2))

    model.add (layers.Conv1D (filters=76, kernel_size=2, strides=1,
                              padding='same', activation='relu'))
    model.add (layers.MaxPool1D (pool_size=2, strides=2, padding='same'))
    model.add (layers.Dropout (rate=0.2))

    model.add (layers.Conv1D (filters=152, kernel_size=2, strides=1,
                              padding='same', activation='relu'))
    model.add (layers.MaxPool1D (pool_size=2, strides=2, padding='same'))
    model.add (layers.Dropout (rate=0.2))

    model.add (layers.Flatten ())
    model.add (layers.Dense (units=64, activation='relu'))
    model.add (layers.Dropout (rate=0.5))

    model.add (layers.Dense (units=32, activation='relu'))
    model.add (layers.Dropout (rate=0.5))

    model.add (layers.Dense (units=output_size, activation='softmax'))

    adam = optimizers.Adam (lr= learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile (optimizer= adam,
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

    return model


def model_train_and_test(url, log_path, SEQ_LEN, N_CLASSES, N_CHANNELS, LEARNING_RATE, EPOCH, BATCH_SIZE, LOSO_MODE, fold=None):
    timeseriesdata = TimeSeriesData (windowSize=SEQ_LEN, classNum=N_CLASSES)
    timeseriesdata.getMetaData ()

    # split dataset for training, validation and testing
    X_tr, X_val, X_test, y_tr, y_val, y_test = timeseriesdata.train_val_test_split(url= url)

    # create model
    model = model_cnn(input_size= (SEQ_LEN, N_CHANNELS), output_size= N_CLASSES, learning_rate= LEARNING_RATE)

    # fit model
    history = model.fit(X_tr, y_tr, epochs= EPOCH, batch_size= BATCH_SIZE, validation_data=(X_val, y_val))

    # evaluate on test set
    test_acc = model.evaluate (X_test, y_test)
    print ('\nTESTING ACCURACY:{:.6f}'.format (test_acc[1]) + '\n')

    # visualize training/val loss, accuracy
    epochs = history.epoch
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']

    plot_loss (log_path, epochs, loss, val_loss, LOSO_MODE, fold) # plot loss
    plot_accuracy (log_path, epochs, acc, val_acc, LOSO_MODE, fold) # plot accuracy

    # get running time on test set
    startT = time.perf_counter()
    y_pred = model.predict_classes(X_test)
    elapseT = (time.perf_counter() - startT)

    print ('\nTRAINING FINISHED!\n')
    print('====================================\n')

    return model, history, y_pred, y_test, elapseT, test_acc[1]


def train_model_holdout(log_path, task_str, SEQ_LEN, N_CLASSES, N_CHANNELS, LEARNING_RATE,
                                                                EPOCH, BATCH_SIZE):
    print ('Holdout MODE\n')
    url = './DATA_Holdout/' + task_str + '/'

    model, history, y_pred, y_test, elapseT, test_acc = model_train_and_test (url, log_path, SEQ_LEN,
                                                                              N_CLASSES, N_CHANNELS, LEARNING_RATE, EPOCH, BATCH_SIZE,
                                                                              LOSO_MODE=False, fold= None)

    prediction_seqs = y_pred
    label_seqs = np.argmax (y_test, axis=1)

    # export history
    pd.DataFrame (history.history).to_csv (log_path + '_history'  + '_holdout' + '.csv')

    return prediction_seqs, label_seqs, elapseT, test_acc


def train_model_loso(log_path, task_str, N_FOLDS, SEQ_LEN, N_CLASSES, N_CHANNELS, LEARNING_RATE, EPOCH, BATCH_SIZE):

    KFoldElapseT = []
    KFoldAcc = []
    prediction_seqs = None
    label_seqs = None

    for fold in range (1, N_FOLDS + 1):
        fold_str = 'fold_{}'.format (fold)
        LOSO_MODE = True
        print ('LOSO mode: ' + fold_str + '\n')
        url = './DATA_LOSO/' + task_str + '/' + fold_str + '/'

        model, history, y_pred, y_test, elapseT, test_acc = model_train_and_test (url, log_path, SEQ_LEN, N_CLASSES, N_CHANNELS, LEARNING_RATE, EPOCH, BATCH_SIZE, LOSO_MODE, fold)
        if prediction_seqs is None:
            prediction_seqs = y_pred
            label_seqs = np.argmax (y_test, axis=1)
        else:
            prediction_seqs = np.concatenate ([prediction_seqs, y_pred])
            label_seqs = np.concatenate ([label_seqs, np.argmax (y_test, axis=1)])

        KFoldElapseT.append (elapseT)  # return running time for all folds in LOSO mode
        KFoldAcc.append (test_acc)

        # export history
        pd.DataFrame (history.history).to_csv (log_path + '_history' + '_LOSO_fold{}'.format (fold) + '.csv')
    return prediction_seqs, label_seqs, KFoldElapseT, KFoldAcc