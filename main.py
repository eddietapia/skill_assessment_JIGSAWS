#!/usr/bin/env python
__author__ = "Ziheng Wang"
__email__ = "zihengwang@utdallas.edu"

from metric_and_output import *
from train_and_summary import *

# parameters
MODE_NAMES = ['LOSO'] # ['LOSO', 'Holdout']
TASK_NAMES = ['SU', 'KT', 'NP']

N_CLASSES = 3
N_CHANNELS = 38
N_FOLDS = 5  # for LOSO mode only: 5 fold
SEQ_LEN = 90  # sequence length: 30, 60, 90
BATCH_SIZE = 600
LEARNING_RATE = 0.0001
EPOCH = 300

if N_CLASSES == 2:
    LABEL_COL_NAMES = ['Novice', 'Expert']
else:
    LABEL_COL_NAMES = ['Novice', 'Interm.', 'Expert']

cwd = os.getcwd() # get current workspace directory

for task_str in TASK_NAMES:
    for mode_str in MODE_NAMES:

        log_path = cwd+ '/RESULTS_'+ mode_str + '/' + task_str + '_W{}'.format (SEQ_LEN) # set log path

        if mode_str == 'LOSO':
            KFoldElapseT = []
            KFoldAcc = []
            prediction_seqs = None
            label_seqs = None

            prediction_seqs, label_seqs, KFoldElapseT, KFoldAcc = train_model_loso(log_path, task_str, N_FOLDS, SEQ_LEN, N_CLASSES, N_CHANNELS, LEARNING_RATE, EPOCH, BATCH_SIZE)

            output_report (log_path, task_str, mode_str, prediction_seqs, label_seqs, SEQ_LEN, LABEL_COL_NAMES,
                           np.mean (KFoldElapseT),
                           np.mean (KFoldAcc))

        if mode_str == 'Holdout':
            prediction_seqs, label_seqs, ElapseT, Acc = train_model_holdout(log_path, task_str, SEQ_LEN, N_CLASSES, N_CHANNELS, LEARNING_RATE,
                                                                            EPOCH, BATCH_SIZE)

            output_report (log_path, task_str, mode_str, prediction_seqs, label_seqs, SEQ_LEN, LABEL_COL_NAMES,
                           ElapseT, Acc)