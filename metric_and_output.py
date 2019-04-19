#!/usr/bin/env python
__author__ = "Ziheng Wang"
__email__ = "zihengwang@utdallas.edu"

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn import metrics
from utilities import report2dict


def compute_metrics(prediction_seqs, label_seqs, LABEL_COL_NAMES):
    """ Compute metrics averaged over sequences
    """
    rpt_metrics = metrics.classification_report(label_seqs, prediction_seqs, target_names= LABEL_COL_NAMES)
    rpt_confusion = metrics.confusion_matrix (label_seqs, prediction_seqs)

    return rpt_metrics, rpt_confusion


""" loss plot """
def plot_loss(log_path, epochs, loss, val_loss, LOSO_MODE, fold):
    plt.figure ()

    plt.plot(epochs, loss, '--', label= 'Training Loss',lw =2)
    plt.plot(epochs, val_loss, '-', label= 'Validation Loss',lw =2)
    plt.title('Training and validation loss', fontsize=18)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout ()

    if LOSO_MODE == False:
        plt.savefig (os.path.join (log_path + '_loss' + '.pdf'),
                     bbox_inches="tight")
    else:
        plt.savefig (os.path.join (log_path + '_loss' + '_fold{}'.format(fold) + '.pdf'),
                     bbox_inches="tight")


""" accuracy plot """
def plot_accuracy(log_path, epochs, acc, val_acc, LOSO_MODE, fold):
    plt.figure ()

    plt.plot (epochs, acc, '--', label='Training Accuracy', lw=2)
    plt.plot (epochs, val_acc, '-', label='Validation Accuracy', lw=2)
    plt.title ('Training and validation accuracy', fontsize=18)
    plt.xlabel ('Epochs', fontsize=14)
    plt.ylabel ('Accuracy', fontsize=14)
    plt.legend (fontsize=12)
    plt.tight_layout ()

    if LOSO_MODE == False:
        plt.savefig (os.path.join (log_path + '_acc' + '.pdf'),
                     bbox_inches="tight")
    else:
        plt.savefig (os.path.join (log_path + '_acc' + '_fold{}'.format(fold) + '.pdf'),
                     bbox_inches="tight")


""" confusion matrix plot """
def plot_confusion_matrix(log_path,
                          cm,
                          classes,
                          normalize= True,
                          cmap= plt.cm.PuBu):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure ()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix', fontsize = 16)

    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.clim(0, 1)
    plt.ylabel('True label', fontsize =12)
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.tight_layout ()

    plt.savefig (os.path.join (log_path + '_confusionmat' + '.pdf'),
                 bbox_inches="tight")


def plot_classif_metrics(log_path,
                               cr,
                               with_avg_total= False,
                               cmap= plt.cm.PuBu):
    lines = cr.split('\n')

    classes = []
    plotMat = []

    for line in lines[2 : (len(lines) - 3)]:
        #print(line)
        t = line.split()
        # print(t)
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        print(v)
        plotMat.append(v)

        if with_avg_total:
            aveTotal = lines[len (lines) - 1].split ()
            classes.append ('avg/total')
            vAveTotal = [float (x) for x in t[1:len (aveTotal) - 1]]
            plotMat.append (vAveTotal)


    plotMat = np.array (plotMat).T  # convert list to array, transform array

    plt.figure ()

    plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
    plt.title('Classification report', fontsize = 16)
    plt.colorbar()
    plt.clim(0, 1)

    thresh = plotMat.max() / 2.
    for i, j in itertools.product (range (plotMat.shape[0]), range (plotMat.shape[1])):
        plt.text (j, i, plotMat[i, j],
                  horizontalalignment="center",
                  color="white" if plotMat[i, j] > thresh else "black")

    x_tick_marks = np.arange(len(classes))
    y_tick_marks = np.arange(3)
    plt.xticks (x_tick_marks, classes)
    plt.yticks(y_tick_marks, ['precisiosn', 'recall', 'f1-score'])
    plt.xlabel('Predicted Labels', fontsize = 12)
    plt.ylabel('Metrics', fontsize = 12)

    plt.tight_layout ()

    plt.savefig (os.path.join (log_path + '_metricsreport' + '.pdf'),
                 bbox_inches="tight")


def output_report(log_path, task_str, mode_str, prediction_seqs, label_seqs, SEQ_LEN, LABEL_COL_NAMES, ElapseT, Acc):
    print ('OUTPUT REPORT...\n')

    rpt_metrics, rpt_confusion = compute_metrics (prediction_seqs, label_seqs, LABEL_COL_NAMES)

    plot_classif_metrics (log_path, rpt_metrics)
    plot_confusion_matrix(log_path, rpt_confusion, LABEL_COL_NAMES)

    report_path = os.path.join (log_path + '_report.txt')
    with open (report_path, "w") as text_file:
        print ("TASK NAME:{}\t TRAINING MODE:{}\t".format (task_str, mode_str), file=text_file)
        print ("SLILDING WINDOW:{}\n".format (SEQ_LEN), file=text_file)
        print ("METRICS:\n{}\n".format (pd.DataFrame (report2dict (rpt_metrics)).T), file=text_file)
        print ("CONFUSION MATRIX:\n{}\n".format (pd.DataFrame (rpt_confusion, columns=None)), file=text_file)
        print ("TESTING AVG RUNNING TIME:{:.6f}\n".format (ElapseT), file=text_file)
        print ('TESTING AVG ACCURACY:{:.6f})\n'.format (Acc), file=text_file)

    print ('FINISHED!\n')