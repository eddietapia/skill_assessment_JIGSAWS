#!/usr/bin/env python
__author__ = "Ziheng Wang"
__email__ = "zihengwang@utdallas.edu"

import numpy as np
from collections import defaultdict


def one_hot(labels, n_class):
    """"One-hot encoding """
    labels= labels.reshape(-1)

    expansion = np.eye(n_class)
    y = expansion[:, labels-1].T
    assert y.shape[1] == n_class, "Wrong number of labels!"
    return y


def get_batches(X, y, batch_size):
    """ Return a generator for batches """
    n_batches = len(X) // batch_size
    X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]

    # Loop over batches and yield
    for b in range(0, len(X), batch_size):
        yield X[b:b+batch_size], y[b:b+batch_size]


def report2dict(cr):
    # Parse rows
    tmp = list ()
    for row in cr.split ("\n"):
        parsed_row = [x for x in row.split ("  ") if len (x) > 0]
        if len (parsed_row) > 0:
            tmp.append (parsed_row)
    # Store in dictionary
    measures = tmp[0]

    D_class_data = defaultdict (dict)
    for row in tmp[1:]:
        class_label = row[0]
        for j, m in enumerate (measures):
            D_class_data[class_label][m.strip ()] = float (row[j + 1].strip ())
    return D_class_data

