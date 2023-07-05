#!/usr/bin/env python -W ignore::DeprecationWarning
# coding: utf-8

'''
Model query test.

External dependencies, to be installed e.g. via pip:
- none

Usage example:
$ python [-Wignore] queryf.py model.xgb samples.file labels.file

Author: Luis A. Leiva <luis@sciling.com>
Date: 2018
'''

from __future__ import print_function, division

import sys
import os
import ioutils
import modelf

model_file, whiten_file, labels_file = sys.argv[1:4]

samples = [list(map(float, line.split(','))) for line in ioutils.read_lines(whiten_file)]
labels = [int(line) for line in ioutils.read_lines(labels_file)]

mymodel = modelf.load_file(model_file)
metrics = modelf.crossvalidate(mymodel, samples, labels)
for name, (mean, sd, conf_interval) in metrics.items():
    print('{:<17} {:.2f} {:.2f} [{:.2f} {:.2f}]'.format(name, 100*mean, 100*sd, 100*conf_interval[0], 100*conf_interval[1]))
