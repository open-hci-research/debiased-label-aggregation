#!/usr/bin/env python -W ignore::DeprecationWarning
# coding: utf-8

'''
Convert in-classroom dataset format to the same format of the crowdsourcing dataset.

External dependencies, to be installed e.g. via pip:
- none

Usage example:
$ python [-Wignore] transform.py dataset.csv

Author: Luis A. Leiva <luis@sciling.com>
Date: 2018
'''

from __future__ import division, print_function

import sys
import csv
import json
import os
from datetime import datetime

datafile = sys.argv[1]
prefix = os.path.splitext(datafile)[0]
ndjsonfile = prefix + '.ndjson'
tsvfile = prefix + '.tsv'

for f in [ndjsonfile, tsvfile]:
    try:
        os.remove(f)
    except OSError:
        pass

# Ensure we re-generate the ndjson and TSV files everytime we process the same dataset.
with open(ndjsonfile, 'w') as f:
    f.truncate()
with open(tsvfile, 'w') as f:
    f.truncate()

with open(datafile) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # Use the same format as in Quickdraw dataset.
        entry = {
          "word": "NA",
          "countrycode": "US",
          "timestamp": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.Z'),
          "recognized": True,
          "key_id": row['id_peeked_sketch'],
          "drawing": json.loads(row['peeked_sketch_strokes_quickdraw'])
        }
        with open(ndjsonfile, 'a') as myfile:
            entry = json.dumps(entry, separators=(',', ':'))
            myfile.write(entry + '\n')

        # Use the same format for groundtruth labels as well.
        with open(tsvfile, 'a') as myfile:
            score = 1 if row['vote'] == 'Interesting' else 0
            entry = '\t'.join(map(str, [row['timestamp'], row['id_user'], row['id_peeked_sketch'], 0, score, 'NA', 'NA', 1]))
            myfile.write(entry + '\n')
