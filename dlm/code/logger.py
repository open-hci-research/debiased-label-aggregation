#!/usr/bin/env python
# coding: utf-8

'''
Logging facility.
Will create a folder with the log session ID, together with a .log file with the same ID.
The experiment config, output, models, etc. will be stored in that folder.

Usage example:
>>> from logger import LogClass
>>> logger = LogClass().getInstance('/tmp/logs').logger
>>> logger.info('Hello')

External dependencies, to be installed e.g. via pip:
- none

Author: Luis A. Leiva <luis@sciling.com>
Date: 2018
'''

from __future__ import print_function, division

import sys
import logging
from time import time
from os import path, makedirs


def singleton(cls):
    '''
    Singleton class factory.
    '''
    instances = {}
    # Implement the Singleton pattern to allow exactly only one instance per class.

    def getInstance():
        '''
        Get singleton instance.
        '''
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return getInstance


@singleton
class LogClass(object):
    '''
    Create logger instances.
    '''

    def __init__(self):
        self.id = int(time())
        self.dir = None
        self.logger = None
        print('Initialized logger ID {}'.format(self.id), file=sys.stderr)

    def getInstance(self, log_dir=None):
        '''
        Get logger instance.
        '''
        # Avoid duplicated log lines.
        if self.logger is None:
            # Configure logging instance.
            # Use a nice, parseable log format for dates.
            logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s')
            # Use the ID as logger namespace (must be string).
            self.logger = logging.getLogger(str(self.id))
            if log_dir is not None:
                self.dir = log_dir
                self.ensuredir()
                # Dump logging messages to file, in addition to stdout/stderr.
                # Write file in the same dir level as `self.dir`, for easy listing.
                handler = logging.FileHandler('{}/{}.log'.format(self.dir, self.id))
                self.logger.addHandler(handler)
        # Allow access to the whole instance.
        return self

    def ensuredir(self):
        '''
        Create the specified dir.
        '''
        out_dir = '{}/{}'.format(self.dir, self.id)
        if not path.exists(out_dir):
            makedirs(out_dir)
