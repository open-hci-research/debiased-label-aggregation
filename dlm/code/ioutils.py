#!/usr/bin/env python
# coding: utf-8

'''
I/O utilities.

External dependencies, to be installed e.g. via pip:
- dill v0.15.2

Author: Luis A. Leiva <luis@sciling.com>
Date: 2018
'''

from __future__ import print_function, division

import os
import json
import yaml
import dill as pickle


def read_file(filename, mode='r'):
    '''
    Read file contents.
    Use mode='rb' to read a binary file.
    '''
    with open(filename, mode) as in_file:
        return in_file.read()


def read_lines(filename, mode='r'):
    '''
    Read file contents line by line.
    Use mode='rb' to read a binary file.
    '''
    return read_file(filename, mode).splitlines()


def read_pickle(filename):
    '''
    Read serialized file contents.
    Requires dill's pickle module imported.
    '''
    # Pickle's serialized files are binary by definition.
    with open(filename, 'rb') as in_file:
        return pickle.load(in_file)


def read_json(filename):
    '''
    Read json file contents.
    Requires native json module imported.
    '''
    # JSON files are not binary by definition.
    with open(filename, 'r') as in_file:
        return json.load(in_file)


def read_yaml(filename):
    '''
    Read yaml file contents.
    Requires native yaml module imported.
    '''
    # YAML files are not binary by definition.
    # Also: https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
    with open(filename, 'r') as in_file:
        return yaml.load(in_file, Loader=yaml.FullLoader)


def ensure_file(filename):
    '''
    Ensure that filename can be written.
    '''
    file_dir = os.path.dirname(filename)
    if file_dir and not os.path.exists(file_dir):
        os.makedirs(file_dir)


def write_file(filename, content, mode='w'):
    '''
    Write contents to file.
    Use mode='wb' to write a binary file.
    '''
    ensure_file(filename)
    with open(filename, 'w') as out_file:
        out_file.write(content + '\n')


def write_pickle(filename, content):
    '''
    Write serialized contents to file.
    Requires Dill (pickle on steroids) module.
    '''
    ensure_file(filename)
    # Pickle's serialized files are binary by definition.
    with open(filename, 'wb') as out_file:
        pickle.dump(content, out_file)


def write_json(filename, content):
    '''
    Write json contents to file.
    Requires native json module.
    '''
    ensure_file(filename)
    # JSON files are not binary by definition.
    with open(filename, 'w') as out_file:
        out_file.write(json.dumps(content) + '\n')


def write_yaml(filename, content):
    '''
    Write yaml contents to file.
    Requires native yaml module.
    '''
    ensure_file(filename)
    # JSON files are not binary by definition.
    with open(filename, 'w') as out_file:
        out_file.write(yaml.dump(content) + '\n')
