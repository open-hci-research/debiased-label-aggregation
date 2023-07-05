#!/usr/bin/env python -W ignore::DeprecationWarning
# coding: utf-8

"""
Train a statistical feature-based model (see `model.py`).

External dependencies, to be installed e.g. via pip:
- numpy v1.10.0
- keras v2.1.2
- scikit-learn v0.19.1

Usage example:
$ python [-Wignore] trainf.py --files *.ndjson [--config some.json]

Author: Luis A. Leiva <luis@sciling.com>
Date: 2018
"""

from __future__ import print_function, division

import sys
import os
import argparse
import numpy as np
import json

from time import time
from datetime import datetime
from collections import defaultdict
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import pickle
import pandas as pd


import data
import ioutils
import modelf
from logger import LogClass

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", default=None, help="experiment configuration, in JSON or YAML format"
)
parser.add_argument("--files", required=True, nargs="+", help="ndjson files")
parser.add_argument("--datadir", help="path to an already processed dataset")
args = parser.parse_args()

logging = LogClass().getInstance()
logger = logging.logger

time_now = time()
# Default experiment configuration.
# Can be overriden by a external JSON file; see `--config` arg.
experiment_config = {
    # Use a known seed, for reproducibility.
    "rand_seed": 123456,
    # This file should be set in each experiment's `config.json` file.
    "dataset_file": "path/to/dataset.tsv",
    # If we already have preprocessed the dataset, only the model will be trained.
    "force_preprocessing": False,
    # It's often beneficial to have the same number of positive & negative classes.
    # This value MUST be > 1.0 to take effect, otherwise no balance will be applied.
    # Also, if you change this value *after* an experiment, you have to force data preprocessing.
    "max_imbalance_ratio": 1.25,
    # It's often beneficial to skip sketches that received only one vote.
    # Also, if you change this value *after* an experiment, you have to force data preprocessing.
    "skip_singleton_votes": True,
    # It's a good idea to compensate for highly skewed users' voting behavior.
    # Also, if you change this value *after* an experiment, you have to force data preprocessing.
    "weight_user_votes": True,
    # Understanding whether rescaling the weight by the total number of user votes is helpful
    "eq3": False,
    # It's often beneficial to train with scaled (whiten) feature vectors.
    "scale_features": True,
    # If you've already done grid search and know what are the best XGB params, set them here.
    "xgboost_params": None,
    # It's often beneficial to oversample the minority class,
    # rather than remove samples from the majority class.
    # Allowed values here are "SMOTE" (recommended) or "ADASYN".
    "over_sampling": "SMOTE",
    # It's often beneficial to remove irrelevant and highly correlated features.
    # Currently available methods for feature selection are "RFECV" and "LinearSVC".
    "feature_selection": "RFECV",
    # Remember the date the experiment was performed.
    "date": datetime.fromtimestamp(time_now).strftime("%Y-%m-%d %H:%M:%S"),
    "time": time_now,
}

if args.config is not None:
    # If user config is provided, override default options.
    # The config file can be either JSON or YAML, so check.
    if os.path.isfile(args.config):
        ext = os.path.splitext(args.config)[-1]
        if ext == ".json":
            user_config = ioutils.read_json(args.config)
        elif ext == ".yaml":
            user_config = ioutils.read_yaml(args.config)
        else:
            raise Exception(
                "Wrong config file. Only JSON and YAML formats are supported."
            )
    else:
        user_config = json.loads(args.config)
    experiment_config = data.merge_dicts(experiment_config, user_config)

# ingest raw image as input
RAW_IMAGE = True

# Inform early on about the experiment configuration.
print(experiment_config)

# Seed the random numbers generators.
# Note: Keras and sklearn use numpy internally.
rand_seed = experiment_config["rand_seed"]
np.random.seed(rand_seed)

# Let's use a common prefix to save the experiment results.
file_prefix = os.path.splitext(os.path.basename(experiment_config["dataset_file"]))[0]
if args.datadir:
    file_prefix = os.path.join(args.datadir, file_prefix)
    # It has no sense to force preprocessing AND load an already preprocessed dataset.
    if experiment_config["force_preprocessing"] is not None:
        logger.info(
            'Note: Ignoring "force_preprocessing" option since --datadir was provided.'
        )
        experiment_config["force_preprocessing"] = False

# Create filenames where the experiment results will be stored.
samples_file = "{}-samples.csv".format(file_prefix)
columns_file = "{}-columns.csv".format(file_prefix)
whiten_file = "{}-samples_whiten.csv".format(file_prefix)
labels_file = "{}-labels.csv".format(file_prefix)

# testing to use raw images as input
if RAW_IMAGE:
    keras_input_df = pd.read_csv(
        experiment_config["dataset_file"].replace(".tsv", ".csv")
    )
    keras_input_df = (
        keras_input_df[["id_peek", "id_peeked_sketch"]].drop_duplicates().copy()
    )
    keras_input_df["id_peek"] = keras_input_df["id_peek"].astype(str) + ".png"
    print("\n\n", keras_input_df["id_peek"].head())

    # find corresponding labels
    # getting new votes
    dataset = data.load_files(args.files)
    dscores = data.compute_scores(experiment_config)
    dataset = data.filter_dataset(dataset, dscores)
    labels = [d[1] for d in dataset]

    # TODO how to match label with md5 id like 5d7ad91e320d0dc2c624b3c1 in "id_peek" column

    # join data set back to csv for the original id_peek value

    print('prepped peek data and corresponding label')
    print(keras_input_df.head())
    print(keras_input_df.shape)

    print('value counts for keras input',keras_input_df['label'].value_counts())
    print("data joining ends")

if not os.path.isfile(labels_file) or experiment_config["force_preprocessing"]:
    dataset = data.load_files(args.files)
    dscores = data.compute_scores(experiment_config)
    dataset = data.filter_dataset(dataset, dscores)
    samples, labels = data.process_dataset(dataset, experiment_config)

    # Save columns as a 1-line CSV, since we're going to use them later for analysis (e.g. in R).
    columns = [k for k, v in samples[0].items()]
    samples = [[v for k, v in vec.items()] for vec in samples]
    logger.info("Writing CSV files ...")
    np.savetxt(labels_file, labels, delimiter=",", fmt="%d")
    np.savetxt(samples_file, samples, delimiter=",", fmt="%.3f")
    np.savetxt(columns_file, np.array([columns]), delimiter=",", fmt="%s")
    # Scaling feature vectors might help most of the time,
    # but remember that we need to deploy the model to production;
    # so we must also apply whitening in the `inspire.js` module.
    logger.info("Whitening samples ...")
    samples = preprocessing.scale(samples)
    np.savetxt(whiten_file, samples, delimiter=",", fmt="%.3f")
else:
    logger.info("Loading already preprocessed dataset ...")
    data_file = whiten_file if experiment_config["scale_features"] else samples_file
    samples = [
        list(map(float, line.split(","))) for line in ioutils.read_lines(data_file)
    ]
    columns = ioutils.read_file(columns_file).strip().split(",")
    labels = [int(line) for line in ioutils.read_lines(labels_file)]

# Silly check.
assert len(samples) == len(labels)

# Helper function to write feature importances.
def write_fimps(filename, feat_indices, feat_importances, colnames):
    rows = "Feature,Importance\n"
    for i, col in enumerate(colnames):
        if i in feat_indices:
            idx = feat_indices.index(i)
            val = feat_importances[idx]
            logger.info("Selected feature #{}: {} {}".format(i, col, val))
        else:
            val = 0
        rows += "{},{}\n".format(col, val)
    ioutils.write_file(filename, rows)


# Helper function to report summary statistics.
def metrics_summary(metrics_dict):
    result = []
    for name, (mean, sd, conf_interval) in metrics_dict.items():
        ci_lo, ci_hi = conf_interval
        ci_err = mean - ci_lo  # Assume symmetric CIs
        # Report results in percentage.
        line = "{:<17} {:.2f} (SD={:.2f}) [{:.2f} {:.2f}] SE={:.2f}".format(
            name, 100 * mean, 100 * sd, 100 * ci_lo, 100 * ci_hi, 100 * ci_err
        )
        result.append(line)
    return "\n".join(result)


## ---
## --- OPTION 1: Perform feature selection over the training data, then crossvalidate over unseen (test) data.
## --- This method is more generalizable but might achieve worse accuracy than other methods fitting the whole dataset (see below).
## ---

# logger.info('Creating partitions ...')
# # Note: The data will be oversampled when creating partitions, if specified in the experiment_config.
# X_train, y_train, X_test, y_test = data.create_partitions(samples, labels, experiment_config)
#
# logger.info('Selecting model features ...')
# feat_indices, feat_importances = data.select_features(X_train, y_train, experiment_config['feature_selection'])
# # Save (all) feature importances for running analysis with 3rd party software, e.g. R.
# fimp_file = '{}-fimps_{}.csv'.format(file_prefix, experiment_config['feature_selection'])
# write_fimps(fimp_file, feat_indices, feat_importances, columns)
# logger.info('Removing irrelevant model features ...')
# X_train = data.keep_features(X_train, feat_indices)
# X_test = data.keep_features(X_test, feat_indices)
#
# if experiment_config['xgboost_params'] is not None:
#     model = modelf.load(experiment_config['xgboost_params'])
# else:
#     logger.info('Searching for the best model parameters ...')
#     model = modelf.load()
#     best_params = modelf.grid_search(model, X_train, y_train)
#     # Reload model with the new params.
#     model = modelf.load(best_params)
#
# logger.info('Training model ...')
# modelf.fit(model, X_all, y_all)
#
# logger.info('Computing model metrics ...')
# train_metrics = modelf.crossvalidate(model, X_train, y_train)
# test_metrics = modelf.crossvalidate(model, X_test, y_test)
#
# print('TRAIN results, N={} observations'.format(len(y_train)))
# print(metrics_summary(train_metrics))
# print('TEST results, N={} observations'.format(len(y_test)))
# print(metrics_summary(test_metrics))


## ---
## --- OPTION 2: Perform feature selection over *all* the data, then crossvalidate.
## --- This method is less generalizable but might achieve better accuracy than the previous method.
## --- This is what we reported in our CSCW'19 submission.
## ---

# feature selection after oversampling
# if experiment_config['feature_selection'] is not None:
#     logger.info('Selecting model features ...')
#     feat_indices, feat_importances = data.select_features(samples, labels, experiment_config['feature_selection'])
#     # Write feature importances for running analysis with 3rd party software, e.g. R.
#     fimp_file = '{}-fimps_{}.csv'.format(file_prefix, experiment_config['feature_selection'])
#     write_fimps(fimp_file, feat_indices, feat_importances, columns)
#     logger.info('Removing irrelevant model features ...')
#     X_all = data.keep_features(samples, feat_indices)
# else:
X_all = samples

# if experiment_config['over_sampling'] is not None:
#     logger.info('Oversampling ...')
#     X_all, y_all = data.over_sample(X_all, labels, method=experiment_config['over_sampling'])
#     print('Oversampling results', np.unique(y_all,return_counts=True))
# else:
y_all = labels

# if experiment_config['xgboost_params'] is not None:
#     model = modelf.create(experiment_config['xgboost_params'])
# else:
#     logger.info('Searching for the best model parameters ...')
#     best_params = modelf.grid_search(model, X_all, y_all)
#     # Reload model with the new params.
#     model = modelf.create(best_params)

# logger.info('Training model ...')
# with open("{}{}_x_all.pickle".format(file_prefix,
#         str(experiment_config['weight_user_votes'])).lower(), "wb") as output_file:
#     pickle.dump(X_all, output_file)
# with open("{}{}_y_all.pickle".format(file_prefix,
#         str(experiment_config['weight_user_votes'])).lower(), "wb") as output_file:
#     pickle.dump(y_all, output_file)

# modelf.fit(model, X_all, y_all)
model = modelf.create(experiment_config["xgboost_params"])

logger.info("Computing model metrics ...")
# TODO should only oversampling when inside the crossvalidate function
all_metrics = modelf.crossvalidate_w_sampling(
    model,
    X_all,
    y_all,
    method="SMOTE",
    feature_selection=experiment_config["feature_selection"],
    columns=columns,
)

import pandas as pd

print("ALL results, N={} observations".format(len(y_all)))
print(all_metrics)

all_metrics = pd.DataFrame.from_dict(all_metrics, orient="index").T
for k, v in experiment_config.items():
    try:
        all_metrics[k] = v
    except:
        all_metrics[k] = str(v)

all_metrics["time"] = pd.to_datetime("today")

all_metrics.to_csv("output/results.csv", mode="a")

print()
logger.info("All done!")
