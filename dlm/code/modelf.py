#!/usr/bin/env python -W ignore::DeprecationWarning
# coding: utf-8

'''
Model utilities for binary classification.

External dependencies, to be installed e.g. via pip:
- numpy v1.14.1
- scikit-learn v0.19.1
- xgboost v0.71

Author: Luis A. Leiva <luis@sciling.com>
Date: 2018
'''

from __future__ import print_function, division

import sys

from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from time import time
import xgboost as xgb
import numpy as np

import ioutils
import mathlib
from data import over_sample, select_features, keep_features

# These default params were estimated in a crossvalidation experiment using the `grid_search()` function above.
# See https://xgboost.readthedocs.io/en/latest/parameter.html
# Also see https://www.dataiku.com/learn/guide/code/python/advanced-xgboost-tuning.html
default_params = {
  'n_estimators': 100,
  'max_depth': 5,
  'n_jobs': 1,
  'verbosity': 0,
  'use_label_encoder': False,
}


def create(custom_params=None):
    '''
    Create xgboost classifier.
    '''
    if type(custom_params) is dict:
        print('[XGB model] Setting custom params:', custom_params, file=sys.stderr)
        par = default_params.copy()
        par.update(custom_params)
    else:
        par = default_params.copy()
    # Instantiate the model with params as kwargs.
    # See https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier
    return xgb.XGBClassifier(**par)


def load_file(filename):
    '''
    Load model from file.
    '''
    clf = create()
    # The XGBoost API is a bit unconventional.
    # See https://github.com/dmlc/xgboost/issues/706#issuecomment-167253974
    booster = xgb.Booster()
    booster.load_model(filename)
    clf._Booster = booster

    return clf


def grid_search(model, X, y):
    '''
    Find the best parameter combination.
    '''
    X, y = np.array(X), np.array(y)

    grid_params = {
      'max_depth': [2, 5, 10, 20],
      'n_estimators': [50, 100, 200]
    }
    gridz = GridSearchCV(model, grid_params, cv=10, n_jobs=1)
    gridz.fit(X, y)
    print('[XGB model] Best params:', gridz.best_params_, file=sys.stderr)
    return gridz.best_params_


def fit(model, X, y):
    '''
    Train the model.
    '''
    X, y = np.array(X), np.array(y)

    # We can reuse the params we used when creating the model before.
    par = model.get_params()
    # However, some values were set to `None` so we have to filter them out
    # in order to reuse the XGBoost training API.
    par = { k:v for k,v in par.items() if v is not None }
    # Notice that the `train` method uses params as dict instead of kwargs.
    # See https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.training
    bst = xgb.train(par, xgb.DMatrix(X, label=y), num_boost_round=100, verbose_eval=False)
    out_file = 'model-{}.xgb'.format(int(time()))
    bst.save_model(out_file)

    return model.fit(X, y, verbose=False)


def crossvalidate(model, X, y):
    '''
    Compute model metrics with crossvalidation.
    '''
    X, y = np.array(X), np.array(y)

    metrics = ['balanced_accuracy', 'accuracy',
        'average_precision', 'precision', 'recall', 'f1',
        'f1_weighted', 'f1_micro', 'f1_macro','roc_auc', 'jaccard']
    scores = cross_validate(model, X, y, scoring=metrics, cv=10, return_train_score=False)
    result = {}
    for metric, values in scores.items():
        # Remove non-relevant metrics such as fit_time, score_time, etc.
        if metric.endswith('_time'):
            continue
        # Remove the "test_" prefix added by sklearn to computed values.
        name = metric.split('test_')[1]
        # Now compute mean/sd/cis for each metric.
        vals = values.tolist()
        mean, sd = mathlib.mean_sd(vals)
        ci95 = mathlib.ci_normal(len(vals), mean, sd)
        result[name] = (mean, sd, ci95)
    return result

# import pickle
# with open('sketchy_dec6_dec13_peeks_updatedtrue_x_all.pickle','rb') as f:
#     X = pickle.load(f)
#     np.array(X).shape
# with open('sketchy_dec6_dec13_peeks_updatedtrue_y_all.pickle','rb') as f:
#     y = pickle.load(f)

def crossvalidate_w_sampling(model, X, y, method='SMOTE', feature_selection=None, columns=None):
    """
    Creates folds manually, and upsamples within each fold.
    Returns an array of validation (recall) scores
    """
    from sklearn.model_selection import KFold
    from sklearn import metrics

    if feature_selection:
        # select features
        # X_upsample, y_upsample = over_sample(X, y, method=method)
        feat_indices, feat_importances = select_features(X, y, feature_selection)
        X = keep_features(X, feat_indices)
        # TODO print out and save the features that we end up keeping, and their importance
        print('Features kept are:', [columns[idx] for idx in feat_indices])
        # print('Features kept are:', feat_indices)

    # validation
    X, y = np.array(X), np.array(y)
    cv = KFold(n_splits=5, random_state=42, shuffle=True)

    scores = []

    for train_fold_index, val_fold_index in cv.split(X, y):
        # Get the training data
        X_train_fold, y_train_fold = X[train_fold_index,:], y[train_fold_index]
        # Get the validation data
        X_val_fold, y_val_fold = X[val_fold_index,:], y[val_fold_index]

        # Upsample only the data in the training section
        X_train_fold_upsample, y_train_fold_upsample = over_sample(
                X_train_fold, y_train_fold, method=method)

        # Fit the model on the upsampled training data
        model_obj = model.fit(X_train_fold_upsample, y_train_fold_upsample)

        # Score the model on the (non-upsampled) validation data
        # TODO add scorer
        scores.append({
            'balanced_accuracy':metrics.balanced_accuracy_score(y_val_fold, model_obj.predict(X_val_fold)),
            'accuracy':metrics.accuracy_score(y_val_fold, model_obj.predict(X_val_fold)),
            'average_precision':metrics.average_precision_score(y_val_fold, model_obj.predict(X_val_fold)),
            'precision':metrics.precision_score(y_val_fold, model_obj.predict(X_val_fold)),
            'recall':metrics.recall_score(y_val_fold, model_obj.predict(X_val_fold)),
            'f1':metrics.f1_score(y_val_fold, model_obj.predict(X_val_fold)),
            'f1_weighted':metrics.f1_score(y_val_fold, model_obj.predict(X_val_fold),average='weighted'),
            'f1_micro':metrics.f1_score(y_val_fold, model_obj.predict(X_val_fold),average='micro'),
            'f1_macro':metrics.f1_score(y_val_fold, model_obj.predict(X_val_fold),average='macro'),
            'roc_auc':metrics.roc_auc_score(y_val_fold, model_obj.predict(X_val_fold)),
            'jaccard':metrics.jaccard_score(y_val_fold, model_obj.predict(X_val_fold)),
        })
    overall_score = {}
    for k in scores[0].keys():
        overall_score[k] = np.mean([d[k] for d in scores])

    return overall_score


def accuracy(model, X, y):
    '''
    Evaluate the model.
    '''
    X, y = np.array(X), np.array(y)

    # See https://github.com/dmlc/xgboost/issues/2073
    model._le = LabelEncoder().fit(y)

    y_pred = model.predict(X)
    return accuracy_score(y, y_pred)


def predict(model, X):
    '''
    Predict class probabilities for unseen data.
    Given a feature vector [x1,...,xN] and M classes, the output is [prob_class1, ..., prob_classM].
    '''
    X = np.array(X)

    predictions = model.predict_proba(X)
    return predictions
