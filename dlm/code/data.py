#!/usr/bin/env python -W ignore::DeprecationWarning
# coding: utf-8

"""
Data processing utilities.

External dependencies, to be installed e.g. via pip:
- numpy v1.14.1
- keras v2.1.2
- scikit-learn v0.19.1
- imblearn v0.3.3

Author: Luis A. Leiva <luis@sciling.com>
Date: 2018
"""

from __future__ import print_function, division

import warnings

from sklearn.utils.fixes import delayed

warnings.filterwarnings(action="ignore", category=DeprecationWarning)

import sys
import os
import json
import math
from collections import defaultdict

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SVMSMOTE, ADASYN

import mathlib
import ioutils
from featurizer import featurize
import matplotlib.pyplot as plt
from logger import LogClass
import logging

logging.getLogger("matplotlib.font_manager").disabled = True

logging = LogClass().getInstance()

logger = logging.logger

# Ensure default configuration.
default_config = {
    "train_ratio": 0.7,
    "sequence_size": None,
    "skip_singleton_votes": True,
}


def compute_scores(experiment_config={}):
    """
    Compute sketch votes, takng into account user voting behavior and past history.
    """
    experiment_config = merge_dicts(default_config, experiment_config)
    gt_datafile = experiment_config["dataset_file"]

    sketch_users = defaultdict(lambda: defaultdict(list))
    sketch_votes = defaultdict(lambda: defaultdict(list))
    # Actually we have to consider the prior probability of voting a sketch as inspirational.
    # See `priors.pdf` paper where I explained the relevant math.
    num_user_votes = defaultdict(int)
    num_user_yes_votes = defaultdict(int)

    logger.info("Reading sketch votes from {} ...".format(gt_datafile))
    for line in ioutils.read_file(gt_datafile).splitlines():
        if not line:
            continue
        # Unpack log line parts.
        (
            timestamp,
            sess_id,
            key_id,
            is_control,
            score,
            display_label,
            actual_label,
            finish_ratio,
        ) = line.split("\t")
        # Skip control sketches, since they were randomly labeled at runtime.
        if float(is_control) == 1:
            continue
        # Cast values.
        ratio = float(finish_ratio)
        score = int(score)
        # There can be multiple scores per sketch and also for different finish ratios,
        # so store all values and then decide what to do.
        sketch_votes[key_id][ratio].append(score)
        sketch_users[key_id][ratio].append(sess_id)
        # Each user might vote one option much more often than the other,
        # so we have to take that into account as well.
        num_user_votes[sess_id] += 1
        # TODO this value shouldn't be too large
        num_user_yes_votes[sess_id] += int(score > 0)

    # print out to np array for plotting distribution
    plt.close("all")
    plt.figure()
    plt.hist([num_votes for user, num_votes in num_user_votes.items()], bins=20)
    file_ext = (
        experiment_config["dataset_file"].replace(".tsv", "").replace("/", "")
        + str(experiment_config["weight_user_votes"])
        + str(experiment_config["eq3"])
    )
    plt.savefig(f"output/dist_num_user_votes_{file_ext}.png", dpi=200)

    logger.info("Computing user voting behavior ...")
    # Now compute the probability of voting a sketch as inspirational.
    user_prob_yes = {}
    for user, num_votes in num_user_votes.items():
        num_yes = num_user_yes_votes[user]
        user_prob_yes[user] = num_yes / num_votes

    logger.info("Computing class histogram ...")
    # Compute histogram of scored sketches per removal ratio.
    sc_freq = defaultdict(int)
    for sketch_id, ratios in sketch_votes.items():
        for ratio, scores in ratios.items():
            # Skip sketches that had only one score.
            if experiment_config["skip_singleton_votes"] and len(scores) == 1:
                continue
            sc_freq[ratio] += 1

    # Ensure classes are sort of balanced, i.e. at least there's a min number of samples in each class.
    sc_values = sc_freq.values()
    if not sc_values:
        logger.info(
            'Note: Ignoring "skip_singleton_votes" option since all sketches have only one vote.'
        )
        experiment_config["skip_singleton_votes"] = False

    if experiment_config["max_imbalance_ratio"] > 1.0 and sc_values:
        max_num_sketches = min(sc_values) * experiment_config["max_imbalance_ratio"]
        for ratio, freq in sc_freq.items():
            logger.info(
                "Sketches with ratio {} have {} values. Max value will be capped to {}".format(
                    ratio, freq, max_num_sketches
                )
            )

    # Now collapse scores per sketch and finish ratio.
    final_scores = defaultdict(lambda: defaultdict(int))
    sc_count = defaultdict(int)
    # Also save the final dataset we're dealing with.
    dataset = ["\t".join(["key_id", "finish_ratio", "num_votes", "inspirational"])]
    for sketch_id, ratios in sketch_votes.items():
        for ratio, votes in ratios.items():
            # Skip sketches that had only one vote.
            if experiment_config["skip_singleton_votes"] and len(votes) == 1:
                continue
            # Get the user IDs who voted this particular sketch at this particular finish ratio.
            users = sketch_users[sketch_id][ratio]
            users_0, users_1 = [], []
            for user, vote in zip(users, votes):
                if experiment_config["weight_user_votes"]:
                    # Compensate user voting bias.
                    p_yes = user_prob_yes[user]
                    weight = p_yes if vote == 0 else 1 - p_yes
                else:
                    weight = 1.0  # will be majority voting
                # Users who scored more should have more importance (Eq.4),
                # so add the number of total votes to the current weight.
                # If we don't care about the user's voting history (Eq.3) then set the number of votes to 1.
                if experiment_config[
                    "eq3"
                ]:  # TODO: Rename this setting to `no_voting_history` or similar.
                    tup = (weight, 1)
                else:
                    tup = (weight, num_user_votes[user])

                if vote == 0:
                    users_0.append(tup)
                else:
                    users_1.append(tup)
            # At this point we can safely call `reweighted_majority_voting()`,
            # no matter if we're testing majority voting or relabeling.
            score = reweighted_majority_voting(users_0, users_1)
            # In case of ties (none score), ignore sketch.
            if score is None:
                continue
            # Balance sketches by removal ratio: Exit as soon as we have a balanced dataset.
            sc_count[ratio] += 1
            if (
                experiment_config["max_imbalance_ratio"] > 1.0
                and sc_count[ratio] > max_num_sketches
            ):
                break
            dataset.append("\t".join(map(str, [sketch_id, ratio, len(votes), score])))
            final_scores[sketch_id][ratio] = score
    # Maybe this preprocessed dataset file can be used by other researchers to train their own classification models.
    # Note: The current dir is relative to the caller program, which I assume to be the parent dir of `./code` dir.
    tmp_data_file = os.path.splitext(gt_datafile)[0] + ".tmp"
    ioutils.write_file(tmp_data_file, "\n".join(dataset))

    return final_scores


def majority_voting(num_0s, num_1s):
    """
    Compute class name based on the number of votes in each option.
    """
    if num_0s == num_1s:
        # Can't decide on ties.
        return None
    return np.argmax([num_0s, num_1s])


def weighted_majority_voting(weights_0s, weights_1s):
    """
    Take into account the number of weights in each option.
    """
    raise NotImplementedError(
        "Deprecated method. Use `reweighted_majority_voting()` instead."
    )

    return majority_voting(sum(weights_0s), sum(weights_1s))


def reweighted_majority_voting(tuples_0, tuples_1):
    """
    Take into account both the number of weights and number of votes in each option.
    """
    tots_0 = sum_votes(tuples_0)
    tots_1 = sum_votes(tuples_1)
    return majority_voting(tots_0, tots_1)


def sum_votes(vote_tuples):
    """
    Perform weighted sum of votes.
    """
    tots = 0
    for (weight, num_votes) in vote_tuples:
        tots += weight * num_votes
    return tots


def merge_dicts(source, target):
    """
    Merge two JSON objects. Target properties will replace source properties.
    Example:
    >>> dictA = {'foo': 1, 'bar': 2}
    >>> dictB = {'foo': 33, 'qux': True}
    >>> merge_dicts(dictA, dictB)
    {'qux': True, 'foo': 33, 'bar': 2}
    """
    merged = source.copy()
    merged.update(target)
    return merged


def load_dir(dirname, ext=".ndjson"):
    """
    Load data from directory.
    The result is a list of <strokes,label> tuples.
    TODO: Allow different formats (e.g. json, csv) in the same dir.
    """
    logger.info("Loading {} files from {} ...".format(ext, dirname))

    res = []
    for dirpath, dnames, fnames in os.walk(dirname):
        res += load_files(fnames, ext)
    return res


def load_files(file_list, ext=".ndjson"):
    """
    Load data from file list.
    The result is a list of <strokes,label> tuples.
    TODO: Allow different formats (e.g. json, csv).
    """
    logger.info("Loading {} {} files ...".format(len(file_list), ext))

    res = []
    for f in file_list:
        name, fmt = os.path.splitext(f)
        if fmt != ext:
            continue
        for obj in parse_ndjson_file(f):
            res.append(obj)
    return res


def filter_dataset(dataset, scores):
    """
    Ensure groudtruth data points in dataset.
    """
    res = []
    for obj in dataset:
        gts = ensure_groundtruth(obj, scores)
        if gts is None:
            continue
        # Concat groundtruth scores.
        res += gts
    return res


def ensure_groundtruth(obj, scores):
    """
    Read groundtruth info from sketch data.
    """
    key_id = obj["key_id"]
    if key_id not in scores:
        return None
    sketch = obj["drawing"]
    tuples = []
    for ratio, score in scores[key_id].items():
        # Some sketches were trimmed, so use the very same version for training.
        vector = rm_strokes(sketch, ratio)
        vector = as_2d_vector(sketch)
        tuples.append((vector, score, ratio))
    return tuples


def rm_strokes(sketch, finish_ratio):
    """
    Remove strokes from sketch according to given finish ratio, in [0, 1].
    """
    finish_ratio = float(finish_ratio)
    num_strokes = len(sketch)
    new_num_strokes = math.ceil(num_strokes * finish_ratio)
    vector = []
    for i, stroke in enumerate(sketch):
        if i + 1 > new_num_strokes:
            continue
        vector.append(stroke)
    return vector


def parse_ndjson_file(filename):
    """
    Process an ndjson file and return all sketches inside that file.
    """
    name, fmt = os.path.splitext(filename)
    res = []
    if fmt == ".ndjson":
        lines = ioutils.read_file(filename).splitlines()
        for line in lines:
            g = json.loads(line)
            res.append(g)
        return res
    else:
        sys.exit("Unsupported sketch format: {}.".format(fmt))


def as_2d_vector(sketch):
    """
    Reformat quickdraw sketch: from `[ [x1, ..., xN], [y1, ..., yN] ]` to `[ [x1,y1], ..., [xN,yN] ]`.
    """
    vector = []
    points = []
    for strokes in sketch:
        xs, ys = strokes
        for i, x in enumerate(xs):
            y = ys[i]
            points.append([x, y])
        vector.append(points)
    return vector


def as_1d_vector(sketch):
    """
    Reformat quickdraw sketch: from `[ [x1, ..., xN], [y1, ..., yN] ]` to `[x1,y1, ..., xN,yN]`.
    """
    vector = []
    points = []
    for strokes in sketch:
        xs, ys = strokes
        for i, x in enumerate(xs):
            y = ys[i]
            points.append(x)
            points.append(y)
        vector.append(points)
    return vector


def flatten(sketch):
    """
    Flatten sketch to a single array of points.
    """
    points = []
    for strokes in sketch:
        points += strokes
    return points


def split_samples(samples, perc=0.5):
    """
    Takes as input a group of sequences and returns a tuple with the train and test partitions.
    Example:
    >>> samples = [ [[3,3], [3,2]], [[3,2], [4,8]], [[6,7], [7,7]] ]
    >>> # Will cut here ----------^ on calling `split_samples(samples)`.
    >>> split_samples(samples)
    ([[[3, 3], [3, 2]]], [[[3, 2], [4, 8]], [[6, 7], [7, 7]]])
    >>> split_samples(samples, 0)
    ([], [[[3, 3], [3, 2]], [[3, 2], [4, 8]], [[6, 7], [7, 7]]])
    >>> split_samples(['a', 'b', 'c'], 0.5)
    (['a'], ['b', 'c'])
    """
    cutoff = int(len(samples) * perc)
    # Simply use pivot to split list.
    samples_ini = samples[:cutoff]
    samples_end = samples[cutoff:]
    return samples_ini, samples_end


def process_dataset(dataset, experiment_config={}):
    """
    Process dataset and create 2 labeled partitions (training, test),
    according to the given experiment configuration.
    """
    experiment_config = merge_dicts(default_config, experiment_config)
    from joblib import delayed, Parallel

    logger.info("Processing dataset ...")

    # running this in parallel
    def create_feature(item):
        sketch, score, ratio = item
        sketch = featurize(sketch)
        if sketch is not None:
            return sketch, score

    samples_list = Parallel(n_jobs=8)(
        delayed(create_feature)(item) for item in dataset
    )
    print("sample list length is", len(samples_list))
    samples = [s[0] for s in samples_list if s is not None]
    labels = [s[1] for s in samples_list if s is not None]

    # samples, labels = [], []
    # for (sketch, score, ratio) in dataset:
    #     sketch = featurize(sketch)
    #     if sketch is not None:
    #         samples.append(sketch)
    #         labels.append(score)

    print("samples length", len(samples))
    print("labels length", len(labels))

    sequence_size = experiment_config["sequence_size"]
    if sequence_size is not None:
        logger.info("Padding sequences to {} items.".format(sequence_size))
        # If `maxlen` is `None`, all sequences are padded to length of the longest sequence seen during training;
        # which means that the model might not work with unseen longer sequences, so we're better off using a fixed length.
        # Also flatten sketch as a single-stroke sequence of 2D points.
        samples = list(map(flatten, samples))
        # For Sketchy we'll use post-padding, since sketches had different finish ratios.
        # TODO compare with numpy.pad, post padding
        samples = pad_sequences(samples, maxlen=sequence_size, padding="post")

    X, y = np.array(samples), np.array(labels)
    print("X shape", X.shape)
    print("y shape", y.shape)
    return X, y


def shuffle_data(X, y):
    """
    Shuffle both samples and labels, preserving their association.
    Assumes numpy arrays.
    """
    assert len(X) == len(y)
    p = np.random.permutation(len(X))
    return X[p], y[p]


def create_partitions(samples, labels, experiment_config={}):
    """
    Create labeled partitions (training, test), according to the given experiment configuration.
    """
    experiment_config = merge_dicts(default_config, experiment_config)

    # Ensure numpy format and shuffle training/testing data.
    samples, labels = np.array(samples), np.array(labels)
    samples, labels = shuffle_data(samples, labels)

    train_ratio = experiment_config["train_ratio"]
    X_train, X_test = split_samples(samples, train_ratio)
    y_train, y_test = split_samples(labels, train_ratio)

    # If no data agumentation technique is provided, nothing will happen.
    X_train, y_train = over_sample(X_train, y_train, experiment_config["over_sampling"])
    X_test, y_test = over_sample(X_test, y_test, experiment_config["over_sampling"])

    logger.info("Will train with %d sequences", len(X_train))
    logger.info("Will test with %d sequences", len(X_test))
    logger.info(
        "Train classes: {} NO vs {} YES".format(
            y_train.tolist().count(0), y_train.tolist().count(1)
        )
    )
    logger.info(
        "Test classes: {} NO vs {} YES".format(
            y_test.tolist().count(0), y_test.tolist().count(1)
        )
    )

    return X_train, y_train, X_test, y_test


def over_sample(X, y, method=None):
    """
    Oversample the minority class in `X` feature vectors given `y` labels.
    """
    if not method:
        return X, y
    elif method == "SMOTE":
        print("oversampling with smote")
        print("before resampling", np.unique(y, return_counts=True))
        (X, y) = SVMSMOTE(k_neighbors=5, random_state=42).fit_resample(X, y)
        print("after resampling", np.unique(y, return_counts=True))
    elif method == "ADASYN":
        (X, y) = ADASYN(random_state=42).fit_resample(X, y)
    else:
        raise Exception("Wrong oversampling method: {}".format(method))
    return X, y


def select_features(X, y, method="LinearSVC"):
    """
    Identify the most relevant features for training.
    """
    logger.info("Will use {} for feature selection".format(method))
    if method == "LinearSVC":
        # SVM with linear kernel and L1 regularization.
        estim = LinearSVC(C=0.01, penalty="l1", dual=False, max_iter=2000).fit(X, y)
        model = SelectFromModel(estim, prefit=True)
        feats = model.get_support()  # Booleans only
        coefs = list(model.estimator.coef_).pop()
    elif method == "RFECV":
        # Recursive feature elimination and cross-validated selection of the best number of features.
        estim = ExtraTreesClassifier(n_estimators=100, n_jobs=8)
        model = RFECV(
            estimator=estim,
            step=1,
            n_jobs=-1,
            cv=StratifiedKFold(10, shuffle=True),
            scoring="roc_auc",
        ).fit(X, y)
        feats = model.support_  # Booleans only
        coefs = model.estimator_.feature_importances_
    else:
        raise Exception("Wrong feature selection method: {}".format(method))

    #    # Manual model selection example:
    #    clf = ExtraTreesClassifier(n_estimators=100)
    #    clf = clf.fit(X_train, y_train)
    #    coefs = clf.feature_importances_
    #    feats = [columns[i] for i, val in enumerate(coefs) if val > 0]
    #    logger.info('{}/{} feats selected: {}'.format(len(feats), len(columns), feats))
    #    model = SelectFromModel(clf, prefit=True)
    #    X_train = model.transform(X_train)
    #    X_test = model.transform(X_test)

    logger.info("Optimum number of features: {}".format(len([f for f in feats if f])))
    # Return only the relevant features, together with their feature importances.
    indices, importances = [], []
    for idx, is_relevant in enumerate(feats):
        if is_relevant:
            indices.append(idx)
            # LinearSVC returns a list with ALL coefficients, so get the relevant ones.
            if len(feats) == len(coefs):
                importances.append(coefs[idx])
    # RFECV already returns the relevent coeffients, so use that list instead.
    if len(feats) != len(coefs):
        importances = coefs

    assert len(indices) == len(importances)
    return indices, importances


def keep_features(partition, feat_indices):
    """
    Remove irrelevant features from train and test partitions.
    """
    if isinstance(partition[0], np.ndarray):
        partition = list(partition)

    # Iterate backwards to avoid re-indexing the array everytime after item removal.
    num_feats = len(partition[0]) - 1
    for idx in range(num_feats, -1, -1):
        if idx not in feat_indices:
            for row_id, row in enumerate(partition):
                partition[row_id] = np.delete(row, idx, 0)
                if isinstance(row, np.ndarray):
                    partition[row_id] = np.delete(row, idx, 0)
                else:
                    del row[idx]
    return partition


def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.

    See https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/sequence.py
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.

    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, str) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        seqlen = len(s)
        if not seqlen:
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        elif truncating == 'sides':
            diff = maxlen - seqlen
            for i in range(0, -diff//2):
                s = np.delete(s, i, axis=0)
            for i in range(diff//2, 0):
                s = np.delete(s, i, axis=0)
            trunc = s
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        seqlen = len(trunc)
        if padding == 'post':
            x[idx, :seqlen] = trunc
        elif padding == 'pre':
            x[idx, -seqlen:] = trunc
        elif padding == 'sides':
            offset = maxlen - seqlen
            for i in range(0, seqlen):
                x[idx, i + offset//2] = trunc[i]
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


# Run basic unit tests by executing: python file.py -v
if __name__ == "__main__":
    import doctest

    doctest.testmod()
