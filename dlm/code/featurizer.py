#!/usr/bin/env python
# coding: utf-8

'''
Stroke feature utilities.

External dependencies, to be installed e.g. via pip:
- numpy v1.14.1

Author: Luis A. Leiva <luis@sciling.com>
Date: 2018
'''

from __future__ import print_function, division

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

import math
import numpy as np

def avg(l):
    '''
    Compute average of all values in the given list.
    >>> avg([1,1,4])
    2.0
    '''
    assert len(l) > 0
    return np.mean(l)

def sqdist(a, b):
    '''
    Compute squared distance between points a and b.
    >>> sqdist([0,1], [1,2])
    2
    '''
    a_x, a_y = a
    b_x, b_y = b
    diff_x, diff_y = a_x - b_x, a_y - b_y
    return diff_x*diff_x + diff_y*diff_y

def dist(a, b):
    '''
    Compute distance between points a and b.
    >>> dist([0,1], [1,2])
    1.4142135623730951
    '''
    sqd = sqdist(a,b)
    return math.sqrt(sqd)

def diff(a, b):
    '''
    Compute difference between points a and b.
    >>> diff([0,1], [1,2])
    2
    '''
    diff = np.subtract(b, a)
    return sum(diff)

def angle(a, b):
    '''
    Compute angle between points a and b.
    >>> angle([0,1], [1,2])
    -0.4636476090008061
    '''
    a_x, a_y = a
    b_x, b_y = b
    dot = a_x*b_x + a_y*b_y
    det = a_x*b_y - a_y*b_x
    return math.atan2(det, dot)

def lst_dist(pts):
    '''
    Compute list of distances in sequence pts.
    >>> lst_dist([[0,1],[1,2]])
    [1.4142135623730951]
    '''
    return [dist(pts[i-1], pts[i]) for i in range(1, len(pts))]

def lst_diff(pts):
    '''
    Compute list of differences in sequence pts.
    >>> lst_diff([[0,1],[1,2]])
    [2]
    '''
    return [diff(pts[i-1], pts[i]) for i in range(1, len(pts))]

def lst_angle(pts):
    '''
    Compute list of agngles in sequence pts.
    >>> lst_angle([[0,1],[1,2]])
    [-0.4636476090008061]
    '''
    return [angle(pts[i-1], pts[i]) for i in range(1, len(pts))]

def sum_dist(pts):
    '''
    Compute cumulative distance in sequence pts.
    >>> sum_dist([[0,1],[1,2]])
    1.4142135623730951
    '''
    return sum(lst_dist(pts))

def sum_diff(pts):
    '''
    Compute cumulative difference in sequence pts.
    >>> sum_diff([[0,1],[1,2]])
    2
    '''
    return sum(lst_diff(pts))

def sum_angle(pts):
    '''
    Compute angle difference in sequence pts.
    >>> sum_angle([[0,1],[1,2]])
    -0.4636476090008061
    '''
    return sum(lst_angle(pts))

def sum_sq_angle(pts):
    '''
    Compute sum of squared angles.
    >>> sum_sq_angle([[0,1],[1,2]])
    0.21496910533216437
    '''
    return sum(a**2 for a in lst_angle(pts))

def avg_dist(pts):
    '''
    Compute average distance in sequence pts.
    >>> avg_dist([[0,1],[1,2]])
    0.7071067811865476
    '''
    return sum_dist(pts) / len(pts)

def avg_diff(pts):
    '''
    Compute average difference in sequence pts.
    >>> avg_diff([[0,1],[1,2]])
    1.0
    '''
    return sum_diff(pts) / len(pts)

def avg_angle(pts):
    '''
    Compute mean angle in sequence pts.
    >>> avg_angle([[0,1],[1,2]])
    -0.23182380450040305
    '''
    return sum_angle(pts) / len(pts)

def std_dist(pts):
    '''
    Compute SD distance in sequence pts.
    >>> std_dist([[0,1],[1,2]])
    0.0
    '''
    arr = np.array(lst_dist(pts))
    return np.std(arr) / len(pts)

def std_diff(pts):
    '''
    Compute SD difference in sequence pts.
    >>> std_diff([[0,1],[1,2]])
    0.0
    '''
    arr = np.array(lst_diff(pts))
    return np.std(arr) / len(pts)

def std_angle(pts):
    '''
    Compute SD angle in sequence pts.
    >>> std_angle([[0,1],[1,2]])
    0.0
    '''
    arr = np.array(lst_angle(pts))
    return np.std(arr) / len(pts)

def box_bounds(pts):
    '''
    Compute bounding box bounds.
    >>> box_bounds([[1,2],[3,6],[9,0]])
    (1, 9, 0, 6)
    '''
    arr = np.array(pts)
    x_max, y_max = np.max(arr, axis=0)
    x_min, y_min = np.min(arr, axis=0)
    return x_min, x_max, y_min, y_max

def bounding_box(pts):
    '''
    Compute bounding box of sequence pts.
    >>> bounding_box([[0,1],[1,2]])
    (1, 1)
    >>> bounding_box([[1,2],[3,6],[9,0]])
    (8, 6)
    '''
    x_min, x_max, y_min, y_max = box_bounds(pts)
    return x_max - x_min, y_max - y_min

def aspect_ratio(pts):
    '''
    Compute aspect ratio of sequence pts.
    >>> aspect_ratio([[0,1],[1,2]])
    1.0
    '''
    width, height = bounding_box(pts)
    if height == 0:
        return 0
    return width / height

def box_area(pts):
    '''
    Compute bounding box area of sequence pts.
    >>> box_area([[0,1],[1,2]])
    1
    '''
    width, height = bounding_box(pts)
    return width * height

def hull_area(pts):
    '''
    Compute convex hull area of sequence pts.
    >>> hull_area([[0,1],[1,2]])
    0.0
    >>> hull_area([[0,1],[1,2],[1,1]])
    0.5
    '''
    # Assume a closed polygon.
    if pts[-1] != pts[0]:
        pts.append(pts[0])

    # See https://stackoverflow.com/a/19875478
    lines = np.hstack([pts, np.roll(pts, -1, axis=0)])
    arsum = sum(a_x*b_y - b_x*a_y for a_x,a_y,b_x,b_y in lines)
    return abs(arsum) / 2

def simplify(pts, eps_dist=0.5):
    '''
    Remove redundant points.
    >>> simplify([[0,0],[0,1],[1,0.4],[1,0],[1,1],[2,1]])
    [[0, 0], [0, 1], [1, 0], [1, 1], [2, 1]]
    '''
    res = []
    for i in range(1, len(pts)):
        next = pts[i]
        curr = pts[i - 1]
        if dist(curr, next) < eps_dist:
            continue
        res.append(curr)
    # Add always last point?
    res.append(next)
    return res

def corners(pts, eps_dist=0.5, eps_angle=0.2):
    '''
    Compute the number of corners in sequence pts.
    >>> corners([[0,0],[0,1],[1,1]])
    [[0, 1]]
    '''
    simpler = simplify(pts, eps_dist)
    indices = []
    corners = []
    for i in range(1, len(simpler)):
        next = simpler[i]
        curr = simpler[i - 1]
        theta = angle(next, curr)
        min_angle = np.pi * eps_angle
        if theta > min_angle:
            indices.append(i)
            corners.append(curr)
    return corners

def flatten(list_of_lists):
    '''
    Flatten a list of lists.
    >>> flatten([[[0,0],[0,1],[1,1]],[[1,0],[1,1],[2,1]]])
    [[0, 0], [0, 1], [1, 1], [1, 0], [1, 1], [2, 1]]
    '''
    lst = []
    for l in list_of_lists:
        lst += l
    return lst

def quantize(pts, grid=10):
    '''
    Applies quantization of fixed grid size (in px) in sequence pts.
    >>> quantize([[0,0],[0,1],[1,2]], 1)
    [0.5, 0.5]
    '''
    x_coords = sorted(p[0] for p in pts)
    y_coords = sorted(p[1] for p in pts)

    x_min, x_max, y_min, y_max = box_bounds(pts)
    # Ensure we're dealing with integer values.
    x_min, x_max = int(round(x_min)), int(round(x_max))
    y_min, y_max = int(round(y_min)), int(round(y_max))

    counts = []
    for i in range(x_min, x_max, grid):
        xs = [x for x in x_coords if x >= i and x < i + grid]
        for j in range(y_min, y_max, grid):
            ys = [y for y in y_coords if y >= j and y < j + grid]
            counts.append(len(xs) + len(ys))
    # Normalize counts.
    return [c/sum(counts) for c in counts]

def entropy(probs):
    '''
    Compute the entropy of a list of probabilities.
    >>> entropy([0,0.25,0.75])
    0.8112781244591328
    '''
    return -1 * sum(p * math.log(p, 2) for p in probs if p > 0);

def box_length(pts):
    '''
    Compute bounding box length.
    >>> box_length([[0,0],[0,1],[1,2]])
    2.23606797749979
    '''
    width, height = bounding_box(pts)
    return math.sqrt(width**2 + height**2)

def box_angle(pts):
    '''
    Compute bounding box angle.
    >>> box_angle([[0,0],[0,1],[1,2]])
    1.1071487177940904
    '''
    width, height = bounding_box(pts)
    # TODO: Check if we can use atan2 (which is more robust) for this feature.
    #return math.atan2(height, width)
    if width == 0:
        return 0
    return math.atan(height / width)

def path_length(pts):
    '''
    Compute path length.
    >>> path_length([[0,0],[0,1],[1,2]])
    2.23606797749979
    '''
    p0, pN = pts[0], pts[-1]
    return dist(p0, pN)

def cosine_initial(pts):
    '''
    Compute cosine of initial angle.
    >>> cosine_initial([[0,0],[0,1],[1,2]])
    0.4472135954999579
    '''
    if len(pts) < 3:
        return 0
    x0, x2 = pts[0][0], pts[2][0]
    y0, y2 = pts[0][1], pts[2][1]
    d = math.sqrt((x2 - x0)**2 + (y2 - y0)**2)
    if d > 0:
        return (x2 - x0) / d
    return 0

def sine_initial(pts):
    '''
    Compute sine of initial angle.
    >>> sine_initial([[0,0],[0,1],[1,2]])
    0.8944271909999159
    '''
    if len(pts) < 3:
        return 0
    x0, x2 = pts[0][0], pts[2][0]
    y0, y2 = pts[0][1], pts[2][1]
    d = math.sqrt((x2 - x0)**2 + (y2 - y0)**2)
    if d > 0:
        return (y2 - y0) / d
    return 0

def cosine_final(pts):
    '''
    Compute cosine of final angle.
    >>> cosine_final([[0,0],[0,1],[1,2]])
    0.4472135954999579
    '''
    if len(pts) < 3:
        return 0
    x0, xN = pts[0][0], pts[-1][0]
    d = path_length(pts)
    if d > 0:
        return (xN - x0) / d
    return 0

def sine_final(pts):
    '''
    Compute sine of final angle.
    >>> sine_final([[0,0],[0,1],[1,2]])
    0.8944271909999159
    '''
    if len(pts) < 3:
        return 0
    y0, yN = pts[0][1], pts[-1][1]
    d = path_length(pts)
    if d > 0:
        return (yN - y0) / d
    return 0

def featurize(strokes):
    '''
    Create a featurized representation of a given stroke-based drawing.
    '''
    # Ignore very short strokes and thus empty strokes too.
    strokes = [stroke for stroke in strokes if len(stroke) > 2]
    if not strokes:
        return None

#    # For mockup testing.
#    return {
#        'num_strokes': 1,
#        'num_fit_strokes': 1,
#        'num_points': 1,
#        'num_fit_points': 1,
#        'num_corners': 1,
#        'sum_dist': 1,
#        'sum_diff': 1,
#        'sum_angle': 1,
#        'sum_sq_angle': 1,
#        'box_area': 1,
#        'hull_area': 1,
#        'aspect_ratio': 1,
#        'entropy': 1,
#        'box_length': 1,
#        'box_angle': 1,
#        'path_length': 1,
#        'cosine_initial': 1,
#        'sine_initial': 1,
#        'cosine_final': 1,
#        'sine_final': 1,
#        'avg_dist': 1,
#        'avg_diff': 1,
#        'avg_angle': 1,
#        'std_dist': 1,
#        'std_diff': 1,
#        'std_angle': 1,
#    }

    fit_strokes = list(map(simplify, strokes))
    flat_points = flatten(strokes)
    len_corners = [len(corners(stroke, 2*std_dist(stroke), 2*std_angle(stroke))) for stroke in strokes]

    return {
        'num_strokes': len(strokes),
        'num_fit_strokes': len(fit_strokes),
        'num_points': sum(map(len, strokes)),
        'num_fit_points': sum(map(len, fit_strokes)),
        'num_corners': sum(len_corners),
        'sum_dist': sum(map(sum_dist, strokes)),
        'sum_diff': sum(map(sum_diff, strokes)),
        'sum_angle': sum(map(sum_angle, strokes)),
        'sum_sq_angle': sum(map(sum_sq_angle, strokes)),
        'box_area': box_area(flat_points),
        'hull_area': hull_area(flat_points),
        'aspect_ratio': aspect_ratio(flat_points),
        'entropy': entropy(quantize(flat_points)),
        'box_length': sum(map(box_length, strokes)),
        'box_angle': sum(map(box_angle, strokes)),
        'path_length': sum(map(path_length, strokes)),
        'cosine_initial': sum(map(cosine_initial, strokes)),
        'sine_initial': sum(map(sine_initial, strokes)),
        'cosine_final': sum(map(cosine_final, strokes)),
        'sine_final': sum(map(sine_final, strokes)),
        # Local features are macro-averaged (mean of means).
        'avg_dist': avg(list(map(avg_dist, strokes))),
        'avg_diff': avg(list(map(avg_diff, strokes))),
        'avg_angle': avg(list(map(avg_angle, strokes))),
        'std_dist': avg(list(map(std_dist, strokes))),
        'std_diff': avg(list(map(std_diff, strokes))),
        'std_angle': avg(list(map(std_angle, strokes))),
    }

# Run basic unit tests by executing: python file.py -v
if __name__ == '__main__':
    import doctest
    doctest.testmod()
