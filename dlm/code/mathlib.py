#!/usr/bin/env python
# coding: utf-8

'''
Math utilities.

External dependencies, to be installed e.g. via pip:
- none

Author: Luis A. Leiva <luis@sciling.com>
Date: 2018
'''

from __future__ import print_function, division

from math import sqrt, fabs
from operator import mul


def mean(l):
    '''
    Return mean value of list 'l'.

    >>> mean([2, 2, 4, 8])
    4.0
    '''
    if not l:
        return 0.0
    return sum(l)/float(len(l))

def sd(l, mean):
    '''
    Return unbiased stdev of list 'l'.

    >>> sd([2, 2, 4, 8], 4)
    2.8284271247461903
    '''
    n, sd = len(l), 0.0
    if n > 1:
        for v in l:
            sd += (v - mean)**2
        sd = sqrt(sd / (n-1))
    return sd

def mean_sd(l):
    '''
    Computes [mean, sd] of list 'l' at once.

    >>> mean_sd([2, 2, 4, 8])
    (4.0, 2.8284271247461903)
    '''
    m = mean(l)
    s = sd(l, m)
    return m, s

def ci_normal(n, mean, sd, z=1.96):
    '''
    Computes confidence interval [lo, hi] using the normal distribution
    for sample of size 'n' with mean 'mean' and standard deviation 'sd'.
    If a 'z' score is not specified, 95% CIs are computed (z=1.96).

    >>> ci_normal(100, 10, 1)
    (9.804, 10.196)
    '''
    se = sd/sqrt(n)
    b = z * se
    return mean - b, mean + b

def sort(l):
    '''
    Sort list l. The original list is not modified.

    >>> sort([8, 5, 1, 2])
    [1, 2, 5, 8]
    '''
    l_sort = l[:]
    l_sort.sort()
    return l_sort

def quantile(l, p):
    '''
    Return 'p' quantile of list 'l'.

    >>> quantile([8, 5, 1, 2], 0.25)
    1.75
    '''
    l_sort = sort(l)
    n = len(l_sort)
    r = 1 + ((n - 1) * p)
    i = int(r)
    f = r - i
    return (1-f)*l_sort[i-1] + f*l_sort[i] if (i < n) else l_sort[i-1]

def median(l):
    '''
    Return median value of list 'l'.

    >>> median([2, 2, 4, 8])
    3.0
    '''
    l_sort = sort(l)
    return quantile(l_sort, 0.5)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
