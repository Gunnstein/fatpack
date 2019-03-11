# -*- coding: utf-8 -*-
"""
Implementation of the racetrack amplitude filter. The implementation is
based on the following resources:

    `H. O. Fuchs et al. Shortcuts in cumulative damage analysis.
    SAE Automobile engineering meeting paper 730565. (1973)`

    `H. Wu et. al. Validation of the multiaxial racetrack amplitude filter.
    International Journal of Fatigue, 87 (2016) 167–179`

"""
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
from math import fabs
from .rainflow import find_reversals

__all__ = ["racetrack_filter", "find_reversals_racetrack_filtered"]


def racetrack_filter(reversals, h):
    """Racetrack filter for accelerated fatigue testing.

    The racetrack amplitude filter removes low amplitude cycles from the
    reversals without altering the sequence of the remaining cycles. The
    racetrack filter therefore allows to accelerate variable amplitude
    fatigue testing by removing low amplitude cycles which does not
    significantly affect the overall fatigue damage and at the same time
    preserves sequence effects inherent in the original sequence.

    Arguments
    ---------
    reversals : ndarray
        An 1D-array of reversals.
    h : float
        Racetrack width, cycles with range lower than width are filtered out.

    Returns
    -------
    signal : ndarray
        Signal after applying racetrack filter.
    indices : ndarray
        Indices of racetrack filtered signal.
    """
    y = reversals
    yprev = None
    ix = []
    for n, yn in enumerate(y):
        if (n == 0) or (n == y.size-1):
            yprev = yn
            ix.append(n)
            continue
        dy = yn - yprev
        if fabs(dy) > h / 2.:
            yprev = yn - dy/fabs(dy) * h/2.
            ix.append(n)
    ix = np.array(ix, dtype=np.int)
    return y[ix], ix


def find_reversals_racetrack_filtered(y, h, k=64):
    """Return reversals (peaks and valleys) and indices of reversals in `y`.

    The data points in the dataseries `y` are classified into `k` constant
    sized intervals and then peak-valley filtered to yield the successive
    extremas of the dataseries `y`. The reversals are then filtered with the
    racetrack amplitude filter and then peak-valley filtered again to find
    the racetrack filtered reversals.

    Arguments
    ---------
    y : ndarray
        Dataseries containing the signal to find the reversals for.
    h : float
        Racetrack width, cycles with range lower than width are filtered out.
    k : int
        The number of intervals to divide the min-max range of the dataseries
        into.

    Returns
    -------
    reversals : ndarray
        Reversals of the initial data series `y` after racetrack filtering.
    indices : ndarray
        The indices of the initial data series `y` which corresponds to the
        reversals.
    """
    _, ix = find_reversals(y, k=k)
    z, ixz = racetrack_filter(y[ix], h)
    ix = ix[ixz]
    rev, ixr = find_reversals(z, k=k)
    return y[ix[ixr]], ix[ixr]
