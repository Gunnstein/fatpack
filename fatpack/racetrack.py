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

    The racetrack amplitude filter removes low amplitude cycles from
    the reversals without altering the sequence of the remaining
    cycles. The racetrack filter therefore allows to accelerate
    variable amplitude fatigue testing by removing low amplitude
    cycles which does not significantly affect the overall fatigue
    damage and at the same time preserves sequence effects inherent in
    the original sequence.

    Arguments
    ---------
    reversals : 1darray
        An 1D-array of reversals.
    h : float
        Racetrack width, cycles with range lower than width are
        filtered out.

    Returns
    -------
    signal : 1darray
        Signal after applying racetrack filter.
    indices : 1darray
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
    ix = np.array(ix, dtype=int)
    return y[ix], ix


def find_reversals_racetrack_filtered(y, h, k=64):
    """Return racetrack filtered reversals and indices in `y`.

    The data points in the dataseries `y` are classified into `k`
    constant sized intervals and then peak-valley filtered to yield
    the successive extremas of the dataseries `y`. The reversals are
    then filtered with the racetrack amplitude filter and then
    peak-valley filtered again to find the racetrack filtered
    reversals.

    Arguments
    ---------
    y : 1darray
        Dataseries containing the signal to find the reversals for.
    h : float
        Racetrack width, cycles with range lower than width are
        filtered out.
    k : int
        The number of intervals to divide the min-max range of the
        dataseries into.

    Returns
    -------
    reversals : 1darray
        Reversals of the initial data series `y` after racetrack
        filtering.
    indices : 1darray
        The indices of the initial data series `y` which corresponds
        to the reversals.

    Example
    -------
    >>> import fatpack
    >>> import numpy as np
    >>> np.random.seed(10)

    Generate a dataseries for the example

    >>> y = np.random.normal(size=100000) * 10.

    Extract reversals with all rainflow ranges lower than 15 removed

    >>> rev_rtf, ix_rtf = fatpack.find_reversals_racetrack_filtered(y, h=15.)

    Below a figure is created which shows reversals of the dataseris with and without the racetrack
    filter

    >>> import matplotlib.pyplot as plt
    >>> rev, ix = fatpack.find_reversals(y)
    >>> l1 = plt.plot(ix, rev, label='reversals')
    >>> l2 = plt.plot(ix_rtf, rev_rtf, label='racetrack filtered reversals')
    >>> xlim = plt.xlim(0, 100)
    >>> leg = plt.legend(loc='best')
    >>> xlab = plt.xlabel("Indices")
    >>> ylab = plt.ylabel("Signal")
    >>> plt.show(block=True)

    """

    _, ix = find_reversals(y, k=k)
    z, ixz = racetrack_filter(y[ix], h)
    ix = ix[ixz]
    rev, ixr = find_reversals(z, k=k)
    return y[ix[ixr]], ix[ixr]


if __name__ == "__main__":
    import doctest
    doctest.testmod()
