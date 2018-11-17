# -*- coding: utf-8 -*-
"""
Implementation of 4-point rainflow counting algorithm in numpy. The
implementation and terminology is based off on the following resources:

    `C. Amzallag et. al. Standardization of the rainflow counting method for
    fatigue analysis. International Journal of Fatigue, 16 (1994) 287-293`

    `ISO 12110-2, Metallic materials - Fatigue testing - Variable amplitude
     fatigue testing.`

    `G. Marsh et. al. Review and application of Rainflow residue processing
    techniques for accurate fatigue damage estimation. International Journal
    of Fatigue, 82 (2016) 757-765`

Note that there are two functions for finding the reversals.

    * `find_reversals_strict` passes the example provided in ISO12110-2 by
    ensuring that data points which fall on a load class boundary are rounded
    upwards if the reversal is a peak and downwards if it is a valley.

    * `find_reversals` classifies the data points by rounding the datapoints
    which lies on the boundary to the lower load class. This function is more
    efficient than the strict version, and yields practically identical results
    if the number of load classes is set sufficiently high.
"""
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
from math import fabs

__all__ = ["find_reversals", "find_rainflow_cycles", "find_rainflow_matrix",
           "find_rainflow_ranges", "find_range_count", "concatenate_reversals"]


def get_load_classes(y, k=64):
    ymax, ymin = y.max(), y.min()
    return np.linspace(ymin, ymax, k+1)


def get_load_class_boundaries(y, k=64):
    ymin, ymax = y.min(), y.max()
    dy = (ymax-ymin) / (2.0*k)
    y0 = ymin - dy
    y1 = ymax + dy
    return np.linspace(y0, y1, k+2)


def find_reversals_strict(y, k=64):
    """Return reversals (peaks and valleys) and indices of reversals in `y`.

    The data points in the dataseries `y` are classified into `k` constant
    sized intervals and then peak-valley filtered to yield the successive
    extremas of the dataseries `y`.

    The function is strict in the sense that data points which fall on a load
    class boundary are rounded upwards if the reversal is a peak and downwards
    to the closest load class if it is a valley.

    Arguments
    ---------
    y : ndarray
        Dataseries containing the signal to find the reversals for.
    k : int
        The number of intervals to divide the min-max range of the dataseries
        into.

    Returns
    -------
    reversals : ndarray
        The reversals of the initial data series `y`.
    indices : ndarray
        The indices of the initial data series `y` which corresponds to the
        reversals.
    """
    y = y.copy()  # Make sure we do not change the original sequence
    sgn = np.sign
    Y = get_load_class_boundaries(y, k)
    dY = Y[1] - Y[0]

    # Classifying points into levels
    for yl, yu in zip(Y[:-1], Y[1:]):
        y[(yl < y) & (y < yu)] = (yl+yu)/2.

    # Classifying points on the level boundary
    for n, yi in enumerate(y):
        if not np.any(yi == Y):
            continue
        if n > 0:
            dy = y[n]-y[n-1]
        else:
            dy = y[n+1] - y[n]
        if dy < 0.:
            y[n] = yi - dY / 2.
        else:
            y[n] = yi + dY / 2.

    # Remove successive datapoints in each class
    ix = [0]
    for n, yi in enumerate(y):
        if n == 0:
            continue
        if yi != y[ix[-1]]:
            ix.append(n)

    # Peak-valley filtering
    revix = [0]
    for n in range(len(ix)-1):
        if n == 0:
            continue
        dy1, dy2 = y[ix[n]]-y[ix[n-1]], y[ix[n+1]]-y[ix[n]]
        if (sgn(dy1) != sgn(dy2)):
            revix.append(ix[n])

    # Peak-valley filtering of last point
    dy1, dy2 = y[revix[-1]]-y[revix[-2]], y[ix[-1]] - y[revix[-1]]
    if sgn(dy1) == sgn(dy2):
        revix[-1] = ix[-1]
    else:
        revix.append(ix[-1])

    return y[revix], np.array(revix)


def find_reversals(y, k=64):
    """Return reversals (peaks and valleys) and indices of reversals in `y`.

    The data points in the dataseries `y` are classified into `k` constant
    sized intervals and then peak-valley filtered to yield the successive
    extremas of the dataseries `y`.

    Arguments
    ---------
    y : ndarray
        Dataseries containing the signal to find the reversals for.
    k : int
        The number of intervals to divide the min-max range of the dataseries
        into.

    Returns
    -------
    reversals : ndarray
        The reversals of the initial data series `y`.
    indices : ndarray
        The indices of the initial data series `y` which corresponds to the
        reversals.
    """
    y = y.copy()  # Make sure we do not change the original sequence
    Y = get_load_class_boundaries(y, k)
    dY = Y[1] - Y[0]

    # Classifying points into levels
    i = np.digitize(y, Y)
    y = Y[0] + dY/2 + (i-1) * dY

    # Find successive datapoints in each class
    dy = y[1:]-y[:-1]
    ix = np.argwhere(dy != 0.).ravel()
    ix = np.append(ix, (ix[-1]+1))
    dy1, dy2 = np.diff(y[ix][:-1]), np.diff(y[ix][1:])
    dy = dy1 * dy2
    revix = ix[np.argwhere(dy < 0.).ravel()+1]
    revix = np.insert(revix, (0), ix[0])
    if (y[revix[-1]]-y[revix[-2]])*(y[ix[-1]] - y[revix[-1]]) < 0.:
        revix = np.append(revix, ix[-1])
    return y[revix], np.array(revix)


def concatenate_reversals(reversals1, reversals2):
    """Concatenate two reversal series.

    The two reversal series are concatenated in the order given in args, i.e
    'reversal1' is  placed before 'reversal2'. The concatenation preserves
    the peak-valley condition at the border by deleting none, one or two
    points.

    Arguments
    ---------
    reversals1, reversals2 : ndarray
    The sequence reversal1 is put first, i.e before reversal2

    Returns
    -------
    ndarray
        Sequence of reversals of the concatenated sequences.
    """
    R1, R2 = reversals1, reversals2
    dRstart, dRend, dRjoin = R2[1] - R2[0], R1[-1] - R1[-2], R2[0] - R1[-1]
    t1, t2 = dRend*dRstart, dRend*dRjoin
    if (t1 > 0) and (t2 < 0):
        result = (R1, R2)
    elif (t1 > 0) and (t2 >= 0):
        result = (R1[:-1], R2[1:])
    elif (t1 < 0) and (t2 >= 0):
        result = (R1, R2[1:])
    elif (t1 < 0) and (t2 < 0):
        result = (R1[:-1], R2)
    return np.concatenate(result)


def find_rainflow_cycles(reversals):
    """Return the rainflow cycles and residue from a sequence of reversals.

    Arguments
    ---------
    reversals : ndarray
        An 1D-array of reversals.

    Returns
    -------
    rainflow_cycles : ndarray
        A (Nx2)-array where the first / second column contains the
        starting / destination point of a rainflow cycle.
    residue : ndarray
        The residue of the reversal series after one pass of the rainflow
        algorithm.
    """
    result = []
    residue = []
    for reversal in reversals:
        residue.append(reversal)
        while len(residue) >= 4:
            S0, S1, S2, S3 = residue[-4], residue[-3], residue[-2], residue[-1]
            dS1, dS2, dS3 = fabs(S1-S0), fabs(S2-S1), fabs(S3-S2)
            if (dS2 <= dS1) and (dS2 <= dS3):
                result.append([S1, S2])
                del residue[-3]
                del residue[-2]
            else:
                break
    return np.array(result), np.array(residue)


def find_rainflow_matrix(cycles, rowbins, colbins):
    """Return the rainflowmatrix

    The classification includes the smallest bin edge, i.e if the bins are
    strictly increasing, bin_i < bin_i+1, and the classification of value x is
    `bin_i <= x < bin_i+1` for all bins except the rightmost bin. Cycles lying
    **on** the rightmost bin edge are included in the last bin, .

    Arguments
    ---------
    cycles : ndarray
        (N x 2) array where the first column determines the row index and the
        second column the column index according to `rowbins` and `colbins`,
        respectively.

    rowbins, colbins : ndarray
        The edges of the bins for classifying the cycles into the rainflow
        matrix. These arrays must be monotonic.

        Cycle values outside the range of the bins are ignored.

    Returns
    -------
    ndarray
        Rainflow matrix corresponding to the row and colbins.

    Raises
    ------
    ValueError
        If rowbins or colbins are not monotonic.
    """
    cc = cycles

    mat = np.zeros((rowbins.size-1, colbins.size-1), dtype=np.float)
    (N, M) = mat.shape

    # Find bin index of each of the cycles
    nrows = np.digitize(cc[:, 0], rowbins)-1
    ncols = np.digitize(cc[:, 1], colbins)-1

    # Include values on the rightmost edge in the last bin
    nrows[cc[:, 0] == rowbins[-1]] = N - 1
    ncols[cc[:, 1] == colbins[-1]] = M - 1

    # Build the rainflow matrix

    for nr, nc in zip(nrows, ncols):
        if (nr >= N) or (nr < 0) or (nc >= M) or (nc < 0):
            continue
        mat[nr, nc] += 1.
    return mat


def find_rainflow_ranges(y, k=64):
    """Returns the ranges of the complete series (incl. residue)

    Returns the ranges by first determining the reversals of the dataseries
    `y` classified into `k` loaded classes, then the cycles and residue
    of the complete series are found by concatenating the residue after the
    first pass of the rainflow algorithm and applying the algorithm a second
    time.

    Arguments
    ---------
    y : ndarray
        Dataseries containing the signal to find the reversals for.
    k : int
        The number of intervals to divide the min-max range of the dataseries
        into.

    Returns
    -------
    ranges : ndarray
        The ranges identified by the rainflow algorithm in the dataseries.
    """
    reversals, __ = find_reversals(y, k)
    cycles_firstpass, residue = find_rainflow_cycles(reversals)
    processed_residue = concatenate_reversals(residue, residue)
    cycles_open_sequence, _ = find_rainflow_cycles(processed_residue)
    cycles = np.concatenate((cycles_firstpass, cycles_open_sequence))
    ranges = np.abs(cycles[:, 1] - cycles[:, 0])
    return ranges


def find_rainflow_ranges_strict(y, k=64):
    """Returns the ranges of the complete series (incl. residue)

    Returns the ranges by first determining the reversals of the dataseries
    `y` classified into `k` loaded classes, then the cycles and residue
    of the complete series are found by concatenating the residue after the
    first pass of the rainflow algorithm and applying the algorithm a second
    time.

    Arguments
    ---------
    y : ndarray
        Dataseries containing the signal to find the reversals for.
    k : int
        The number of intervals to divide the min-max range of the dataseries
        into.

    Returns
    -------
    ranges : ndarray
        The ranges identified by the rainflow algorithm in the dataseries.
    """
    reversals, __ = find_reversals_strict(y, k)
    cycles_firstpass, residue = find_rainflow_cycles(reversals)
    processed_residue = concatenate_reversals(residue, residue)
    cycles_open_sequence, _ = find_rainflow_cycles(processed_residue)
    cycles = np.concatenate((cycles_firstpass, cycles_open_sequence))
    ranges = np.abs(cycles[:, 1] - cycles[:, 0])
    return ranges


def find_range_count(ranges, bins=10, weights=None):
    """Return count and the values ranges (midpoint of bin).

    Arguments
    ---------
    ranges : ndarray
        Array with the values to be counted
    bins : Optional[ndarray,int]
        If bins is a sequence, the values are treated as the left edges (and
        the rightmost edge) of the bins.
        if bins is an int, a sequence is created diving the range `min`--`max`
        of y into `bin` number of equally sized bins.
    weights : Optional[ndarray]
        Array with same size as y, can be used to account for half cycles, i.e
        applying a weight of 0.5 to a value in yields a counting value of 0.5

    Returns
    -------
    N, S : ndarray
        The count and the characteristic value for the range.
    """
    N, bns = np.histogram(ranges, bins=bins, weights=weights)
    S = bns[:-1] + np.diff(bns) / 2.
    return N, S
