# -*- coding: utf-8 -*-
"""
Implementation a 4-point rainflow counting algorithm in numpy,
roughly follows the terminology and implementation presented in

    `ISO 12110-2, Metallic materials - Fatigue testing - Variable amplitude
     fatigue testing`
"""
import numpy as np


def get_load_classes(y, k=64):
    ymax, ymin = y.max(), y.min()
    dY = (ymax-ymin) / float(k)
    return np.linspace(ymin, ymax, k+1)


def get_load_class_boundaries(y, k=64):
    Y = get_load_classes(y, k)
    dY = Y[1] - Y[0]
    return np.linspace(Y.min()-dY/2., Y.max()+dY/2., k+2)


def get_reversals(y, k=64):
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


def duplicate_reversals(reversals):
    """Duplicate, join and return the reversal series.

    The residual after one pass with the 4-point rainflow algorithm should be
    processsed for the remaining cycles, the approach is to duplicate and join
    the residual and then run the rainflow algorithm on the duplicated
    serices once more. See ISO 12110-2:2013, A.3.3.2 for further information

    Arguments
    ---------
    reversals : ndarray
        The sequence of reversals.

    Returns
    -------
    ndarray
        Series of reversals of the duplicated and joined input sequence.
    """
    R = reversals.copy()
    dRstart, dRend, dRjoin = R[1] - R[0], R[-1] - R[-2], R[0] - R[-1]
    t1, t2 = dRend*dRstart, dRend*dRjoin
    if (t1 > 0) and (t2 < 0):
        result = (R, R)
    elif (t1 > 0) and (t2 >= 0):
        result = (R[:-1], R[1:])
    elif (t1 < 0) and (t2 >= 0):
        result = (R, R[1:])
    elif (t1 < 0) and (t2 < 0):
        result = (R[:-1], R)
    return np.concatenate(result)


def get_rainflow_cycles(reversals):
    """Return the rainflow cycles and residual from a sequence of reversals.

    Arguments
    ---------
    reversals : ndarray
        An 1D-array of reversals.

    Returns
    -------
    rainflow_cycles : ndarray
        A (Nx2)-array where the first / second column contains the
        starting / destination point of a rainflow cycle.
    """
    input_array = reversals.copy()
    output_array = np.zeros((len(input_array), 2), np.double)
    ix_output_array = 0

    residual = []
    for n, reversal in enumerate(input_array):
        residual.append(reversal)
        while len(residual) >= 4:
            S0, S1, S2, S3 = residual[-4], residual[-3], residual[-2], residual[-1]
            dS1, dS2, dS3 = np.abs(S1-S0), np.abs(S2-S1), np.abs(S3-S2)

            if (dS2 <= dS1) and (dS2 <= dS3):
                output_array[ix_output_array] = [S1, S2]
                ix_output_array += 1
                residual.pop(-3)
                residual.pop(-2)
            else:
                break

    output_array = output_array[:ix_output_array]
    return output_array, np.array(residual)


def get_rainflow_matrix(cycles, rowbins, colbins):
    """Return the rainflowmatrix

    Arguments
    ---------
    cycles : ndarray
        (N x 2) array where the first column determines the row index and the
        second column the column index according to `rowbins` and `colbins`,
        respectively.

    rowbins, colbins : ndarray
        The edges of the bins for classifying the cycles into the rainflow
        matrix. These arrays must be monotonic. The classification includes
        the smallest bin edge.

    Returns
    -------
    ndarray
        Rainflow matrix corresponding to the row and colbins.

    Raises
    ------
    ValueError
        If rowbins or colbins are not monotonic.
    """
    mat = np.zeros((rowbins.size, colbins.size), dtype=np.float)
    nrows = np.digitize(cycles[:, 0], rowbins)-1
    ncols = np.digitize(cycles[:, 1], colbins)-1
    for nr, nc in zip(nrows, ncols):
        mat[nr, nc] += 1.
    return mat


def get_rainflow_ranges(y, k=64):
    """Returns the ranges of the complete series (incl. open cycle sequence)

    Returns the ranges by first determining the reversals of the dataseries
    `y` classified into `k` loaded classes, then the cycles and residual
    are found

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
    reversals, __ = get_reversals(y, k)
    cycles_firstpass, residual = get_rainflow_cycles(reversals)
    processed_residual = duplicate_reversals(residual)
    cycles_open_sequence, _ = get_rainflow_cycles(processed_residual)
    cycles = np.concatenate((cycles_firstpass, cycles_open_sequence))
    ranges = np.abs(cycles[:, 1] - cycles[:, 0])
    return ranges


if __name__ == "__main__":
    import unittest
    from tests import testsuite

    runner = unittest.TextTestRunner()
    runner.run(testsuite())


def get_range_count(y, bins=10, weights=None):
    """Return count and the values ranges (midpoint of bin).

    Arguments
    ---------
    y : ndarray
        Array with the values where the
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
    N, bns = np.histogram(y, bins=bins, weights=weights)
    dbns = np.diff(bns)
    S = bns[:-1] - dbns / 2.
    return N, S
