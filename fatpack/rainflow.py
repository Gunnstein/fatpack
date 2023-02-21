# -*- coding: utf-8 -*-
"""
Implementation of 4-point rainflow counting algorithm in numpy. The
implementation and terminology is based on the following resources:

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
    y : 1darray
        Dataseries containing the signal to find the reversals for.
    k : int
        The number of intervals to divide the min-max range of the dataseries
        into.

    Returns
    -------
    reversals : 1darray
        The reversals of the initial data series `y`.
    indices : 1darray
        The indices of the initial data series `y` which corresponds to the
        reversals.

    See Also
    --------
    find_reversals: Faster version of this function

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
    y : 1darray
        Dataseries containing the signal to find the reversals for.
    k : int
        The number of intervals to divide the min-max range of the dataseries
        into.

    Returns
    -------
    reversals : 1darray
        The reversals of the initial data series `y`.
    indices : 1darray
        The indices of the initial data series `y` which corresponds to the
        reversals.

    Example
    -------
    >>> import fatpack
    >>> import numpy as np
    >>> np.random.seed(10)

    Generate a dataseries for the example

    >>> y = np.random.normal(size=100000) * 10.

    The reversals (peaks and valleys) and corresponding indices are then obtained by

    >>> reversals, indices = fatpack.find_reversals(y)

    """

    Y = get_load_class_boundaries(y, k)
    dY = Y[1] - Y[0]

    # Classifying points into levels
    i = np.digitize(y, Y)
    z = Y[0] + dY/2. + (i-1) * dY

    # Find successive datapoints in each class
    dz = z[1:]-z[:-1]
    ix = np.argwhere(dz != 0.).ravel()
    ix = np.append(ix, (ix[-1]+1))
    dz1, dz2 = np.diff(z[ix][:-1]), np.diff(z[ix][1:])
    dz = dz1 * dz2
    revix = ix[np.argwhere(dz < 0.).ravel()+1]
    revix = np.insert(revix, (0), ix[0])
    if (z[revix[-1]] - z[revix[-2]])*(z[ix[-1]] - z[revix[-1]]) < 0.:
        revix = np.append(revix, ix[-1])
    return z[revix], np.array(revix)


def concatenate_reversals(reversals1, reversals2):
    """Concatenate two reversal series.

    The two reversal series are concatenated in the order given in args, i.e
    'reversal1' is  placed before 'reversal2'. The concatenation preserves
    the peak-valley condition at the border by deleting none, one or two
    points.

    Arguments
    ---------
    reversals1, reversals2 : 1darray
    The sequence reversal1 is put first, i.e before reversal2

    Returns
    -------
    1darray
        Sequence of reversals of the concatenated sequences.

    Raises
    ------
    ValueError
        If the end/beginning of inputs 'reversals1'/'reversals2' are not
        reversals.

    Example
    -------
    >>> import fatpack
    >>> import numpy as np
    >>> np.random.seed(10)

    Generate two dataseries for the example

    >>> y1 = np.random.normal(size=50000) * 10.
    >>> y2 = np.random.normal(size=50000) * 10.

    Find the reversals in the dataseries

    >>> rev_1, ix1 = fatpack.find_reversals(y1)
    >>> rev_2, ix2 = fatpack.find_reversals(y2)

    The reversals may then be concatenated (joined) into a new
    sequence of reversals by

    >>> rev = fatpack.concatenate_reversals(rev_1, rev_2)

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
    else:
        raise ValueError(
            "Input must be reversals, end/start value of reversals1/reversals2 repeated.")
    return np.concatenate(result)


def find_rainflow_cycles(reversals):
    """Return the rainflow cycles and residue from a sequence of reversals.

    Arguments
    ---------
    reversals : 1darray
        An 1D-array of reversals.

    Returns
    -------
    rainflow_cycles : 2darray
        A (Nx2)-array where the first / second column contains the
        starting / destination point of a rainflow cycle.
    residue : 1darray
        The residue of the reversal series after one pass of the rainflow
        algorithm.

    Example
    -------
    >>> import fatpack
    >>> import numpy as np
    >>> np.random.seed(10)

    Generate a dataseries for the example

    >>> y = np.random.normal(size=100000) * 10.

    First we must find the reversals in the dataseries

    >>> rev_1, ix1 = fatpack.find_reversals(y)

    then the (closed) rainflow cycles and residual reversals are extracted

    >>> cyc_1, res = fatpack.find_rainflow_cycles(rev_1)

    The residual after the first pass of the rainflow algorithm contains open
    rainflow cycles. If the dataseries y represents a repeating process, these cycles
    are closed on the next repetition and should be included in the cycle count by
    concatenating the reversals in the residual with itself and rainflow cycle counting
    is applied once more. The process of extracting the rainflow cycles from the residue
    is shown below

    >>> rev_res = fatpack.concatenate_reversals(res, res)
    >>> cyc_res, _ = fatpack.find_rainflow_cycles(rev_res)

    Finally, all rainflow cycles in the original sequence is joined together in a single
    array

    >>> rainflow_cycles = np.concatenate((cyc_1, cyc_res))

    """

    result = []
    residue = []
    len_residue = 0
    for reversal in reversals:
        residue += [reversal]
        len_residue += 1
        while len_residue >= 4:
            S0, S1, S2, S3 = residue[-4], residue[-3], residue[-2], residue[-1]
            dS1, dS2, dS3 = fabs(S1-S0), fabs(S2-S1), fabs(S3-S2)
            if (dS2 <= dS1) and (dS2 <= dS3):
                result += [[S1, S2]]
                del residue[-3]
                del residue[-2]
                len_residue -= 2
            else:
                break
    return np.array(result), np.array(residue)


def find_rainflow_matrix(data_array, rowbins, colbins, return_bins=False):
    """Return the rainflowmatrix for the data in data_array

    The data in the first and second column of data_array are binned into
    the row and column of the rainflow matrix, respectively.

    The classification includes the smallest bin edge, i.e if the bins are
    strictly increasing, bin_i < bin_i+1, and the classification of value x is
    `bin_i <= x < bin_i+1` for all bins except the rightmost bin. Data lying
    **on** the rightmost bin edge are included in the last bin.

    Arguments
    ---------
    data_array : 2darray
        (N x 2) array where the first column determines the row index and the
        second column the column index according to `rowbins` and `colbins`,
        respectively.

    rowbins, colbins : 1darray or int
        The edges of the bins for classifying the data_array into the rainflow
        matrix.
        - If bins is a sequence, the values are treated as the left edges (and
        the rightmost edge) of the bins. These arrays must increase monotonically
        and data values outside the range of the bins are ignored.
        - If bins is an int, a sequence is created diving the range `min`--`max`
        of y into `bin` number of equally sized bins.

    return_bins : bool, optional
        If true, row and column bins are also returned together with the rainflow
        matrix

    Returns
    -------
    rfcmat : 2darray
        Rainflow matrix corresponding to the row and colbins.

    or if return_bins is True:

    rowbins, colbins, rfcmat : 1darray, 1darray, 2darray
        Rainflow matrix and the corresponding row and column bins.

    Raises
    ------
    ValueError
        If rowbins or colbins are not monotonic.

    Example
    -------
    >>> import fatpack
    >>> import numpy as np
    >>> np.random.seed(10)

    Generate a dataseries for the example

    >>> y = np.random.normal(size=100000) * 10.

    Let us create a mean-stress vs stress-range rainflow matrix,
    start by extracting the stress-range and means from the dataseries.

    >>> S, Sm = fatpack.find_rainflow_ranges(y, return_means=True, k=256)

    Next create the bins to divide the stress ranges and means into

    >>> bins_S = np.arange(0., 76., 1)
    >>> bins_Sm = np.arange(-25., 26., 1.)

    and establish the data array. Note that other datavectors vectors, e.g.
    Smin and Smax, may also be used to create other data arrays and
    resulting rainflow matrices.

    >>> data_array = np.array([Sm, S]).T

    Finally, establish the rainflow matrix from the data array and the
    specified row and column bins.

    >>> rfcmat = fatpack.find_rainflow_matrix(data_array, bins_Sm, bins_S)

    A figure of the rainflow matrix is useful, first find the coordinates of
    each element in the rainflow matrix

    >>> X, Y = np.meshgrid(bins_Sm, bins_S, indexing='ij')

    and plot the rainflow matrix with matplotlib

    >>> import matplotlib.pyplot as plt
    >>> C = plt.pcolormesh(X, Y, rfcmat)
    >>> cbar = plt.colorbar(C)
    >>> title = plt.title("Rainflow matrix")
    >>> xlab = plt.xlabel("Mean stress range (MPa)")
    >>> ylab = plt.ylabel("Stress range (MPa)")
    >>> plt.show(block=True)

    """

    cc = data_array

    if isinstance(rowbins, int):
        rowbins = np.linspace(cc[:, 0].min(), cc[:, 0].max(), rowbins)
    if isinstance(colbins, int):
        colbins = np.linspace(cc[:, 1].min(), cc[:, 1].max(), colbins)
    N = rowbins.size-1
    M = colbins.size-1
    mat = np.zeros((N, M), dtype=float)

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

    if return_bins:
        return rowbins, colbins, mat
    else:
        return mat


def find_rainflow_ranges(y, k=64, return_means=False, return_cycles=False):
    """Returns the ranges of the complete series (incl. residue)

    Returns the ranges by first determining the reversals of the dataseries
    `y` classified into `k` loaded classes, then the cycles and residue
    of the complete series are found by concatenating the residue after the
    first pass of the rainflow algorithm and applying the algorithm a second
    time.

    Arguments
    ---------
    y : 1darray
        Dataseries to extract rainflow ranges from.
    k : int
        The number of intervals to divide the min-max range of the dataseries
        into.
    return_means : bool
        Return mean for each rainflow range.
    return_cycles : bool
        Return cycles for each mean-range - pair.

    Returns
    -------
    ranges : 1darray
        The ranges identified by the rainflow algorithm in the dataseries.
    means : 1darray, optional
        The mean values for each range.
    cycles : 1darray, optional
        The cycles for each mean-range - pair.

    Raises
    ------
    ValueError
        If no rainflow cycles are found in the sequence.

    Example
    -------
    >>> import fatpack
    >>> import numpy as np
    >>> np.random.seed(10)

    Generate a dataseries for the example

    >>> y = np.random.normal(size=100000) * 10.

    Rainflow ranges S and the corresponding mean values Sm found in the dataseries
    are then obtained by

    >>> S, Sm = fatpack.find_rainflow_ranges(y, return_means=True)
    
    Optionally, the cycles N for each mean-range pair are also returned by
    
    >>> S, N = fatpack.find_rainflow_ranges(y, return_cycles=True)
    
    or
    
    >>> S, Sm, N = fatpack.find_rainflow_ranges(y, return_means=True, return_cycles=True)

    """

    reversals, _ = find_reversals(y, k)
    cycles_firstpass, residue = find_rainflow_cycles(reversals)
    processed_residue = concatenate_reversals(residue, residue)
    cycles_open_sequence, _ = find_rainflow_cycles(processed_residue)
    found_cycles_firstpass = len(cycles_firstpass.shape) == 2
    found_cycles_open_sequence = len(cycles_open_sequence.shape) == 2
    if found_cycles_firstpass and found_cycles_open_sequence:
        cycles = np.concatenate((cycles_firstpass, cycles_open_sequence))
    elif found_cycles_firstpass:
        cycles = cycles_firstpass
    elif found_cycles_open_sequence:
        cycles = cycles_open_sequence
    else:
        raise ValueError("Could not find any cycles in sequence")
    ranges = np.abs(cycles[:, 1] - cycles[:, 0])
    out = ranges
    if return_means:
        means = 0.5 * (cycles[:, 0] + cycles[:, 1])
        if return_cycles:
            out = ranges, means, cycles
        else:
            out = ranges, means
    else:
        if return_cycles:
            out = ranges, cycles
        else:
            out = ranges
    return out


def find_rainflow_ranges_strict(y, k=64, return_means=False):
    """Returns the ranges of the complete series (incl. residue)

    Returns the ranges by first determining the reversals of the dataseries
    `y` classified into `k` load classes, then the cycles and residue
    of the complete series are found by concatenating the residue after the
    first pass of the rainflow algorithm and applying the algorithm a second
    time.

    Note that `find_rainflow_ranges_strict` uses
    `find_reversals_strict`, which classify reversals that lie on the
    load class boundary to the upper or lower load classes depending
    on wether the reversal is a peak or valley. It is recommended to
    use `find_rainflow_ranges` because it is faster than
    `find_rainflow_ranges_strict` and finds virtually identical
    results if `k` is sufficiently large.

    Arguments
    ---------
    y : 1darray
        Dataseries to extract rainflow ranges from.
    k : int
        The number of intervals to divide the min-max range of the dataseries
        into.
    return_means : bool
        Return mean for each rainflow range.

    Returns
    -------
    ranges : 1darray
        The ranges identified by the rainflow algorithm in the dataseries.
    means : 1darray, optional
        The mean values for each range.

    Raises
    ------
    ValueError
        If no rainflow cycles are found in the sequence.

    See Also
    --------
    find_rainflow_ranges: Faster version of this function.

    """

    reversals, _ = find_reversals_strict(y, k)
    cycles_firstpass, residue = find_rainflow_cycles(reversals)
    processed_residue = concatenate_reversals(residue, residue)
    cycles_open_sequence, _ = find_rainflow_cycles(processed_residue)
    found_cycles_firstpass = len(cycles_firstpass.shape) == 2
    found_cycles_open_sequence = len(cycles_open_sequence.shape) == 2
    if found_cycles_firstpass and found_cycles_open_sequence:
        cycles = np.concatenate((cycles_firstpass, cycles_open_sequence))
    elif found_cycles_firstpass:
        cycles = cycles_firstpass
    elif found_cycles_open_sequence:
        cycles = cycles_open_sequence
    else:
        raise ValueError("Could not find any cycles in sequence")
    ranges = np.abs(cycles[:, 1] - cycles[:, 0])
    if return_means:
        means = 0.5 * (cycles[:, 0] + cycles[:, 1])
        return ranges, means
    else:
        return ranges


def find_range_count(ranges, bins=10, weights=None):
    """Return count and the values ranges (midpoint of bin).

    Arguments
    ---------
    ranges : 1darray
        Array with the values to be counted
    bins : 1darray, int, optional
        If bins is a sequence, the values are treated as the left edges (and
        the rightmost edge) of the bins.
        If bins is an int, a sequence is created diving the range `min`--`max`
        of y into `bin` number of equally sized bins.
    weights : 1darray, optional
        Array with same size as y, can be used to account for half cycles, i.e
        applying a weight of 0.5 to a value in yields a counting value of 0.5

    Returns
    -------
    N, S : 1darray
        The count and the characteristic value for the ranges.

    Example
    -------
    >>> import fatpack
    >>> import numpy as np
    >>> np.random.seed(10)

    Generate a dataseries for the example

    >>> y = np.random.normal(size=100000) * 10.

    Extract ranges and establish the bins.

    >>> ranges = fatpack.find_rainflow_ranges(y, k=256)
    >>> bins = np.arange(0, 76., 1)

    Find range count and midpoint (average) of corresponding bin

    >>> N, S = fatpack.find_range_count(ranges, bins)

    A figure of the range count is useful, below a bar plot of the
    range count is shown with matplotlib

    >>> import matplotlib.pyplot as plt
    >>> fig1 = plt.figure()
    >>> bars = plt.bar(S, N)
    >>> title = plt.title("Range count")
    >>> xlab = plt.xlabel("Stress range (MPa)")
    >>> ylab = plt.ylabel("Count (1)")

    Note that the cumulative count can also be easily obtained from
    these results

    >>> fig2 = plt.figure()
    >>> Ncum = N.sum() - np.cumsum(N)
    >>> cumulative_plot = plt.loglog(Ncum, S)
    >>> title = plt.title("Cumulative plot")
    >>> xlab = plt.xlabel("Cumulative count")
    >>> ylab = plt.ylabel("Stress range (MPa)")
    >>> plt.show(block=True)

    """

    N, bns = np.histogram(ranges, bins=bins, weights=weights)
    S = bns[:-1] + np.diff(bns) / 2.
    return N, S


if __name__ == "__main__":
    import doctest
    doctest.testmod()
