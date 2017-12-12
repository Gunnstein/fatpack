# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from fatpack.rainflow import (get_reversals, get_rainflow_cycles,
                              get_rainflow_matrix, duplicate_reversals,
                              get_rainflow_ranges)

np.random.seed(1)

# Generate a signal
y = np.random.normal(size=10000) * 25.

# Find reversals (peaks and valleys), extract cycles and residual (open cycle
# sequence), process and extract closed cycles for residual.
reversals, reversals_ix = get_reversals(y)
cycles, residual = get_rainflow_cycles(reversals)
processed_residual = duplicate_reversals(residual)
cycles_residual, _ = get_rainflow_cycles(processed_residual)
cycles_total = np.concatenate((cycles, cycles_residual))

# Find the rainflow ranges from the cycles
ranges = np.abs(cycles_total[:, 1] - cycles_total[:, 0])

# alternatively the rainflow ranges can be obtained from the signal directly
# by the wrapper function `get_rainflow_ranges`, i.e
# ranges = get_rainflow_ranges(y)


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





fig = plt.figure(dpi=144)

# Plotting signal with reversals.
ax_signal = plt.subplot2grid((2, 2), (0, 0))
ax_signal.plot(y)
ax_signal.plot(reversals_ix, y[reversals_ix], 'ro', fillstyle='none',
               label='reversal')
ax_signal.legend()
ax_signal.set(title="Signal", ylabel="y", xlabel="Index", xlim=[250, 500])

# Plotting the cumulative distribution of the cycle count
ax_cumdist = plt.subplot2grid((2, 2), (1, 0))
N, S = get_range_count(ranges, 64)
Ncum = N.sum() - np.cumsum(N)
ax_cumdist.semilogx(Ncum, S)
ax_cumdist.set(title="Cumulative distribution, rainflow ranges",
               xlabel="Count, N", ylabel="Range, S")

# Plotting the rainflow matrix of the total cycle count
ax_rfcmat = plt.subplot2grid((2, 2), (0, 1), rowspan=2, aspect='equal')
bins = np.linspace(cycles_total.min(), cycles_total.max(), 64)
rfcmat = get_rainflow_matrix(cycles_total, bins, bins)
X, Y = np.meshgrid(bins, bins, indexing='ij')
ax_rfcmat.pcolormesh(X, Y, rfcmat)
ax_rfcmat.set(title="Starting-destination rainflow matrix",
              xlabel="Starting point", ylabel="Destination point")

plt.show(block=True)


