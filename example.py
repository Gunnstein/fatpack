# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from fatpack.rainflow import (get_reversals, get_rainflow_cycles,
                              get_rainflow_matrix, duplicate_reversals,
                              get_rainflow_ranges, get_range_count)

np.random.seed(10)

# Generate a signal
y = np.random.normal(size=100000) * 25.

# Find reversals (peaks and valleys), extract cycles and residual (open cycle
# sequence), process and extract closed cycles from residual.
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



figsize = np.array([140., 70.]) / 25.
fig = plt.figure(dpi=300, figsize=figsize)

# Plotting signal with reversals.
ax_signal = plt.subplot2grid((2, 2), (0, 0))
ax_signal.plot(y)
ax_signal.plot(reversals_ix, y[reversals_ix], 'ro', fillstyle='none',
               label='reversal')
ax_signal.legend()
ax_signal.set(title="Signal", ylabel="y", xlabel="Index", xlim=[400, 500])

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
C = ax_rfcmat.pcolormesh(X, Y, rfcmat, cmap='magma')
fig.colorbar(C)
ax_rfcmat.set(title="Rainflow matrix",
              xlabel="Starting point", ylabel="Destination point")
fig.tight_layout()
fig.savefig('example.png', figsize=figsize, dpi=300)
plt.show(block=True)
