# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import fatpack

np.random.seed(10)

# Generate a signal
y = np.random.normal(size=100000) * 25.

# Find reversals (peaks and valleys), extract cycles and residue (open cycle
# sequence), process and extract closed cycles from residue.
reversals, reversals_ix = fatpack.find_reversals(y)
cycles, residue = fatpack.find_rainflow_cycles(reversals)
processed_residue = fatpack.concatenate_reversals(residue, residue)
cycles_residue, _ = fatpack.find_rainflow_cycles(processed_residue)
cycles_total = np.concatenate((cycles, cycles_residue))

# Find the rainflow ranges from the cycles
ranges = np.abs(cycles_total[:, 1] - cycles_total[:, 0])

# alternatively the rainflow ranges can be obtained from the signal directly
# by the wrapper function `find_rainflow_ranges`, i.e
# ranges = fatpack.find_rainflow_ranges(y)

figsize = np.array([140., 140.]) / 25.
fig = plt.figure(dpi=96, figsize=figsize)

# Plotting signal with reversals.
ax_signal = plt.subplot2grid((3, 2), (0, 0))
ax_signal.plot(y)
ax_signal.plot(reversals_ix, y[reversals_ix], 'ro', fillstyle='none',
               label='reversal')
ax_signal.legend()
ax_signal.set(title="Signal", ylabel="y", xlabel="Index", xlim=[400, 500])

# Plotting the cumulative distribution of the cycle count
ax_cumdist = plt.subplot2grid((3, 2), (1, 0))
N, S = fatpack.find_range_count(ranges, 64)
Ncum = N.sum() - np.cumsum(N)
ax_cumdist.semilogx(Ncum, S)
ax_cumdist.set(title="Cumulative distribution, rainflow ranges",
               xlabel="Count, N", ylabel="Range, S")

# Plotting the rainflow matrix of the total cycle count
ax_rfcmat = plt.subplot2grid((3, 2), (0, 1), rowspan=2, aspect='equal')
bins = np.linspace(cycles_total.min(), cycles_total.max(), 64)
rfcmat = fatpack.find_rainflow_matrix(cycles_total, bins, bins)
X, Y = np.meshgrid(bins, bins, indexing='ij')
C = ax_rfcmat.pcolormesh(X, Y, rfcmat, cmap='magma')
fig.colorbar(C)
ax_rfcmat.set(title="Rainflow matrix",
              xlabel="Starting point", ylabel="Destination point")

# Let us also get the EC3 endurance curve for detail category 160 and plot it.
ax = plt.subplot2grid((3, 2), (2, 0), colspan=2,)
curve = fatpack.TriLinearEnduranceCurve(160)

N = np.logspace(6, 9)
S = curve.get_stress(N)

ax.loglog(N, S)
ax.set(xlim=(1e6, 2e8), ylim=(1., 1000),
       title="Endurance curve, detail category 160 Mpa",
       xlabel="Endurance [1]", ylabel="Stress Range [Mpa]")
ax.grid()
ax.grid(which='both')
fig.tight_layout()
plt.show(block=True)
