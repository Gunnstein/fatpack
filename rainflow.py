# -*- coding: utf-8 -*-
import numpy as np
import unittest

TESTDATA = dict(
            dataseries  = np.array([
                                4.2,  7.3,  2.0,  9.8,  9.6, 10.3,  5.2,  8.5,
                                3.0,  4.4,  2.2,  2.4,  2.2, 12.0,  5.5, 11.1,
                                1.0,  4.3,  3.5,  9.5,  6.0, 12.0,  3.9,  8.3,
                                1.2,  8.6, 3.9,  6.2
                                ]),
            reversals   = np.array([
                                4.,  7.,  2., 10.,  5.,  9.,  3.,  4.,
                                2., 12.,  5., 11.,  1.,  4.,  3., 10.,
                                6., 12.,  4.,  8.,  1.,  9.,  4.,  6.
                                ]),
            cycles      = np.array([
                                [ 5.,  9.], [ 3.,  4.], [10.,  2.], [ 5., 11.],
                                [ 4.,  3.], [10.,  6.], [ 4.,  8.], [ 1., 12.],
                                ]),
            residue     = np.array([4., 7., 2., 12., 1., 9., 4., 6.]),
            duplicated_residue = np.array(
                                [4., 7., 2., 12., 1., 9., 4., 6.,
                                 4., 7., 2., 12., 1., 9., 4., 6.]),
            cycles_residue = np.array([
                                [ 4.,  6.], [ 4.,  7.], [ 9.,  2.], [ 1., 12],
                                ]),
            cycles_total = np.array([
                                [ 5.,  9.], [ 3.,  4.], [10.,  2.], [ 5., 11.],
                                [ 4.,  3.], [10.,  6.], [ 4.,  8.], [ 1., 12.],
                                [ 4.,  6.], [ 4.,  7.], [ 9.,  2.], [ 1., 12.],
                                ]),
            classes = np.array([1., 2., 3., 4., 5., 6.,
                                7., 8., 9., 10., 11., 12.]),
            class_boundaries = np.array([
                                0.5,  1.5,  2.5,  3.5, 4.5, 5.5, 6.5, 7.5, 8.5,
                                9.5, 10.5, 11.5, 12.5]),
            starting_destination_rainflow_matrix = np.array(
                            [[ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                             [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [ 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [ 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                             [ 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0.],
                             [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [ 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                             [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]
                                                            )
   )


class BaseArrayTestCase(unittest.TestCase):
    def test_array_equal(self):
        np.testing.assert_array_equal(self.result, self.result_true)

    def test_allclose(self):
        np.testing.assert_allclose(self.result, self.result_true)


class FindReversalsTests(BaseArrayTestCase):
    def setUp(self):
        self.result_true = TESTDATA['reversals']
        y = TESTDATA['dataseries']
        self.result, __ = find_reversals(y, k=11)


class DuplicateOpenCycleSequenceTests(BaseArrayTestCase):
    def setUp(self):
        self.result_true = TESTDATA['duplicated_residue']
        self.result = duplicate_open_cycle_sequence(TESTDATA['residue'])


class FindRainflowCyclesTests(unittest.TestCase):
    def setUp(self):
        self.cycles_true = TESTDATA['cycles']
        self.residue_true = TESTDATA['residue']
        self.reversals = TESTDATA['reversals']
        self.cycles, self.residue = find_rainflow_cycles(self.reversals)
        self.result = self.cycles
        self.result_true = self.cycles_true

    def test_cycles_allclose(self):
        np.testing.assert_allclose(self.cycles, self.cycles_true)

    def test_cycles_array_equal(self):
        np.testing.assert_array_equal(self.cycles, self.cycles_true)

    def test_residue_allclose(self):
        np.testing.assert_allclose(self.residue, self.residue_true)

    def test_residue_array_equal(self):
        np.testing.assert_array_equal(self.residue, self.residue_true)


class GetLoadClassesTests(BaseArrayTestCase):
    def setUp(self):
        self.result_true = TESTDATA['classes']
        y = TESTDATA['dataseries']
        self.result = get_load_classes(y, k=11)


class GetLoadClassBoundariesTests(BaseArrayTestCase):
    def setUp(self):
        self.result_true = TESTDATA['class_boundaries']
        y = TESTDATA['dataseries']
        self.result = get_load_class_boundaries(y, k=11)


class GetRainflowMatrixTests(BaseArrayTestCase):
    def setUp(self):
        self.result_true = TESTDATA['starting_destination_rainflow_matrix']
        cycles = TESTDATA['cycles']
        bins = TESTDATA['class_boundaries']
        self.result = get_rainflow_matrix(cycles, bins, bins)


def get_load_classes(y, k=64):
    ymax, ymin = y.max(), y.min()
    dY = (ymax-ymin) / float(k)
    return np.linspace(ymin, ymax, k+1)


def get_load_class_boundaries(y, k=64):
    Y = get_load_classes(y, k)
    dY = Y[1] - Y[0]
    return np.linspace(Y.min()-dY/2., Y.max()+dY/2., k+2)


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
    sgn = lambda x: x / np.abs(x)
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


def duplicate_open_cycle_sequence(reversals):
    """Duplicate, join and return the reversal series.

    See ISO 12110-2:2013, A.3.3.2 for further information

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
    """
    input_array = reversals.copy()
    output_array = np.zeros((len(input_array), 2), np.double)
    ix_output_array = 0

    S = []
    for n, reversal in enumerate(input_array):
        S.append(reversal)
        while len(S) >= 4:
            S0, S1, S2, S3 = S[-4], S[-3], S[-2], S[-1]
            dS1, dS2, dS3 = np.abs(S1-S0), np.abs(S2-S1), np.abs(S3-S2)

            if (dS2 <= dS1) and (dS2 <= dS3):
                output_array[ix_output_array] = [S1, S2]
                ix_output_array += 1
                S.pop(-3)
                S.pop(-2)
            else:
                break

    residue = np.array(S)
    output_array = output_array[:ix_output_array]
    return output_array, residue


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
    mat = np.zeros((rowbins.size-1, colbins.size-1), dtype=np.float)
    nrows = np.digitize(cycles[:, 0], rowbins)-1
    ncols = np.digitize(cycles[:, 1], colbins)-1
    for nr, nc in zip(nrows, ncols):
        mat[nr, nc] += 1.
    return mat


def extract_ranges(y, k=64):
    """Returns the ranges of the complete series (incl. open cycle sequence)

    Returns the ranges which

    """
    reversals, __ = find_reversals(y, k)
    cycles_firstpass, residue = find_rainflow_cycles(reversals)
    processed_residue = duplicate_open_cycle_sequence(residue)
    cycles_open_sequence, _ = find_rainflow_cycles(processed_residue)
    cycles = np.concatenate((cycles_firstpass, cycles_open_sequence))
    means = (cycles[:, 1]+cycles[:, 0]) / 2.
    ranges = np.abs(cycles[:, 1] - cycles[:, 0])
    return ranges, means

def testsuite():
    suite = unittest.TestSuite()
    for Tests in [FindReversalsTests, DuplicateOpenCycleSequenceTests,
                  GetLoadClassesTests, GetLoadClassBoundariesTests,
                  GetRainflowMatrixTests]:
        for f in ['test_array_equal', 'test_allclose']:
            suite.addTest(Tests(f))
    suite.addTest(FindRainflowCyclesTests('test_cycles_allclose'))
    suite.addTest(FindRainflowCyclesTests('test_cycles_array_equal'))
    suite.addTest(FindRainflowCyclesTests('test_residue_allclose'))
    suite.addTest(FindRainflowCyclesTests('test_residue_array_equal'))
    return suite

if __name__ == "__main__":
    from time import time
    runner=unittest.TextTestRunner()
    runner.run(testsuite())

    y = np.random.normal(size=int(10**6)) * 25
    ts = []
    t = lambda: ts.append(time())
    t0 = time()
    reversals, _ = find_reversals(y)
    t()
    cycles_firstpass, residue = find_rainflow_cycles(reversals)
    t()
    processed_residue = duplicate_open_cycle_sequence(residue)
    t()
    cycles_open_sequence, _ = find_rainflow_cycles(processed_residue)
    t()
    cycles = np.concatenate((cycles_firstpass, cycles_open_sequence))
    t()
    t0i = t0
    for ti in ts:
        print "{0:.5f}".format(ti-t0i)
        t0i = ti
