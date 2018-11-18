# -*- coding: utf-8 -*-
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
import unittest
from .rainflow import (get_load_classes, get_load_class_boundaries,
                       find_reversals_strict, find_rainflow_ranges_strict)
from . import *

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
            concatenated_residue = np.array(
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
            ranges_total = np.array([4.,  1.,  8.,  6.,  1.,  4.,
                                     4., 11.,  2.,  3.,  7., 11.]),
            ranges_count = np.array([2, 1, 1, 3, 0, 1, 1, 1, 0, 0, 2, 0]),
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


class BaseArrayTestCase:
    def test_array_equal(self):
        np.testing.assert_array_equal(self.result, self.result_true)

    def test_allclose(self):
        np.testing.assert_allclose(self.result, self.result_true)


class TestFindReversalsStrict(BaseArrayTestCase, unittest.TestCase):
    def setUp(self):
        self.result_true = TESTDATA['reversals']
        y = TESTDATA['dataseries']
        self.result, __ = find_reversals_strict(y, k=11)


class TestConcatenateResidue(BaseArrayTestCase, unittest.TestCase):
    def setUp(self):
        self.result_true = TESTDATA['concatenated_residue']
        residue = TESTDATA['residue']
        self.result = concatenate_reversals(residue, residue)


class TestFindRainflowCycles(unittest.TestCase):
    def setUp(self):
        self.cycles_true = TESTDATA['cycles']
        self.residue_true = TESTDATA['residue']
        self.reversals = TESTDATA['reversals']
        self.cycles, self.residue = find_rainflow_cycles(
                                                            self.reversals)
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


class TestGetLoadClasses(BaseArrayTestCase, unittest.TestCase):
    def setUp(self):
        self.result_true = TESTDATA['classes']
        y = TESTDATA['dataseries']
        self.result = get_load_classes(y, k=11)


class TestGetLoadClassBoundaries(BaseArrayTestCase, unittest.TestCase):
    def setUp(self):
        self.result_true = TESTDATA['class_boundaries']
        y = TESTDATA['dataseries']
        self.result = get_load_class_boundaries(y, k=11)


class TestFindRainflowMatrix(BaseArrayTestCase, unittest.TestCase):
    def setUp(self):
        self.result_true = TESTDATA['starting_destination_rainflow_matrix']
        cycles = TESTDATA['cycles']
        bins = TESTDATA['class_boundaries']
        self.result = find_rainflow_matrix(cycles, bins, bins)


class TestFindRainflowRangesStrict(BaseArrayTestCase, unittest.TestCase):
    def setUp(self):
        self.result_true = TESTDATA['ranges_total']
        self.result = find_rainflow_ranges_strict(
                                                TESTDATA['dataseries'], k=11)


class TestFindRangeCount(unittest.TestCase):
    def setUp(self):
        self.N_true = TESTDATA['ranges_count']
        self.S_true = TESTDATA['classes']
        self.N, self.S = find_range_count(
                            TESTDATA['ranges_total'],
                            bins=TESTDATA['class_boundaries'])

    def test_count_allclose(self):
        np.testing.assert_allclose(self.N, self.N_true)

    def test_class_allclose(self):
        np.testing.assert_allclose(self.S, self.S_true)

    def test_count_array_equal(self):
        np.testing.assert_allclose(self.N, self.N_true)

    def test_class_array_equal(self):
        np.testing.assert_allclose(self.S, self.S_true)


class TestFindReversals(unittest.TestCase):
    def setUp(self):
        y = np.cos(2*np.pi*3.*np.arange(0, 1.01, .01)) * 25.
        self.reversal_est, self.index_est = find_reversals(y)
        self.reversal_true = np.array(
            [25., -25.,  25., -25.,  25., -25.,  25.])
        self.index_true = np.array(
            [0,  17,  34,  50,  67,  84, 100])

    def test_index(self):
        np.testing.assert_allclose(self.index_est, self.index_true)

    def test_reversal(self):
        np.testing.assert_allclose(self.reversal_est, self.reversal_true)


class TestEndurance(unittest.TestCase):
    def setUp(self):
        self.crv = LinearEnduranceCurve(160.)
        self.crv.m = 3.0
        self.crv.Nc = 2e6
        self.tricrv = TriLinearEnduranceCurve(160.)

    def test_intercept_constant(self):
        self.assertEqual(8.192e+12, self.crv.C)

    def test_get_endurance(self):
        self.assertEqual(2e6, self.crv.get_endurance(160.))

    def test_get_stress(self):
        self.assertAlmostEqual(160., self.crv.get_stress(2e6))

    def test_get_bilinear_endurance(self):
        N = self.tricrv.get_endurance([160., (2./5.)**(1./3.)*160., 20.])
        for f, s in zip(N, [2e6, 5e6, 1e32]):
            self.assertEqual(f, s)


if __name__ == "__main__":
    unittest.main()
