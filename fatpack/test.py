# -*- coding: utf-8 -*-
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
import unittest
from .rainflow import (get_load_classes, get_load_class_boundaries,
                       find_reversals_strict, find_rainflow_ranges_strict,
                       )
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


class TestLinearEnduranceCurve(unittest.TestCase):
    def setUp(self):
        self.m_true = 3.0
        self.Nc_true = 5.0e6
        self.Sc_true = 90
        self.C_true = self.Nc_true*self.Sc_true**self.m_true

        self.crv = LinearEnduranceCurve(self.Sc_true)
        self.crv.m = self.m_true
        self.crv.Nc = self.Nc_true

        self.stress_true = np.array([93., 122., 14., 230., 94., 12., 1.])
        self.endurance_true = (self.Sc_true / self.stress_true)**self.m_true * self.Nc_true


    def test_m(self):
        self.assertEqual(self.m_true, self.crv.m)

    def test_Nc(self):
        self.assertEqual(self.Nc_true, self.crv.Nc)

    def test_Sc(self):
        self.assertEqual(self.Sc_true, self.crv.Sc)

    def test_C(self):
        self.assertEqual(self.C_true, self.crv.C)

    def test_get_endurance(self):
        np.testing.assert_allclose(
            self.endurance_true, self.crv.get_endurance(self.stress_true))

    def test_get_stress(self):
        np.testing.assert_allclose(
            self.stress_true, self.crv.get_stress(self.endurance_true))

    def test_find_miner_sum(self):
        miner_sum = self.crv.find_miner_sum(self.stress_true)
        miner_sum_true = np.sum(1. / self.endurance_true)
        self.assertEqual(miner_sum_true, miner_sum)


class TestBiLinearEnduranceCurve(TestLinearEnduranceCurve):
    def setUp(self):
        self.m1_true = 4.0
        self.Nc_true = 5.0e6
        self.Sc_true = 29.0
        self.C1_true = self.Nc_true*self.Sc_true**self.m1_true

        self.m2_true = 6.0
        self.Nd_true = 1.0e7
        self.Sd_true = (self.Nc_true/self.Nd_true)**(1/self.m1_true)*self.Sc_true
        self.C2_true = self.Nd_true*self.Sd_true**self.m2_true

        self.crv = BiLinearEnduranceCurve(self.Sc_true)
        self.crv.m1 = self.m1_true
        self.crv.Nc = self.Nc_true
        self.crv.m2 = self.m2_true
        self.crv.Nd = self.Nd_true

        self.stress_true = np.array([93., 122., 14., 230., 94., 12., 1.])
        self.endurance_true = (self.Sc_true / self.stress_true)**self.m1_true * self.Nc_true
        S2s = self.stress_true[self.stress_true < self.Sd_true]
        self.endurance_true[self.stress_true < self.Sd_true] = (
            (self.Sd_true / S2s)**self.m2_true * self.Nd_true)

    def test_m(self):
        self.assertEqual(self.m1_true, self.crv.m1)
        self.assertEqual(self.m2_true, self.crv.m2)

    def test_Nd(self):
        self.assertEqual(self.Nd_true, self.crv.Nd)

    def test_Sd(self):
        self.assertEqual(self.Sd_true, self.crv.Sd)

    def test_C(self):
        self.assertEqual(self.C1_true, self.crv.C1)
        self.assertEqual(self.C2_true, self.crv.C2)


class TestTriLinearEnduranceCurve(TestBiLinearEnduranceCurve):
    def setUp(self):
        super(TestTriLinearEnduranceCurve, self).setUp()
        self.crv = TriLinearEnduranceCurve(self.Sc_true)
        self.crv.m1 = self.m1_true
        self.crv.Nc = self.Nc_true
        self.crv.m2 = self.m2_true
        self.crv.Nd = self.Nd_true

        self.Nl_true = 2.0e8
        self.crv.Nl = self.Nl_true
        self.Sl_true = (self.Nd_true / self.Nl_true) ** (1/self.m2_true) * self.Sd_true

        self.stress_true[self.stress_true < self.Sl_true] = self.Sl_true
        self.endurance_true = (self.Sc_true / self.stress_true)**self.m1_true * self.Nc_true
        S2s = self.stress_true[self.stress_true < self.Sd_true]
        self.endurance_true[self.stress_true < self.Sd_true] = (
            (self.Sd_true / S2s)**self.m2_true * self.Nd_true)
        self.endurance_true[self.stress_true < self.Sl_true] = np.inf

    def test_Sl(self):
        self.assertEqual(self.Sl_true, self.crv.Sl)

    def test_Nl(self):
        self.assertEqual(self.Nl_true, self.crv.Nl)


class TestRaceTrackFilter(BaseArrayTestCase, unittest.TestCase):
    def setUp(self):
        self.result_true = np.array(
            [4.2,  7.3,  2., 10.3,  5.2,  8.5,  2.2, 12.,  5.5, 11.1,  1.,
             9.5,  6., 12.,  3.9,  8.3,  1.2,  8.6,  3.9,  6.2])
        y = TESTDATA['dataseries']
        self.result, _ = find_reversals_racetrack_filtered(y, 2, k=64)


if __name__ == "__main__":
    unittest.main()
