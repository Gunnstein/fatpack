# -*- coding: utf-8 -*-
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import numpy as np
import abc

__all__ = ["LinearEnduranceCurve", "BiLinearEnduranceCurve",
           "TriLinearEnduranceCurve"]


class AbstractEnduranceCurve(object):
    """Abstract endurance curve.

    Concrete subclasses should define methods:
        `get_endurance`
        `get_stress`

    """
    __metaclass__ = abc.ABCMeta

    Ninf = 1e32

    def __init__(self, Sc):
        """Define a linear endurance curve.

        See class docstring for more information.

        Arguments
        ---------
        Sc : float
            Characteristic stress
        """
        self.Sc = Sc

    @abc.abstractmethod
    def get_endurance(self, S):
        pass

    @abc.abstractmethod
    def get_stress(self, N):
        pass

    @property
    def C(self):
        """Characteristic intercept constant."""
        return self.Nc * self.Sc ** self.m

    def find_miner_sum(self, S):
        """Calculate Miner sum of stress ranges S with the endurance curve.

        The method calculates the Miner sum from the stress ranges.

        Arguments
        ---------
        S : 1darray or 2darray
            If `S` is a 1darray (N) it is assumed that each
            stress range corresponds to one full cycle. If `S` is
            given in a 2darray (N, 2), it is assumed that the first
            column is stress range and the second column contains the
            count of that stress range.

        Returns
        -------
        float
            Miner sum

        Raises
        ------
        ValueError
            If `S` is not a 1darray (N, ) or 2darray (N, 2).
        """
        Sr = np.asarray(S)
        shape = Sr.shape
        miner_sum = None
        if len(shape) == 1:
            miner_sum = np.sum(1. / self.get_endurance(Sr))
        elif len(shape) == 2 and shape[1] == 2:
            miner_sum = np.sum(Sr[:, 1] / self.get_endurance(Sr[:, 0]))
        else:
            raise ValueError("S must be 1darray (N) or 2darray (N, 2)")
        return miner_sum


def ensure_array(method):
    def f(self, x):
        if isinstance(x, float):
            xm = np.array([x])
        else:
            xm = np.asarray(x)
        ym = method(self, xm)
        if isinstance(x, float):
            ym = ym[0]
        return ym
    return f


class LinearEnduranceCurve(AbstractEnduranceCurve):
    """Define a linear endurance curve.

            ^                log N - log S
            |
            | *
            |    *
            |       *         m
            |          * - - - - - +
      S     |             *        .
      t  Sc + - - - - - - - -*     . 1
      r     |                .  *  .
      e     |                .     *
      s     |                .        *
      s     |                .           *
            |                .              *
            |                .                 *
            |                .                    *
            |----------------|-------------------------->
                             Nc
                          Endurance

    In fatpack the slope parameter (m) and the `detail category` or
    `characteristic` stress and endurance (Sc, Nc) is used to define
    the linear curve. The default values for the characteristic
    endurance `Nc` (2.0e6) and the slope `m` (5.0) properties can be
    adjusted on the instance and class.
    """
    m = 5.0
    Nc = 2.0e6

    @ensure_array
    def get_endurance(self, S):
        return self.Nc * (self.Sc / S) ** self.m

    @ensure_array
    def get_stress(self, N):
        return self.Sc * (self.Nc / N) ** (1. / self.m)


class BiLinearEnduranceCurve(AbstractEnduranceCurve):
    """Define a bilinear endurance curve.

            ^
            |
            |*
            | *  m1
            |  *- -+
            |   *  .
     S   Sc +- - * . 1
     t      |    .*.
     r      |    . *
     e      |    .  *
     s   Sd +- - - - *
     s      |    .    . *      m2
            |    .    .    * - - - -+
            |    .    .       *     . 1
            |    .    .          *  .
            |    .    .             *
            |    .    .                *
            |    .    .                   *
            |----|----|---------------------------------->
                 Nc   Nd
                             Endurance


    In fatpack the slope parameters (m1, m2), endurance value at the
    knee point of the bilinear curve (Nd) and the `detail category` or
    `characteristic` stress and endurance (Sc, Nc) is used to define
    the bilinear curve. The default values for the characteristic
    endurance `Nc` (2.0e6) and the slope `m` (5.0) properties can be
    adjusted on the instance and class.

    """
    Nc = 2.0e6
    Nd = 5.0e6
    m1 = 3.0
    m2 = 5.0

    def __init__(self, Sc):
        super(BiLinearEnduranceCurve, self).__init__(Sc)
        self.curve1 = LinearEnduranceCurve(Sc)
        self.curve1.m = self.m1
        self.curve1.Nc = self.Nc
        self.curve2 = LinearEnduranceCurve(self.Sd)
        self.curve2.m = self.m2
        self.curve2.Nc = self.Nd

    @property
    def Sd(self):
        return self.Sc * (self.Nc / self.Nd) ** (1. / self.m1)

    @ensure_array
    def get_endurance(self, S):
        Sd = self.Sd
        N = np.ones_like(S) * self.Ninf
        N[S > Sd] = self.curve1.get_endurance(S[S > Sd])
        N[S <= Sd] = self.curve2.get_endurance(S[S <= Sd])
        return N

    @ensure_array
    def get_stress(self, N):
        S = np.zeros_like(N)
        S[N <= self.Nd] = self.curve1.get_stress(N[N <= self.Nd])
        S[N > self.Nd] = self.curve2.get_stress(N[N > self.Nd])
        return S

    @property
    def C1(self):
        """Intercept constant for the first slope."""
        return self.Nd * self.Sd ** self.m1

    @property
    def C2(self):
        """Intercept constant for the second slope."""
        return self.Nd * self.Sd ** self.m2


class TriLinearEnduranceCurve(BiLinearEnduranceCurve):
    """Define a trilinear endurance curve.
            ^
            |
            |*
            | *  m1
            |  *---+
            |   *  |
     S   Sc +    * | 1
     t      |     *|
     r   Sd +      *       m2
     e      |         *--------+
     s      |            *     | 1
     s      |               *  |
            |                  *
         Sl +                     *  *  *  *  *  *  *  *
            |
            |
            |---|----|-------------|-------------------->
               Nc   Nd             Nl
                             Endurance

    The trilinear curve is defined by six parameters. In fatpack the
    slope parameters (m1, m2), endurance values at the knee points of
    the trilinear curve (Nd, Nl) and the `detail category` or
    `characteristic` stress and endurance (Sc, Nc) is used to define
    the trilinear curve. This definition is in accordance with CEN
    Eurocode EC-3-9 and the default values for Nc, Nd, Nl, m1, m2 are
    taken from EC-3-9.

    """
    Nc = 2.0e6
    Nd = 5.0e6
    Nl = 1.0e8
    m1 = 3.0
    m2 = 5.0

    @ensure_array
    def get_endurance(self, S):
        Sd, Sl = self.Sd, self.Sl
        c1, c2 = self.curve1, self.curve2
        N = np.zeros_like(S)
        N[S > Sd] = c1.get_endurance(S[S > Sd])
        N[S<=Sd] = c2.get_endurance(S[S<=Sd])
        N[S<Sl] = self.Ninf
        return N

    @ensure_array
    def get_stress(self, N):
        S = np.zeros_like(N)
        S[N <= self.Nd] = self.curve1.get_stress(N[N<=self.Nd])
        S[N > self.Nd] = self.curve2.get_stress(N[N > self.Nd])
        S[N > self.Nl] = self.curve2.get_stress(self.Nl)
        return S

    @property
    def Sl(self):
        """Variable amplitude fatigue limit."""
        return self.Sd * (self.Nd / self.Nl) ** (1. / self.m2)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(dpi=300)
    S = np.logspace(1., 3, 1000)
    C = TriLinearEnduranceCurve(71.)
    Cbi = BiLinearEnduranceCurve(71.)
    N = C.get_endurance(S)
    Nbi = Cbi.get_endurance(S)
    print(C.get_stress(2e6))
    print(C.get_endurance(71.))
    ax.loglog(N, S)
    ax.loglog(Nbi, S)
    ax.set(xlim=(1e4, 2e8), ylim=(1, 1000), xlabel='Endurance, Sc=71 MPa',
           ylabel="Stress range [Mpa]")
    plt.grid(which='major')
    plt.grid(which='both')
    plt.show(block=True)
