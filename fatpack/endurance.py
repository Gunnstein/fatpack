# -*- coding: utf-8 -*-
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import abc
from functools import wraps
import numpy as np


__all__ = ["LinearEnduranceCurve", "BiLinearEnduranceCurve",
           "TriLinearEnduranceCurve", "QuadLinearEnduranceCurve", "BiLinearEnduranceCurveFlat"]


def ensure_array(method):
    @wraps(method)
    def wrapped_method(self, x):
        x_is_float_or_int = isinstance(x, float) or isinstance(x, int)
        if x_is_float_or_int:
            xm = np.array([x])
        else:
            xm = np.asfarray(x)
        ym = method(self, xm)
        if x_is_float_or_int:
            ym = ym[0]
        return ym
    return wrapped_method


class AbstractEnduranceCurve(object):
    """Abstract endurance curve.

    Concrete subclasses should define methods:
        `get_endurance`
        `get_stress`

    """
    __metaclass__ = abc.ABCMeta

    Ninf = np.inf

    def __init__(self, Sc):
        """Define endurance curve.

        See class docstring for more information.

        Arguments
        ---------
        Sc : float
            Characteristic stress
        """
        self.Sc = Sc

    @abc.abstractmethod
    def get_endurance(self, S):
        """Return endurance value(s) for stress range(s) S.

        Arguments
        ---------
        S : float or 1darray
            Stress range(s) to find the corresponding endurance(s) for.

        Returns
        -------
        float or 1darray
            Endurance for the stress range(s) S.
        """

    @abc.abstractmethod
    def get_stress(self, N):
        """Return stress range(s) for the endurance(s) N.

        Arguments
        ---------
        N : float or 1darray
            Endurance(s) to find the corresponding stress for.

        Returns
        -------
        float or 1darray
            Stress range(s) for endurance(s) N.
        """

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

        Example
        -------
        >>> import fatpack
        >>> import numpy as np
        >>> np.random.seed(10)

        First we create an endurance curve with detail category 90.
        >>> curve = fatpack.LinearEnduranceCurve(90.)

        The miner sum is then found from a signal y after extracting 
        rainflow ranges in the following way

        >>> y = np.random.normal(size=100000) * 10.
        >>> S = fatpack.find_rainflow_ranges(y)
        >>> D = curve.find_miner_sum(S)
        >>> print("The damage in signal y is D={0:3.2e}".format(D))
        The damage in signal y is D=6.56e-05

        """

        Sr = np.asfarray(S)
        shape = Sr.shape
        if len(shape) == 1:
            miner_sum = np.sum(1. / self.get_endurance(Sr))
        elif len(shape) == 2 and shape[1] == 2:
            miner_sum = np.sum(Sr[:, 1] / self.get_endurance(Sr[:, 0]))
        else:
            raise ValueError("S must be 1darray (N) or 2darray (N, 2)")
        return miner_sum


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

    Example
    -------
    >>> import fatpack
    >>> import numpy as np
    >>> np.random.seed(10)

    First we create an endurance curve with detail category 90.
    >>> curve = fatpack.LinearEnduranceCurve(90.)

    Let us find the damage according to Miner's linear damage rule from
    a rainflow counted signal y

    >>> y = np.random.normal(size=100000) * 10.
    >>> S = fatpack.find_rainflow_ranges(y)
    >>> D = curve.find_miner_sum(S)
    >>> print("The damage in signal y is D={0:3.2e}".format(D))
    The damage in signal y is D=6.56e-05

    Finally, we can create a figure of the endurance curve with matplotlib

    >>> import matplotlib.pyplot as plt
    >>> N = np.logspace(4, 9, 1000)
    >>> S = curve.get_stress(N)
    >>> line = plt.loglog(N, S)
    >>> grd = plt.grid(which='both')
    >>> title = plt.title("Linear endurance curve")
    >>> xlab = plt.xlabel("Cycles to failure (1)")
    >>> ylab = plt.ylabel("Stress range (MPa)")
    >>> plt.show(block=True)

    """

    m = 5.0
    Nc = 2.0e6

    @property
    def C(self):
        """Characteristic intercept constant."""
        return self.Nc * self.Sc ** self.m

    @ensure_array
    def get_endurance(self, S):
        return self.Nc * (self.Sc / S) ** self.m

    @ensure_array
    def get_stress(self, N):
        return self.Sc * (self.Nc / N) ** (1. / self.m)


class BiLinearEnduranceCurve(AbstractEnduranceCurve):
    """Define a bilinear endurance curve.

            ^                log N - log S
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


    The slope parameters (m1, m2), endurance value at the knee point
    of the bilinear curve (Nd) and the `detail category` or
    `characteristic` stress and endurance (Sc, Nc) is used to define
    the bilinear curve. The default values for the characteristic
    endurance `Nc` (2.0e6) and the slope `m` (5.0) properties can be
    adjusted on the instance and class.

    Example
    -------
    >>> import fatpack
    >>> import numpy as np
    >>> np.random.seed(10)

    First we create an endurance curve with detail category 90.
    >>> curve = fatpack.BiLinearEnduranceCurve(90.)

    Let us find the damage according to Miner's linear damage rule from
    a rainflow counted signal y

    >>> y = np.random.normal(size=100000) * 10.
    >>> S = fatpack.find_rainflow_ranges(y)
    >>> D = curve.find_miner_sum(S)
    >>> print("The damage in signal y is D={0:3.2e}".format(D))
    The damage in signal y is D=1.19e-04

    Finally, we can create a figure of the endurance curve with matplotlib

    >>> import matplotlib.pyplot as plt
    >>> N = np.logspace(4, 9, 1000)
    >>> S = curve.get_stress(N)
    >>> line = plt.loglog(N, S)
    >>> grd = plt.grid(which='both')
    >>> title = plt.title("Bi linear endurance curve")
    >>> xlab = plt.xlabel("Cycles to failure (1)")
    >>> ylab = plt.ylabel("Stress range (MPa)")
    >>> plt.show(block=True)

    """

    Nc = 2.0e6
    Nd = 5.0e6
    m1 = 3.0
    m2 = 5.0

    @property
    def Sd(self):
        return self.Sc * (self.Nc / self.Nd) ** (1. / self.m1)

    @property
    def curve1(self):
        curve = LinearEnduranceCurve(self.Sc)
        curve.Nc = self.Nc
        curve.m = self.m1
        return curve

    @property
    def curve2(self):
        curve = LinearEnduranceCurve(self.Sd)
        curve.Nc = self.Nd
        curve.m = self.m2
        return curve

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
        return self.curve1.C

    @property
    def C2(self):
        """Intercept constant for the second slope."""
        return self.curve2.C


class BiLinearEnduranceCurveFlat(AbstractEnduranceCurve):
    """Define a bilinear endurance curve.

            ^                log N - log S
            |
            |*
            | *  m1
            |  *- -+
            |   *  .
     S   Sc +- - * . 1
     t      |    .*.
     r      |    . *
     e      |    .  *
     s   Sd +- - - - *  *  *  *  *  *  *  *  *
     s      |    .    
            |----|----|---------------------------------->
                 Nc   Nd
                             Endurance


    The slope parameter (m1), endurance value at the knee point
    of the bilinear curve (Nd) and the `detail category` or
    `characteristic` stress and endurance (Sc, Nc) is used to define
    the bilinear curve. The default values for the characteristic
    endurance `Nc` (2.0e6) and the slope `m` (5.0) properties can be
    adjusted on the instance and class.

    Example
    -------
    >>> import fatpack
    >>> import numpy as np
    >>> np.random.seed(10)

    First we create an endurance curve with detail category 90.
    >>> curve = fatpack.BiLinearEnduranceCurve(90.)

    Let us find the damage according to Miner's linear damage rule from
    a rainflow counted signal y

    >>> y = np.random.normal(size=100000) * 10.
    >>> S = fatpack.find_rainflow_ranges(y)
    >>> D = curve.find_miner_sum(S)
    >>> print("The damage in signal y is D={0:3.2e}".format(D))
    The damage in signal y is D=1.19e-04

    Finally, we can create a figure of the endurance curve with matplotlib

    >>> import matplotlib.pyplot as plt
    >>> N = np.logspace(4, 9, 1000)
    >>> S = curve.get_stress(N)
    >>> line = plt.loglog(N, S)
    >>> grd = plt.grid(which='both')
    >>> title = plt.title("Bi linear endurance curve")
    >>> xlab = plt.xlabel("Cycles to failure (1)")
    >>> ylab = plt.ylabel("Stress range (MPa)")
    >>> plt.show(block=True)

    """

    Nc = 2.0e6
    Nd = 5.0e6
    m1 = 3.0

    @property
    def Sd(self):
        return self.Sc * (self.Nc / self.Nd) ** (1. / self.m1)

    @property
    def curve1(self):
        curve = LinearEnduranceCurve(self.Sc)
        curve.Nc = self.Nc
        curve.m = self.m1
        return curve

    @ensure_array
    def get_endurance(self, S):
        Sd = self.Sd
        N = np.ones_like(S) * self.Ninf
        N[S > Sd] = self.curve1.get_endurance(S[S > Sd])
        N[S < self.Sd] = self.Ninf
        return N

    @ensure_array
    def get_stress(self, N):
        S = np.zeros_like(N)
        S[N <= self.Nd] = self.curve1.get_stress(N[N <= self.Nd])
        S[N > self.Nd] = self.curve1.get_stress(self.Nd)
        return S

    @property
    def C1(self):
        """Intercept constant for the first slope."""
        return self.curve1.C


class TriLinearEnduranceCurve(BiLinearEnduranceCurve):
    """Define a trilinear endurance curve.
            ^                log N - log S
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

    The slope parameters (m1, m2), endurance values at the knee points
    of the trilinear curve (Nd, Nl) and the `detail category` or
    `characteristic` stress and endurance (Sc, Nc) is used to define
    the trilinear curve.

    Example
    -------
    >>> import fatpack
    >>> import numpy as np
    >>> np.random.seed(10)

    First we create an endurance curve with detail category 90.
    >>> curve = fatpack.TriLinearEnduranceCurve(90.)

    Let us find the damage according to Miner's linear damage rule from
    a rainflow counted signal y

    >>> y = np.random.normal(size=100000) * 10.
    >>> S = fatpack.find_rainflow_ranges(y)
    >>> D = curve.find_miner_sum(S)
    >>> print("The damage in signal y is D={0:3.2e}".format(D))
    The damage in signal y is D=9.23e-05

    Finally, we can create a figure of the endurance curve with matplotlib

    >>> import matplotlib.pyplot as plt
    >>> N = np.logspace(4, 9, 1000)
    >>> S = curve.get_stress(N)
    >>> line = plt.loglog(N, S)
    >>> grd = plt.grid(which='both')
    >>> title = plt.title("Tri linear endurance curve")
    >>> xlab = plt.xlabel("Cycles to failure (1)")
    >>> ylab = plt.ylabel("Stress range (MPa)")
    >>> plt.show(block=True)

    """

    Nc = 2.0e6
    Nd = 5.0e6
    Nl = 1.0e8
    m1 = 3.0
    m2 = 5.0

    @property
    def Sl(self):
        """Variable amplitude fatigue limit."""
        return self.Sd * (self.Nd / self.Nl) ** (1. / self.m2)

    @ensure_array
    def get_endurance(self, S):
        N = super(TriLinearEnduranceCurve, self).get_endurance(S)
        N[S < self.Sl] = self.Ninf
        return N

    @ensure_array
    def get_stress(self, N):
        S = super(TriLinearEnduranceCurve, self).get_stress(N)
        S[N > self.Nl] = self.curve2.get_stress(self.Nl)
        return S


if __name__ == "__main__":
    import doctest
    doctest.testmod()


class QuadLinearEnduranceCurve(AbstractEnduranceCurve):
    """Define a quadlinear endurance curve.

            ^                log N - log S
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
         Sl +- - - - - - - - - - - - - - -*
                 .    .                 .  *
            |    .    .                 .   *  m3
            |    .    .                 .    *- -+
            |    .    .                 .     *  . 1
     S           .    .                 .      * .
     t      |    .    .                 .       *.
     r      |    .    .                 .        *
     e      |    .    .                 .         *
     s   Sk +- - - - - - - - - - - - - - - - - - - *
     s      |    .    .                 .          . *      m4
            |    .    .                 .          .    * - - - -+
            |    .    .                 .          .       *     . 1
            |    .    .                 .          .          *  .
            |    .    .                 .          .             *
            |    .    .                 .          .                *
            |    .    .                 .          .                   *                                
            |----|----|-----------------|----------|---------------------->
                 Nc   Nd                Nl         Nk
                             Endurance


    The slope parameters (m1, m2, m3, m4), endurance value at the knee points
    of the quadlinear curve (Nd, Nl, Nk) and the `detail category` or
    `characteristic` stress and endurance (Sc, Nc) is used to define
    the quadlinear curve. 

    Example
    -------
    >>> import fatpack
    >>> import numpy as np
    >>> np.random.seed(10)

    First we create an endurance curve with detail category 90.
    >>> curve = fatpack.BiLinearEnduranceCurve(90.)

    Let us find the damage according to Miner's linear damage rule from
    a rainflow counted signal y

    >>> y = np.random.normal(size=100000) * 10.
    >>> S = fatpack.find_rainflow_ranges(y)
    >>> D = curve.find_miner_sum(S)
    >>> print("The damage in signal y is D={0:3.2e}".format(D))
    The damage in signal y is D=1.19e-04

    Finally, we can create a figure of the endurance curve with matplotlib

    >>> import matplotlib.pyplot as plt
    >>> N = np.logspace(4, 9, 1000)
    >>> S = curve.get_stress(N)
    >>> line = plt.loglog(N, S)
    >>> grd = plt.grid(which='both')
    >>> title = plt.title("Bi linear endurance curve")
    >>> xlab = plt.xlabel("Cycles to failure (1)")
    >>> ylab = plt.ylabel("Stress range (MPa)")
    >>> plt.show(block=True)

    """

    Nc = 2e5
    Nd = 1e6
    Nl = 10e6
    Nk = 5*10e6

    m1 = 3.0
    m2 = 5.0
    m3 = 3.0
    m4 = 3.0

    @property
    def Sd(self):
        return self.Sc * (self.Nc / self.Nd) ** (1. / self.m1)

    @property
    def Sl(self):
        return self.Sd * (self.Nd / self.Nl) ** (1. / self.m2)

    @property
    def Sk(self):
        return self.Sl * (self.Nl / self.Nk) ** (1. / self.m3)

    @property
    def curve1(self):
        curve = LinearEnduranceCurve(self.Sc)
        curve.Nc = self.Nc
        curve.m = self.m1
        return curve

    @property
    def curve2(self):
        curve = LinearEnduranceCurve(self.Sd)
        curve.Nc = self.Nd
        curve.m = self.m2
        return curve

    @property
    def curve3(self):
        curve = LinearEnduranceCurve(self.Sl)
        curve.Nc = self.Nl
        curve.m = self.m3
        return curve

    @property
    def curve4(self):
        curve = LinearEnduranceCurve(self.Sk)
        curve.Nc = self.Nk
        curve.m = self.m4
        return curve

    @ensure_array
    def get_endurance(self, S):
        Sd = self.Sd
        Sl = self.Sl
        Sk = self.Sk
        N = np.ones_like(S) * self.Ninf
        for index, s in enumerate(S):
            if s > Sd:
                N[index] = self.curve1.get_endurance(s)
            elif s > Sl:
                N[index] = self.curve2.get_endurance(s)
            elif s > Sk:
                N[index] = self.curve3.get_endurance(s)
            else:
                N[index] = self.curve4.get_endurance(s)
        return N

    @ensure_array
    def get_stress(self, N):
        S = np.zeros_like(N)
        Nd = self.Nd
        Nl = self.Nl
        Nk = self.Nk
        for index, n in enumerate(N):
            if n <= Nd:
                S[index] = self.curve1.get_stress(n)
            elif n <= Nl:
                S[index] = self.curve2.get_stress(n)
            elif n <= Nk:
                S[index] = self.curve3.get_stress(n)
            else:
                S[index] = self.curve4.get_stress(n)
        return S

    @property
    def C1(self):
        """Intercept constant for the first slope."""
        return self.curve1.C

    @property
    def C2(self):
        """Intercept constant for the second slope."""
        return self.curve2.C

    @property
    def C3(self):
        """Intercept constant for the first slope."""
        return self.curve3.C

    @property
    def C4(self):
        """Intercept constant for the second slope."""
        return self.curve4.C
