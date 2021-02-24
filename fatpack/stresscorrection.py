# -*- coding: utf-8 -*-
"""
Implementation of stress correction methods for stress-life
analysis. The implementation is based on the following resources:

    `N. E. Dowling et. al. Mean Stress Effects in Stress-Life Fatigue and
    the Walker Equation. Fatigue & Fracture of Engineering Materials &
    Structures, 32 (2009) 163-179`

    `DNV GL AS, RP-C203 Fatigue Design of Offshore Steel Structures.
    (2016)`

    `CEN/TC250, Eurocode 3: Design of Steel Structures. Part 1-9:
    Fatigue. EN 1993-1-9. (2005)`
"""
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import abc
import numpy as np


__all__ = [
    "find_walker_equivalent_stress",
    "find_swt_equivalent_stress",
    "find_morrow_equivalent_stress",
    "find_goodman_equivalent_stress",
    "find_reduced_compressive_stress",
    ]


def find_walker_equivalent_stress(S, Sm, gamma):
    """Walker mean stress corrector for fatigue life analysis.

    Walker mean stress corrector introduces a material factor gamma
    and transforms the stress range S at mean stress Sm to a equivalen
    zero mean stress range St.

    Arguments
    ---------
    S : 1darray
        Stress ranges
    Sm : 1darray
        Mean stress of stress ranges
    gamma : float
        Material factor for Walker mean stress correction.

    Returns
    -------
    1darray
        Equivalent zero mean stress range

    Raises
    ------
    ValueError
        SWT and Walker stress correction not defined for Smax < 0,
        i.e. purely compressive stress ranges.

    Example
    -------
    >>> import fatpack
    >>> import numpy as np
    >>> np.random.seed(10)

    Generate a dataseries for the example
    
    >>> y = np.random.normal(size=100000) * 10. + 15.

    Let us create a mean-stress vs stress-range rainflow matrix,
    start by extracting the stress-range and means from the dataseries.

    >>> S, Sm = fatpack.find_rainflow_ranges(y, return_means=True, k=256)

    Remove purely compressive stress ranges

    >>> Smax = Sm + S / 2.
    >>> mask = (Smax > 0.)
    >>> S, Sm = S[mask], Sm[mask]

    Now, we can determine the equivalent zero mean stress range assuming
    gamma=0.4

    >>> St = fatpack.find_walker_equivalent_stress(S, Sm, 0.4)

    and then we can calculate the damage for the transformed equivalent stress 
    with the help of an endurance curve.

    >>> curve = fatpack.LinearEnduranceCurve(90.)
    >>> Dcorr = curve.find_miner_sum(St)

    """
    Smax = Sm + S / 2.
    if np.any(Smax<=0.):
        raise ValueError("SWT and Walker stress correction not defined for Smax <= 0")
    return (2*Sm + S)**(1-gamma) * S**gamma


def find_swt_equivalent_stress(S, Sm):
    """SWT mean stress corrector for fatigue life analysis.

    Smith, Watson and Topper (SWT) mean stress corrector is a special
    case of the Walker mean stress corrector (gamma=0.5). SWT only
    relies on the stress state and is therefore simpler in use than
    many of the other stress corrector methods.

    Arguments
    ---------
    S : 1darray
        Stress ranges
    Sm : 1darray
        Mean stress of stress ranges

    Returns
    -------
    1darray
        Equivalent zero mean stress range

    Raises
    ------
    ValueError
        SWT and Walker stress correction not defined for Smax < 0,
        i.e. purely compressive stress ranges.

    Example
    -------
    >>> import fatpack
    >>> import numpy as np
    >>> np.random.seed(10)

    Generate a dataseries for the example
    
    >>> y = np.random.normal(size=100000) * 10. + 15.

    Let us create a mean-stress vs stress-range rainflow matrix,
    start by extracting the stress-range and means from the dataseries.

    >>> S, Sm = fatpack.find_rainflow_ranges(y, return_means=True, k=256)

    Remove purely compressive stress ranges

    >>> Smax = Sm + S / 2.
    >>> mask = (Smax > 0.)
    >>> S, Sm = S[mask], Sm[mask]

    Now, we can determine the equivalent zero mean stress range

    >>> St = fatpack.find_swt_equivalent_stress(S, Sm, 0.4)

    and then we can calculate the damage for the transformed equivalent stress 
    with the help of an endurance curve.

    >>> curve = fatpack.LinearEnduranceCurve(90.)
    >>> Dcorr = curve.find_miner_sum(St)
    
    """
    return find_walker_equivalent_stress(S, Sm, 0.5)


def find_morrow_equivalent_stress(S, Sm, sf):
    """Morrow mean stress corrector for fatigue life analysis.

    Morrow mean stress corrector uses the true fracture strength of
    the material and transforms the stress range S at mean stress Sm
    to a equivalent zero mean stress range St.

    Arguments
    ---------
    S : 1darray
        Stress ranges
    Sm : 1darray
        Mean stress of stress ranges
    sf : float
        True fracture strength of material, also equivalent to the
        stress amplitude at 1/2 cycle of a stress endurance curve for
        some materials.

    Returns
    -------
    1darray
        Equivalent zero mean stress range

    Example
    -------
    >>> import fatpack
    >>> import numpy as np
    >>> np.random.seed(10)

    Generate a dataseries for the example
    
    >>> y = np.random.normal(size=100000) * 10. + 15.

    Let us create a mean-stress vs stress-range rainflow matrix,
    start by extracting the stress-range and means from the dataseries.

    >>> S, Sm = fatpack.find_rainflow_ranges(y, return_means=True, k=256)

    Now, we can determine the equivalent zero mean stress range assuming
    fracture strength of 800 MPa

    >>> St = fatpack.find_morrow_equivalent_stress(S, Sm, 800.)

    and then we can calculate the damage for the transformed equivalent stress 
    with the help of an endurance curve.

    >>> curve = fatpack.LinearEnduranceCurve(90.)
    >>> Dcorr = curve.find_miner_sum(St)
    
    """
    return S / (1 - Sm / sf)


def find_goodman_equivalent_stress(S, Sm, su):
    """Goodman mean stress corrector for fatigue life analysis.

    Goodman mean stress corrector uses the ultimate strength of the
    material and transforms the stress range S at mean stress Sm to a
    equivalent zero mean stress range St.

    Arguments
    ---------
    S : 1darray
        Stress ranges
    Sm : 1darray
        Mean stress of stress ranges
    su : float
        Ultimate strength of material.

    Returns
    -------
    1darray
        Equivalent zero mean stress range

    Example
    -------
    >>> import fatpack
    >>> import numpy as np
    >>> np.random.seed(10)

    Generate a dataseries for the example
    
    >>> y = np.random.normal(size=100000) * 10. + 15.

    Let us create a mean-stress vs stress-range rainflow matrix,
    start by extracting the stress-range and means from the dataseries.

    >>> S, Sm = fatpack.find_rainflow_ranges(y, return_means=True, k=256)

    Now, we can determine the equivalent zero mean stress range assuming
    ultimate strength of 500 MPa

    >>> St = fatpack.find_goodman_equivalent_stress(S, Sm, 500.)

    and then we can calculate the damage for the transformed equivalent stress 
    with the help of an endurance curve.

    >>> curve = fatpack.LinearEnduranceCurve(90.)
    >>> Dcorr = curve.find_miner_sum(St)
    
    """
    return find_morrow_equivalent_stress(S, Sm, su)


def find_reduced_compressive_stress(S, Sm, alpha):
    """Direct compressive stress correction for fatigue life analysis.

    Compressive stress correction reduces the compressive part of the
    stress range by a factor (alpha).

    Arguments
    ---------
    S : 1darray
        Stress ranges
    Sm : 1darray
        Mean stress of stress ranges
    alpha : float
        Reduction factor for the compressive part of stress range.

    Returns
    -------
    1darray
        Transformed stress range with reduced compressive stress of
        original stress range.

    Example
    -------
    >>> import fatpack
    >>> import numpy as np
    >>> np.random.seed(10)

    Generate a dataseries for the example
    
    >>> y = np.random.normal(size=100000) * 10. + 15.

    Let us create a mean-stress vs stress-range rainflow matrix,
    start by extracting the stress-range and means from the dataseries.

    >>> S, Sm = fatpack.find_rainflow_ranges(y, return_means=True, k=256)

    Now, we can determine the equivalent zero mean stress range assuming
    reduction factor of 60 %

    >>> St = fatpack.find_reduced_compressive_stress(S, Sm, 0.6)

    and then we can calculate the damage for the transformed equivalent stress 
    with the help of an endurance curve.

    >>> curve = fatpack.LinearEnduranceCurve(90.)
    >>> Dcorr = curve.find_miner_sum(St)
    
    """
    St = S.copy()
    Sa = S/2
    mask = Sm<Sa
    St[mask] = (1-alpha)*Sm[mask] + (1+alpha)*Sa[mask]
    return St


if __name__ == '__main__':
    pass
