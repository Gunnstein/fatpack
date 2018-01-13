# -*- coding: utf-8 -*-
import numpy as np

__all__ = ["get_basquin_constant", "get_basquin_endurance",
           "get_basquin_stress", "get_cutoff_stress", "get_bilinear_endurance"]


def get_basquin_constant(Sc, Nc, m):
    return Nc * Sc**m


def get_basquin_endurance(S, Sc, Nc, m):
    return get_basquin_constant(Sc, Nc, m) * S**(-m)


def get_basquin_stress(N, Sc, Nc, m):
    return (get_basquin_constant(Sc, Nc, m) / N) ** (1. / m)


def get_cutoff_stress(Sc, Nc=2e6, m1=3., Nd=5e6, m2=5., Nl=1e8):
    """Returns constant and variable cutoff limit stress.

    Arguments
    ---------
    Nc : float
        The endurance where the detail category is defined.
    m1/m2 : float
        The slope in the first/second part of the bilinear curve.
    Nd : float
        The endurance at the first knee in the bilinear curve.
    Nl : float
        The endurance at the cutoff

    Returns
    -------
    float, float
        The constant and variable cutoff limit stress.
    """
    Nl = Nl or 1e32
    Sd = (Nc / Nd)**(1./m1) * Sc
    Sl = (Nd / Nl)**(1./m2) * Sd
    return Sd, Sl


def get_bilinear_endurance(S, Sc, Nc=2e6, m1=3., Nd=5e6, m2=5., Nl=1e8):
    """Returns the endurance for at stress range `S` and detail class `Sc`.

    The function applies the Basquin bilinear curve. The default values for
    the limit endurances corresponds to the Eurocode 3-9 curve for normal
    stresses.

    Arguments
    ---------
    S : ndarray or float
        The stress range to determine the endurance for.
    Sc : float
        The detail category of the endurance curve.
    Nc : float
        The endurance where the detail category is defined.
    m1/m2 : float
        The slope in the first/second part of the bilinear curve.
    Nd : float
        The endurance at the first knee in the bilinear curve.
    Nl : float
        The endurance at the cutoff

    Returns
    -------
    ndarray or float
        Endurance
    """
    Nl = Nl or 1e32

    Sd, Sl = get_cutoff_stress(Sc, Nc=Nc, m1=m1, Nd=Nd, m2=m2, Nl=Nl)
    try:
        N = np.ones_like(S) * 1e32
        for i, Si in enumerate(S):
            if Si > Sd:
                N[i] = (Sc / Si) ** m1 * Nc
            elif Sl <= Si <= Sd:
                N[i] = (Sd / Si) ** m2 * Nd
    except TypeError:
        N = 1e32
        if S > Sd:
            N = (Sc / S) ** m1 * Nc
        elif Sl <= S <= Sd:
            N = (Sd / S) ** m2 * Nd
    return N


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(dpi=300)
    S = np.logspace(1., 3, 1000)
    N = get_bilinear_endurance(S, 71.)
    ax.loglog(N, S)
    ax.set(xlim=(1e4, 2e8), ylim=(1, 1000), xlabel='Endurance, Sc=71 MPa',
           ylabel="Stress range [Mpa]")
    plt.grid(which='major')
    plt.grid(which='both')
    plt.show(block=True)