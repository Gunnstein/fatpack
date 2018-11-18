|logo_img|

=======
fatpack
=======

Python package for fatigue analysis of data series. The package
requires `numpy`.


Installation
------------

Either download the repository to your computer and install, e.g. by **pip**

::

   pip install .


or install directly from the python package index.

::

   pip install fatpack


Usage
-----

The package provides functions for rainflow cycle counting and
defining endurance curves, which can easily be combined with a damage
accumulation rule to determine the fatigue damage in a component. The
code example below shows how fatigue damage can be calculated:

.. code:: python

    import numpy as np
    import fatpack


    # Assume that `x` is the data series, we generate one here
    x = np.random.normal(0., 30., size=10000)

    # Extract the stress ranges by rainflow counting
    S = fatpack.find_rainflow_ranges(x)

    # Determine the fatigue damage, using a trilinear fatigue curve
    # with detail category Sc, Miner's linear damage summation rule.
    Sc = 90.0
    curve = fatpack.TriLinearEnduranceCurve(Sc)
    fatigue_damage = curve.find_miner_sum(S)

An example is included (example.py) which extracts rainflow cycles,
generates the rainflow matrix and rainflow stress spectrum, see the
figure presented below. The example is a good place to start to get
into the use of the package.

|example_img|


Support
-------

Please `open an issue <https://github.com/Gunnstein/fatpack/issues/new>`_
for support.


Contributing
------------

Please contribute using `Github Flow
<https://guides.github.com/introduction/flow/>`_.
Create a branch, add commits, and
`open a pull request <https://github.com/Gunnstein/fatpack/compare/>`_.

.. |logo_img| image:: https://github.com/Gunnstein/fatpack/blob/master/fatpack-logo.png
    :target: https://github.com/gunnstein/fatpack/

.. |example_img| image:: https://github.com/Gunnstein/fatpack/blob/master/example.png
    :target: https://github.com/gunnstein/fatpack/
