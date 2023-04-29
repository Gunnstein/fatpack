|logo_img|

=======
fatpack
=======
 
.. image:: https://zenodo.org/badge/113768119.svg
   :target: https://zenodo.org/badge/latestdoi/113768119
   
Python package for fatigue analysis of data series. The package
requires `numpy`.


Installation
------------

Either install from the github repository (latest version),

::

   pip install git+https://github.com/gunnstein/fatpack.git


install from the python package index

::

   pip install fatpack


or from the conda-forge:

::

   conda install --channel=conda-forge fatpack


Usage
-----

The package provides functionality for rainflow cycle counting, defining 
endurance curves, mean and compressive stress range correction 
and racetrack filtering. The code example below shows how fatigue damage 
can be calculated:

.. code:: python

    import numpy as np
    import fatpack


    # Assume that `y` is the data series, we generate one here
    y = np.random.normal(0., 30., size=10000)

    # Extract the stress ranges by rainflow counting
    S = fatpack.find_rainflow_ranges(y)

    # Determine the fatigue damage, using a trilinear fatigue curve
    # with detail category Sc, Miner's linear damage summation rule.
    Sc = 90.0
    curve = fatpack.TriLinearEnduranceCurve(Sc)
    fatigue_damage = curve.find_miner_sum(S)

An example is included (`example.py <https://github.com/Gunnstein/fatpack/blob/master/example.py>`_) which extracts rainflow cycles,
generates the rainflow matrix and rainflow stress spectrum, see the
figure presented below. The example is a good place to start to get
into the use of the package. 

|example_img|


Additional examples are found in the `examples folder <https://github.com/Gunnstein/fatpack/tree/master/examples>`_.


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