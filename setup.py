# -*- coding: utf-8 -*-
from distutils.core import setup

setup(name='fatpack',
      version='0.1',
      url='https://github.com/Gunnstein/FatPACK',
      license='MIT License',
      description='Package for fatigue analysis, FatPACK',
      author='Gunnstein T. Froeseth',
      author_email='gunnstein.t.froseth@ntnu.no',
      package_dir = {'fatpack': 'fatpack'},
      packages=['fatpack'],
     )