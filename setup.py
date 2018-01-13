# -*- coding: utf-8 -*-
from setuptools import setup

setup(name='fatpack',
      version='0.4',
      url='https://github.com/Gunnstein/FatPACK',
      license='MIT License',
      description='Package for fatigue analysis, FatPACK',
      author='Gunnstein T. Froeseth',
      author_email='gunnstein.t.froseth@ntnu.no',
      packages=['fatpack'],
      install_requires=[
        'numpy>=1.0']
     )