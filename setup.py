# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

# Parse the version from the module.
with open('fatpack/__init__.py') as f:
    for line in f:
        if line.find("__version__") >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"')
            version = version.strip("'")
            break

with open('README.md', 'r') as fin:
    long_description = fin.read()

setup(name='fatpack',
      version=version,
      url='https://github.com/Gunnstein/FatPACK',
      license='ISC',
      author='Gunnstein T. Frøseth',
      author_email='gunnstein@mailbox.org',
      description='Package for fatigue analysis, FatPACK',
      packages=find_packages(exclude=["test"]),
      install_requires=[
        'numpy']
     )
