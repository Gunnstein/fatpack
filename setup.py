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

with open('README.rst', 'r') as fin:
    long_description = fin.read()


setup(
    name='fatpack',
    version=version,
    author='Gunnstein T. Frøseth',
    author_email='gunnstein@mailbox.org',
    description='Fatigue analysis in python',
    license='ISC',
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url='https://github.com/gunnstein/fatpack',
    packages=find_packages(exclude=["test"]),
    classifiers=[
        "License :: OSI Approved :: ISC License (ISCL)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        ],
    install_requires=["numpy"])
