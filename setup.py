#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Inspired from https://github.com/kennethreitz/setup.py

from pathlib import Path

from setuptools import find_packages, setup

NAME = 'sing'
DESCRIPTION = 'SING: Symbol-to-Instrument Neural Generator'
URL = 'https://github.com/facebookresearch/SING'
EMAIL = 'defossez@fb.com'
AUTHOR = 'Alexandre Defossez'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = "1.0"

HERE = Path(__file__).parent

REQUIRED = [i.strip() for i in open(HERE / "requirements.txt").readlines()]

try:
    with open(HERE / "README.md", encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    install_requires=REQUIRED,
    include_package_data=True,
    license='Creative Common Attribution-NonCommercial 4.0 International',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)