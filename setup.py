#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-


__author__ = 'Justin S Bayer, bayer.justin@googlemail.com'


import os
import sys

from setuptools import setup, find_packages
from distutils.ccompiler import new_compiler

setup(
    name="PyBrain",
    version="0.3pre",
    description="PyBrain is the swiss army knife for neural networking.",
    license="BSD",
    keywords="Neural Networks Machine Learning",
    url="http://pybrain.org",
    packages=find_packages(exclude=['examples', 'docs']),
    include_package_data=True,
    test_suite='pybrain.tests.runtests.make_test_suite',
)
