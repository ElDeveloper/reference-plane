#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright (c) 2016--, reference plane development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE.md, distributed with this software.
# ----------------------------------------------------------------------------

from distutils.core import setup
from glob import glob

__version__ = "0.0.1-dev"
__maintainer__ = "Reference Plane Developers"
__email__ = "yoshiki@ucsd.edu"

# based on the text found in github.com/qiime/pynast
classes = """
    Development Status :: 3 - Alpha
    License :: OSI Approved :: BSD 3-Clause License
    Topic :: Software Development :: Libraries :: Application Frameworks
    Programming Language :: Python
    Programming Language :: Python :: Implementation :: CPython
    Operating System :: OS Independent
"""

classifiers = [s.strip() for s in classes.split('\n') if s]

long_description = """Quantify variability in an ordinated space."""

base = {"numpy >= 1.7", "scikit-bio >= 0.4.2"}
test = {"nose >= 0.10.1", "pep8", "flake8"}
all_deps = base | test

setup(
    name='reference-plane',
    version=__version__,
    description='Reference Plane',
    author="Antonio Gonzalez Pena, Rob Knight & Yoshiki Vazquez Baeza",
    author_email=__email__,
    maintainer=__maintainer__,
    maintainer_email=__email__,
    url='http://github.com/ElDeveloper/reference-plane',
    packages=['plane'],
    data_files={},
    install_requires=base,
    extras_require={'test': test, 'all': all_deps},
    long_description=long_description,
    classifiers=classifiers)
