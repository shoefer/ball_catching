import os
import re
from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))
try:
    # obtain version string from __init__.py
    init_py = open(os.path.join(here, 'ball_catching', '__init__.py')).read()
    version = re.search('__version__ = "(.*)"', init_py).groups()[0]
except Exception:
    version = ''
try:
    # obtain long description from README and CHANGES
    README = open(os.path.join(here, 'README.md')).read()
except IOError:
    README = ''

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

tests_require = [
#    'mock',
#    'pytest',
#    'pytest-cov',
#    'pytest-pep8',
    ]

setup(name='ball_catching',
    version=version,
    description='Lightweight framework for learning with side information, based on Theano and Lasagne',
    long_description=README,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    keywords="",
    author='Sebastian Hoefer',
    author_email='mail@sebastianhoefer.de',
    url='https://github.com/shoefer/ball_catching',
    license="MIT",
    packages=find_packages("src"),
    include_package_data=False,
    zip_safe=False,
    install_requires=install_requires,
#    extras_require={
#        'testing': tests_require,
#        },
    )
