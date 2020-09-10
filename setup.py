# coding=utf-8

from io import open  # compatible enconding parameter
from os import environ, path

from setuptools import find_packages, setup

__version__ = '0.4.0'

_description = 'Automated calibration of the InVEST urban cooling model with'\
    ' simulated annealing'

classifiers = [
    'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
]

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

# See https://github.com/readthedocs/readthedocs.org/issues/5512
on_rtd = environ.get('READTHEDOCS') == 'True'
if on_rtd:
    install_requires = ['click']
else:
    install_requires = [x.strip() for x in all_reqs if 'git+' not in x]

dependency_links = [
    x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')
]

setup(
    name='invest-ucm-calibration', version=__version__,
    description=_description, long_description=long_description,
    long_description_content_type='text/markdown', classifiers=classifiers,
    url='https://github.com/martibosch/invest-ucm-calibration',
    author='Martí Bosch', author_email='marti.bosch@epfl.ch',
    license='GPL-3.0', packages=find_packages(exclude=['docs', 'tests*']),
    include_package_data=True, install_requires=install_requires,
    dependency_links=dependency_links, entry_points='''
    [console_scripts]
    invest-ucm-calibration=invest_ucm_calibration.cli.main:cli
    ''')
