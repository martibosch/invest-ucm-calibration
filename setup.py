# coding=utf-8

from io import open  # compatible enconding parameter
from os import environ, path

from setuptools import find_packages, setup

__version__ = "0.5.0"

_description = (
    "Automated calibration of the InVEST urban cooling model with"
    " simulated annealing"
)

classifiers = [
    "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
]

here = path.abspath(path.dirname(__file__))


def read_reqs(reqs_filepath):
    with open(reqs_filepath, encoding="utf-8") as f:
        return f.read().split("\n")


# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# get the dependencies and installs
install_reqs = read_reqs(path.join(here, "requirements.txt"))
dev_reqs = read_reqs(path.join(here, "requirements-dev.txt"))
docs_reqs = read_reqs(path.join(here, "docs", "requirements.txt"))

dependency_links = [
    x.strip().replace("git+", "") for x in install_reqs if x.startswith("git+")
]

setup(
    name="invest-ucm-calibration",
    version=__version__,
    description=_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=classifiers,
    url="https://github.com/martibosch/invest-ucm-calibration",
    author="Martí Bosch",
    author_email="marti.bosch@epfl.ch",
    license="GPL-3.0",
    packages=find_packages(exclude=["docs", "tests*"]),
    include_package_data=True,
    install_requires=install_reqs,
    dependency_links=dependency_links,
    entry_points={
        "console_scripts": [
            "invest-ucm-calibration=invest_ucm_calibration.cli.main:main",
        ],
    },
    extras_require={"dev": dev_reqs, "docs": docs_reqs},
)
