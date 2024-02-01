import os

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "requirements.txt")) as f:
    install_requires = f.readlines()

setup(
    name="brd",
    version="0.0.1",
    description="Toolbox for numerical modeling of bio reactors",
    url="https://github.com/NREL/BioReactorDesign",
    author="Malik Hassanaly",
    license="BSD 3-Clause",
    package_dir={"brd": "brd"},
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: BSD 3 License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
    ],
    package_data={"": ["*.json", "*.yaml"]},
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=install_requires,
)
