import os

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "bird", "requirements.txt")) as f:
    install_requires = f.readlines()

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    readme = f.read()

with open(os.path.join(here, "bird", "version.py"), encoding="utf-8") as f:
    version = f.read()
version = version.split("=")[-1].strip().strip('"').strip("'")

setup(
    name="nrel-bird",
    version=version,
    description="Bio Reactor Design (BiRD): a toolbox to simulate and analyze different designs of bioreactors in OpenFOAM",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/NREL/BioReactorDesign",
    author="Malik Hassanaly",
    license="BSD 3-Clause",
    package_dir={"bird": "bird"},
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    package_data={
        "": [
            "*requirements.txt",
        ]
    },
    extras_require={
        "calibration": [
            "joblib",
            "tensorflow",
            "scikit-learn",
            "tf2jax",
        ],
        "optim": [
            "optuna",
            "pandas",
        ],
    },
    project_urls={
        "Documentation": "https://nrel.github.io/BioReactorDesign/",
        "Repository": "https://github.com/NREL/BioReactorDesign",
    },
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=install_requires,
)
