<div align="center"><img src="https://raw.githubusercontent.com/scikit-hep/cabinetry/master/docs/_static/cabinetry_logo_small.png" alt="cabinetry logo"></div>

[![CI status](https://github.com/scikit-hep/cabinetry/workflows/CI/badge.svg)](https://github.com/scikit-hep/cabinetry/actions?query=workflow%3ACI)
[![Documentation Status](https://readthedocs.org/projects/cabinetry/badge/?version=latest)](https://cabinetry.readthedocs.io/en/latest/?badge=latest)
[![Codecov](https://codecov.io/gh/scikit-hep/cabinetry/branch/master/graph/badge.svg)](https://codecov.io/gh/scikit-hep/cabinetry)
[![PyPI version](https://badge.fury.io/py/cabinetry.svg)](https://badge.fury.io/py/cabinetry)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/cabinetry.svg)](https://github.com/conda-forge/cabinetry-feedstock)
[![Python version](https://img.shields.io/pypi/pyversions/cabinetry.svg)](https://pypi.org/project/cabinetry/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4742752.svg)](https://doi.org/10.5281/zenodo.4742752)
[![Scikit-HEP](https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg)](https://scikit-hep.org/)

`cabinetry` is a Python library for building and steering binned template fits.
It is written with applications in High Energy Physics in mind.
`cabinetry` interfaces many other powerful libraries to make it easy for an analyzer to run their statistical inference pipeline.

Statistical models in [HistFactory](https://cds.cern.ch/record/1456844) format can be built by `cabinetry` from instructions in a declarative configuration.
`cabinetry` makes heavy use of [`pyhf`](https://pyhf.readthedocs.io/) for statistical inference, and provides additional utilities to help study and disseminate fit results.
This includes commonly used visualizations.
Due to its modular approach, analyzers are free to use all of `cabinetry`'s functionality or only some pieces.
`cabinetry` can be used for inference and visualization with any `pyhf`-compatible model, whether it was built with `cabinetry` or not.


## Installation

`cabinetry` can be installed with `pip`:

```bash
python -m pip install cabinetry
```

This will only install the minimum requirements for the core part of `cabinetry`.
The following will install additional optional dependencies needed for [`ROOT`](https://root.cern/) file reading:

```bash
python -m pip install cabinetry[contrib]
```


## Hello world

To run the following example, first generate the input files via the script `utils/create_ntuples.py`.

```python
import cabinetry

config = cabinetry.configuration.load("config_example.yml")

# create template histograms
cabinetry.templates.build(config)

# perform histogram post-processing
cabinetry.templates.postprocess(config)

# build a workspace
ws = cabinetry.workspace.build(config)

# run a fit
model, data = cabinetry.model_utils.model_and_data(ws)
fit_results = cabinetry.fit.fit(model, data)

# visualize the post-fit model prediction and data
prediction_postfit = cabinetry.model_utils.prediction(model, fit_results=fit_results)
cabinetry.visualize.data_mc(prediction_postfit, data, config=config)
```

The above is an abbreviated version of an example included in `example.py`, which shows how to use `cabinetry`.
It requires additional dependencies obtained with `pip install cabinetry[contrib]`.


## Documentation

Find more information in the [documentation](https://cabinetry.readthedocs.io/) and tutorial material in the [cabinetry-tutorials](https://github.com/cabinetry/cabinetry-tutorials) repository.
`cabinetry` is also described in a paper submitted to vCHEP 2021: [10.5281/zenodo.4627037](https://doi.org/10.5281/zenodo.4627037).


## Acknowledgements

[![NSF-1836650](https://img.shields.io/badge/NSF-1836650-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1836650)

This work was supported by the U.S. National Science Foundation (NSF) cooperative agreement [OAC-1836650 (IRIS-HEP)](https://nsf.gov/awardsearch/showAward?AWD_ID=1836650).
