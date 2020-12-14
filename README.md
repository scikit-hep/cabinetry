# cabinetry

[![CI status](https://github.com/alexander-held/cabinetry/workflows/CI/badge.svg)](https://github.com/alexander-held/cabinetry/actions?query=workflow%3ACI)
[![Documentation Status](https://readthedocs.org/projects/cabinetry/badge/?version=latest)](https://cabinetry.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/alexander-held/cabinetry/branch/master/graph/badge.svg)](https://codecov.io/gh/alexander-held/cabinetry)
[![PyPI version](https://badge.fury.io/py/cabinetry.svg)](https://badge.fury.io/py/cabinetry)
[![python version](https://img.shields.io/pypi/pyversions/cabinetry.svg)](https://pypi.org/project/cabinetry/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Table of contents

- [Introduction](#introduction)
- [Hello world](#hello-world)
- [Template fits](#template-fits)
- [Scope](#scope)
- [Code](#code)
- [Acknowledgements](#acknowledgements)

## Introduction

`cabinetry` is a Python package to build and steer (profile likelihood) template fits with applications in high energy physics in mind.
It acts as an interface to many powerful tools to make it easier for an analyzer to run their statistical inference pipeline.
An incomplete list of interesting tools to interface:

- [ServiceX](https://github.com/ssl-hep/ServiceX) for data delivery,
- [coffea](https://github.com/CoffeaTeam/coffea) for histogram processing,
- [uproot](https://github.com/scikit-hep/uproot4) for reading [ROOT](https://root.cern.ch/) files,
- for building likelihood functions (captured in so-called workspaces in [RooFit](https://root.cern.ch/roofit)) and inference:
  - [RooFit](https://root.cern.ch/roofit) to model probability distributions,
  - [RooStats](http://cds.cern.ch/record/1289965) for statistical tools,
  - [HistFactory](https://cds.cern.ch/record/1456844/) to implement a subset of binned template fits,
  - [pyhf](https://github.com/scikit-hep/pyhf) for a pythonic take on [HistFactory](https://cds.cern.ch/record/1456844/),
  - [zfit](https://github.com/zfit/zfit) for a pythonic take on [RooFit](https://root.cern.ch/roofit),
  - [MadMiner](https://github.com/diana-hep/madminer) for likelihood-free inference techniques (see [Scope](#scope)).

The project is a work in progress.
Configuration of `cabinetry` happens in a declarative manner, and is easily serializable via JSON/YAML into a configuration file.

Interesting related projects:

- [pyhfinput](https://github.com/lukasheinrich/pyhfinput)
- [ServiceX for TRExFitter](https://github.com/kyungeonchoi/ServiceXforTRExFitter)
- [Template fit workflows](https://github.com/alexander-held/template-fit-workflows)
- [TRExFitter config translation](https://github.com/alexander-held/TRExFitter-config-translation)

## Hello world

To run the following example, first generate the input files via the script `util/create_ntuples.py`.

```python
import cabinetry

cabinetry_config = cabinetry.configuration.load("config_example.yml")

# create template histograms
cabinetry.template_builder.create_histograms(cabinetry_config)

# perform histogram post-processing
cabinetry.template_postprocessor.run(cabinetry_config)

# build a workspace
ws = cabinetry.workspace.build(cabinetry_config)

# run a fit
fit_results = cabinetry.fit.fit(ws)

# visualize the post-fit model prediction and data
cabinetry.visualize.data_MC(cabinetry_config, ws, fit_results=fit_results)
```

The above is an abbreviated version of an example included in `example.py`, which shows how to use `cabinetry`.
It requires additional libraries beyond the core dependencies of `cabinetry`, which can be installed via `pip install cabinetry[contrib]` (or `pip install -e .[contrib]` from the repository).
Eventually the basic implementation (from `cabinetry/contrib`) will be replaced by calls to external modules (see also [Code](#code)).

## Template fits

The operations needed in a template fit workflow can be summarized as follows:

1. [Template histogram production](#1-template-histogram-production),
2. [Histogram adjustments](#2-histogram-adjustments),
3. [Workspace creation from histograms](#3-workspace-creation-from-histograms),
4. [Inference from workspace](#4-inference-from-workspace),
5. [Visualization](#5-visualization).

While the first four points need to happen in this order (as each step uses as input the output of the previous step), the visualization is relevant at all stages to not only show final results, but also intermediate steps.

### 1. Template histogram production

The production of a template histogram requires the following information:

- where to find the data (and how to read it),
- what kind of selection requirements (filtering) and weights to apply to the data,
- the variable to bin in, and what bins to use (for binned fits)
- a unique name (key) for this histogram to be able to refer to it later on.

In practice, histogram information can be given by specifying lists of:

- regions of phase space (or channels, independent regions obtained via different selection requirements),
- samples (physics processes),
- systematic uncertainties for the samples, which might vary across samples and phase space regions.

For LHC-style template profile likelihood fits, typically a few thousand histograms are needed.
An analysis that considers 5 different phase space regions, with 10 different physics processes (simulated as 10 independent Monte Carlo samples), and an average of 50 systematic uncertainties for all the samples (implemented by specifying variations from the nominal configuration in two directions), needs `5x10x100=5000` histograms.

### 2. Histogram adjustments

Histogram post-processing can include re-binning, smoothing, or symmetrization of systematic uncertainties.
These operations should be handled by tools outside of `cabinetry`.
Such tools might either need some additional steering via an additional configuration, or the basic configuration file has to support arbitrary settings to be passed to these tools (depending on what each tool can interpret).

### 3. Workspace creation from histograms

Taking the example of [pyhf](https://github.com/scikit-hep/pyhf), the workspace creation consists of plugging histograms into the right places in a JSON file.
This can be relatively straightforward if the configuration file is very explicit about these assignments.
In practice, it is desirable to support convenience options in the configuration file.
An example is the ability to de-correlate the effect of a systematic uncertainty across different phase space regions via a simple flag.
This means that instead of one nuisance parameter, many nuisance parameters need to be created automatically.
The treatment can become complicated when many such convenience functions interact with each other.

A possible approach is to define a lowest level configuration file format that supports no convenience functions at all and everything specified in a very explicit manner.
Convenience functions could be supported in small frameworks that can read configuration files containing flags for convenience functions, and those small frameworks could convert the configuration file into the low level format.

The basic task of building the workspace should have well-defined inputs (low-level configuration file) and outputs (such as [HistFactory](https://cds.cern.ch/record/1456844/) workspaces).
Support of convenience functions can be factored out, with a well-defined output (low-level configuration file) and input given by an enhanced configuration file format.

### 4. Inference from workspace

Inference happens via fits of the workspace, to obtain best-fit results and uncertainties, limits on parameters, significances of observations and so on.
External tools are called to perform inference, configured as specified by the configuration file.

### 5. Visualization

Some information of relevant kinds of visualization is provided in [as-user-facing/fit-visualization.md](https://github.com/iris-hep/as-user-facing/blob/master/fit-visualization.md) and the links therein.

## Scope

For now, `cabinetry` is focused on [HistFactory](https://cds.cern.ch/record/1456844/) style template fit models.
Those traditional binned template fits are substantially easier to support than the open world of binned and unbinned models.
Likelihood-free inference approaches in  the style of [MadMiner](https://github.com/diana-hep/madminer) have a more well-defined scope than the open world of [RooFit](https://root.cern.ch/roofit), and might be easier to integrate.

## Code

Everything in `cabinetry/contrib` are basic implementation of tasks that should be done by other tools, and interfaces to those tools should be added.
The basic implementations that exist there help with API design.

## Acknowledgements

[![NSF-1836650](https://img.shields.io/badge/NSF-1836650-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1836650)

This work was supported by the U.S. National Science Foundation (NSF) cooperative agreement [OAC-1836650 (IRIS-HEP)](https://nsf.gov/awardsearch/showAward?AWD_ID=1836650).
