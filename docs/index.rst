`cabinetry`
==============================

.. toctree::
   :hidden:
   :maxdepth: 1


   api
   license

`cabinetry` is a tool to build and steer (profile likelihood) template fits with applications in high energy physics in mind.
It acts as an interface to many powerful tools to make it easier for an analyzer to run their statistical inference pipeline.

This documentation can be built and viewed via

.. code-block:: bash

   sphinx-build docs docs/_build
   open docs/_build/index.html

Hello world
-----------

.. code-block:: python

   import cabinetry

   cabinetry_config = cabinetry.configuration.read("config_example.yml")

   # create template histograms
   histo_folder = "histograms/"
   cabinetry.template_builder.create_histograms(
       cabinetry_config, histo_folder, method="uproot"
   )

   # perform histogram post-processing
   cabinetry.template_postprocessor.run(cabinetry_config, histo_folder)

   # visualize templates and data
   cabinetry.visualize.data_MC(
       cabinetry_config, histo_folder, "figures/", prefit=True, method="matplotlib"
   )

   # build a workspace
   ws = cabinetry.workspace.build(cabinetry_config, histo_folder)

   # run a fit
   cabinetry.fit.fit(ws)

The above is an abbreviated version of an example included in `example.py`, which shows how to use `cabinetry`.
Beyond the core dependencies of `cabinetry` (currently `pyyaml`, `numpy`, `pyhf`, `iminuit`), it also requires additional libraries: `uproot`, `scipy`, `matplotlib`, `numexpr`.

Acknowledgements
----------------
This work was supported by the U.S. National Science Foundation (NSF) cooperative agreement OAC-1836650 (IRIS-HEP).
