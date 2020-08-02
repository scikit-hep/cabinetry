CLI
===

The installation of `cabinetry` includes a command line interface.
Below is an example workflow that builds template histograms defined by the config file `config_example.yml`, and applies post-processing to them.
A `pyhf` workspace is then constructed and a maximum likelihood fit is performed.
The resulting correlation matrix and pull plot are saved to the default output folder `figures/`.

.. code-block:: bash

   cabinetry templates config_example.yml
   cabinetry postprocess config_example.yml
   cabinetry workspace config_example.yml workspaces/example_workspace.json
   cabinetry fit --pulls --corrmat workspaces/example_workspace.json


.. click:: cabinetry.cli:cabinetry
   :prog: cabinetry
   :nested: full
