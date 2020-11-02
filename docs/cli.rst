CLI
===

After installing ``cabinetry``, a command line interface is available.
Below is an example workflow that builds template histograms defined by the config file ``config_example.yml``, and applies post-processing to them.
A ``pyhf`` workspace is then constructed and a maximum likelihood fit is performed.
The resulting correlation matrix and pull plot are saved to the default output folder ``figures/``.

.. code-block:: bash

    cabinetry templates config_example.yml
    cabinetry postprocess config_example.yml
    cabinetry workspace config_example.yml workspaces/example_workspace.json
    cabinetry fit --pulls --corrmat workspaces/example_workspace.json

The ``--help`` flag can be used to obtain more information on the command line:

.. code-block:: bash

    cabinetry --help

shows the available commands, while

.. code-block:: bash

    cabinetry fit --help

shows what the ``fit`` command does, and which options it accepts.

It is possible to read the ``cabinetry`` config and workspaces from stdin, and to write workspaces to stdout:

.. code-block:: bash

    # read config from stdin
    cat config_example.yml | cabinetry workspace - workspaces/example_workspace.json
    # read workspace from stdin
    cat workspaces/example_workspace.json | cabinetry fit -
    # write workspace to stdout
    cabinetry workspace config_example.yml -


.. click:: cabinetry.cli:cabinetry
    :prog: cabinetry
    :nested: full
