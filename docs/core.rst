Core concepts
=============

Inputs to cabinetry: ntuples or histograms
------------------------------------------

``cabinetry`` supports two types of input files when building a workspace: ntuples containing columnar data and histograms.
When using ntuple inputs, ``cabinetry`` needs to know not only where to find the input files for every template histogram it needs to build, but also what selections to apply, which column to extract and how to weigh every event.
The configuration schema lists the required options, see :ref:`config` for more information.
Less information is required when using histogram inputs: only the path to each histogram needs to be specified in this case.

Input file path specification for ntuples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Paths to ntuple input files for histogram production are specified with the mandatory ``InputPath`` setting in the ``General`` config section.
If everything is in one file, the value should be the path to this file.
It is common to have multiple input files, split across phase space regions or samples.
For this purpose, the ``InputPath`` value can take two placeholders: ``{RegionPath}`` and ``{SamplePath}``.

RegionPath
""""""""""

When building histograms for a specific region, the ``{RegionPath}`` placeholder takes the value specified in the ``RegionPath`` setting of the corresponding region.
The value of ``RegionPath`` has to be a string.

SamplePath
""""""""""

The ``{SamplePath}`` placeholder takes the value given by ``SamplePath`` of the sample currently processed.
This value can either be a string or a list of strings.
If it is a list, multiple copies of ``InputPath`` are created, and in each of them the ``{SamplePath}`` placeholder takes the value of a different entry in the list.
All input files are processed, and their contributions are summed together.
The histogram created by ``SamplePath: ["a.root", "b.root"]`` is equivalent to the histogram created with ``SamplePath: "a_plus_b.root"``, where ``a_plus_b.root`` is produced by merging both files.

Systematics
"""""""""""

It is possible to specify overrides for the ``RegionPath`` and ``SamplePath`` values in systematic templates.
If those settings are specified in the ``Up`` or ``Down`` template section of a systematic uncertainty, then the corresponding values are used when building the path to the file used to construct the histogram for this specific template.

An example
""""""""""

The following configuration file excerpt shows an example of specifying paths to input files.

.. code-block:: yaml

    General:
      InputPath: "inputs/{RegionPath}/{SamplePath}"

    Regions:
      - Name: "Signal_region"
        RegionPath: "signal_region"

      - Name: "Control_region"
        RegionPath: "control_region"

    Samples:
      - Name: "Data"
        SamplePath: "data.root"

      - Name: "Signal"
        SamplePath: ["signal_1.root", "signal_2.root"]

    Systematics:
      - Name: "Signal_modeling"
        Up:
          SamplePath: "modeling_variation_up.root"
        Down:
          SamplePath: "modeling_variation_down.root"
        Samples: "Signal"

The following files will be read to create histograms:

- for *Signal_region*:

    - *Data*: ``inputs/signal_region/data.root``
    - *Signal*: ``inputs/signal_region/signal_1.root``, ``inputs/signal_region/signal_2.root``

        - systematic uncertainty:

            - *up*: ``inputs/signal_region/modeling_variation_up.root``
            - *down*: ``inputs/signal_region/modeling_variation_down.root``

- for *Control_region*:

    - *Data*: ``inputs/control_region/data.root``
    - *Signal*: ``inputs/control_region/signal_1.root``, ``inputs/control_region/signal_2.root``

        - systematic uncertainty:

            - *up*: ``inputs/control_region/modeling_variation_up.root``
            - *down*: ``inputs/control_region/modeling_variation_down.root``

Input file path specification for histograms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The specification of paths to histograms works very similarly to the ntuple case.
The ``InputPath`` setting in the ``General`` config section is still mandatory.
It can again take placeholders: ``{RegionPath}``, ``{SamplePath}``, and ``{VariationPath}``.
The ``VariationPath`` setting will default to an empty string if not specified, but it can be set to another value (such as ``"nominal"``) in the ``General`` block.

A major difference to the ntuple path construction is that the histogram path needs to not only include the path to the file containing a given histogram, but also to the histogram within the file.
This is achieved by using a colon ``:`` to distinguish between both parts of the path: ``folder/file.root:abc/h1`` points to a histogram called ``h1`` located in a folder called ``abc`` which itself exists within a file called ``file.root`` which can be found in a folder called ``folder``.

RegionPath
""""""""""

This works in the same way as it does for ntuples: the ``RegionPath`` setting in each region sets the value for the ``{RegionPath}`` placeholder.
Note that the value cannot be overridden on a per-systematic basis in the histogram case.

SamplePath
""""""""""

The ``SamplePath`` setting sets the value for the ``{SamplePath}`` placeholder.
In contrast to the ntuple case, this value cannot be a list of strings.
It also cannot be overridden on a per-systematic basis, just like ``RegionPath``.

VariationPath
"""""""""""""

Each systematic template can set the value for the ``{VariationPath}`` placeholder via the ``VariationPath`` setting.
``RegionPath`` and ``SamplePath`` settings cannot be overridden.

An example
""""""""""

The following shows an example, similar to the ntuple example.

.. code-block:: yaml

    General:
      InputPath: "inputs/{RegionPath}.root:{SamplePath}_{VariationPath}"
      VariationPath: "nominal"

    Regions:
      - Name: "Signal_region"
        RegionPath: "signal_region"

      - Name: "Control_region"
        RegionPath: "control_region"

    Samples:
      - Name: "Data"
        SamplePath: "data"

      - Name: "Signal"
        SamplePath: "signal"

    Systematics:
      - Name: "Signal_modeling"
        Up:
          VariationPath: "modeling_variation_up"
        Down:
          VariationPath: "modeling_variation_down"
        Samples: "Signal"

The following histograms will be read:

- for *Signal_region*:

    - *Data*: ``inputs/signal_region.root:data_nominal``
    - *Signal*: ``inputs/signal_region.root:signal_nominal``

        - systematic uncertainty:

            - *up*: ``inputs/signal_region.root:signal_modeling_variation_up``
            - *down*: ``inputs/signal_region:signal_modeling_variation_down``

- for *Control_region*:

    - *Data*: ``inputs/control_region.root:data_nominal``
    - *Signal*: ``inputs/control_region.root:signal_nominal``

        - systematic uncertainty:

            - *up*: ``inputs/control_region.root:signal_modeling_variation_up``
            - *down*: ``inputs/control_region:signal_modeling_variation_down``
