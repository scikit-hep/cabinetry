Core concepts
=============

Input file path specification
-----------------------------

Paths to input files for histogram production are specified with the mandatory ``InputPath`` setting in the ``General`` config section.
If everything is in one file, the value should be the path to this file.
It is common to have multiple input files, split across phase space regions or samples.
For this purpose, the ``InputPath`` value can take two placeholders: ``{RegionPath}`` and ``{SamplePaths}``.

RegionPath
^^^^^^^^^^

When building histograms for a specific region, the ``{RegionPath}`` placeholder takes the value specified in the ``RegionPath`` setting of the corresponding region.
The value of ``RegionPath`` has to be a string.

SamplePaths
^^^^^^^^^^^

The ``{SamplePaths}`` placeholder takes the value given by ``SamplePaths`` of the sample currently processed.
This value can either be a string or a list of strings.
If it is a list, multiple copies of ``InputPath`` are created, and in each of them the ``{SamplePaths}`` placeholder takes the value of a different entry in the list.
All input files are processed, and their contributions are summed together.
The histogram created by ``SamplePaths: ["a.root", "b.root"]`` is equivalent to the histogram created with ``SamplePaths: "a_plus_b.root"``, where ``a_plus_b.root`` is produced by merging both files.

Systematics
^^^^^^^^^^^

It is possible to specify overrides for the ``RegionPath`` and ``SamplePaths`` values in systematic templates.
If those settings are specified in the ``Up`` or ``Down`` template section of a systematic uncertainty, then the corresponding values are used when building the path to the file used to construct the histogram for this specific template.

An example
^^^^^^^^^^

The following configuration file excerpt shows an example of specifying paths to input files.

.. code-block:: yaml

    General:
      InputPath: "ntuples/{RegionPath}/{SamplePaths}"

    Regions:
      - Name: "Signal_region"
        RegionPath: "signal_region"

      - Name: "Control_region"
        RegionPath: "control_region"

    Samples:
      - Name: "Data"
        SamplePaths: "data.root"

      - Name: "Signal"
        SamplePaths: ["signal_1.root", "signal_2.root"]

    Systematics:
      - Name: "Signal_modeling"
        Up:
          SamplePaths: "signal_variation_up.root"
        Down:
          SamplePaths: "signal_variation_down.root"
        Samples: "Signal"

The following files will be read to create histograms:

- for *Signal_region*:

    - *Data*: ``ntuples/signal_region/data.root``
    - *Signal*: ``ntuples/signal_region/signal_1.root``, ``ntuples/signal_region/signal_2.root``

        - systematic uncertainty:

            - *up*: ``ntuples/signal_region/signal_variation_up.root``
            - *down*: ``ntuples/signal_region/signal_variation_down.root``

- for *Control_region*:

    - *Data*: ``ntuples/control_region/data.root``
    - *Signal*: ``ntuples/control_region/signal_1.root``, ``ntuples/control_region/signal_2.root``

        - systematic uncertainty:

            - *up*: ``ntuples/control_region/signal_variation_up.root``
            - *down*: ``ntuples/control_region/signal_variation_down.root``
