Advanced concepts
=================

Accessing vector branches
-------------------------

The transverse momentum of the first jet in a vector branch ``jet_pT`` is obtained via ``jet_pT[0]`` in ``ROOT``.
The ``uproot`` backend for ntuple reading treats expressions (such as what is written in ``Filter`` and ``Weight`` configuration file options) as Python code.
The correct way to access the same information through ``cabinetry`` is ``jet_pT[:,0]``, where the first index runs over events.


Overrides for template building
-------------------------------

Introduction
^^^^^^^^^^^^

It is possible to define functions that are called when ``cabinetry`` tries to construct a template histogram.
Such functions need to accept four arguments in the following order:

- a dictionary with information about the region being processed,
- a dictionary with information about the sample being processed,
- a dictionary with information about the systematic being processed,
- the template being considered: a string ``"Up"`` / ``"Down"`` for variations, or ``None`` for the nominal template.

The function needs to return a `boost-histogram Histogram <https://boost-histogram.readthedocs.io/en/latest/usage/histogram.html>`_.
This histogram is then further processed in ``cabinetry``.

Example
^^^^^^^

The example below defines a function ``build_data_hist``.
The decorator specifies that this function should be applied to all histograms for samples with name ``ttbar``.
It is also possible to specify ``region_name``, ``systematic_name`` and ``template`` for the names of the region, systematic and template.
Not specifying these options means not restricting the applicability of the function.
When no user-defined function matches a given histogram that has to be produced, ``cabinetry`` falls back to use the default histogram creation methods.

.. code-block:: python

    from typing import Optional

    import boost_histogram as bh
    import numpy as np
    import cabinetry

    my_router = cabinetry.route.Router()

    # define a custom template builder function that is executed for data samples
    @my_router.register_template_builder(sample_name="ttbar")
    def build_data_hist(
        region: dict, sample: dict, systematic: dict, template: str | None
    ) -> bh.Histogram:
        hist = bh.Histogram(
            bh.axis.Variable(region["Binning"], underflow=False, overflow=False),
            storage=bh.storage.Weight(),
        )
        yields = np.asarray([17, 12, 25, 20])
        variance = np.asarray([1.5, 1.2, 1.8, 1.6])
        hist[...] = np.stack([yields, variance], axis=-1)
        return hist


    cabinetry.templates.build(
        cabinetry_config, method="uproot", router=my_router
    )

The instance of ``cabinetry.route.Router`` is handed to ``cabinetry.templates.build`` to enable the use of ``build_data_hist``.

The function ``build_data_hist`` in this example always returns the same histogram.
Given that the dictionaries in the function signature provide additional information, it is for example possible to return different yields per region:

.. code-block:: python

    if region["Name"] == "Signal_region":
        yields = np.asarray([17, 12, 25, 20])
    elif region["Name"] == "Background_region":
        yields = np.asarray([102, 121, 138, 154])


Wildcards and multiple requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is also possible to use wildcards to specify which templates a function should be applied to.
The implementation currently makes use of `fnmatch <https://docs.python.org/3/library/fnmatch.html>`_.
The following decorator

.. code-block:: python

    @my_router.register_template_builder(sample_name="ttbar_*")

means that the function will for example be applied if the sample name is `ttbar_ljets` or `ttbar_dil`, but not if it is `single_top`.
All conditions need to be fulfilled to apply a user-defined function, so

.. code-block:: python

    @my_router.register_template_builder(
        region_name="signal_region",
        sample_name="signal",
        systematic="alpha_S",
        template="*",
    )

means that for the decorated function to be executed, the region name needs to be `signal_region`, the sample needs to be called `signal`, the systematic needs to be `alpha_S`, but there is no restriction to the template name.

Since ``template`` can be a string or ``None``, its behavior is slightly different:

- ``template="*"`` is the default, and means that any histogram matches (nominal, as well as variations),
- ``template=None`` matches only nominal histograms,
- ``template=string``, where ``string`` is any string other than ``"*"``, can never match the nominal template, but could match the systematic variations called ``"Up"`` and ``"Down"``.


Fixed parameters
----------------

The ``cabinetry`` configuration file contains the ``Fixed`` option (in the ``General`` group of options), which allows for the creation of a workspace with parameters set to be constant.

.. code-block:: yaml

    Fixed:
      - Name: par_a
        Value: 2
      - Name: par_b
        Value: 1

The same can be written in a more compact way:

.. code-block:: yaml

    Fixed: [{"Name": "par_a", "Value": 2},{"Name": "par_b", "Value": 1}]

The associated ``pyhf`` workspace will contain the following:

.. code-block:: json

    {
      "measurements": [
        {
          "config": {
            "parameters": [
              {"fixed": true, "inits": [2], "name": "par_a"},
              {"fixed": true, "inits": [1], "name": "par_b"}
            ]
          }
        }
      ]
    }

Fixed parameters are not allowed to vary in fits.
Both their pre-fit and post-fit uncertainty are set to zero.
This means that the associated nuisance parameters do not contribute to uncertainty bands in data/MC visualizations either.
The impact of such parameters on the parameter of interest (for nuisance parameter ranking) is also zero.


Manually correlating systematics
--------------------------------

Systematic uncertainties are correlated if the modifiers defining them in the ``pyhf`` workspace have the same names.
The example below shows a modifier called `correlated_modifier`, correlated between two samples in a workspace.

.. code-block:: json

    [
      {
        "data": [25.0],
        "modifiers": [
          {
            "data": {"hi": 1.05, "lo": 0.95},
            "name": "correlated_modifier",
            "type": "normsys"
          }
        ],
        "name": "Signal"
      },
      {
        "data": [55.0],
        "modifiers": [
          {
            "data": {"hi": 1.05, "lo": 0.95},
            "name": "correlated_modifier",
            "type": "normsys"
          }
        ],
        "name": "Background"
      }
    ]

The names of modifiers written to the workspace are by default picked up from the name of the associated systematic in the ``cabinetry`` configuration.
Names of systematics in the configuration need to be unique, so it is not possible to define multiple systematics with the same name.
Instead, the option ``ModifierName`` can be used to specify the name of the associated modifier(s) used in the workspace:

.. code-block:: yaml

    Systematics:
      - Name: "first_systematic"
        Up:
          Normalization: 0.05
        Down:
          Normalization: -0.05
        Type: "Normalization"
        Samples: "Signal"
        ModifierName: "correlated_modifier"

      - Name: "second_systematic"
        Up:
          Normalization: 0.05
        Down:
          Normalization: -0.05
        Type: "Normalization"
        Samples: "Background"
        ModifierName: "correlated_modifier"

This results in a workspace like the example shown above.
Without ``ModifierName``, the two modifiers would be uncorrelated and called `first_systematic` and `second_systematic`.

In this simple example, the following settings result in the same workspace:

.. code-block:: yaml

    Systematics:
      - Name: "correlated_modifier"
        Up:
          Normalization: 0.05
        Down:
          Normalization: -0.05
        Type: "Normalization"
        Samples: ["Signal", "Background"]

The approach of manually correlating different systematics however allows to define systematics in different ways (e.g. different normalization effect per sample), while still keeping them correlated.

Internally, ``cabinetry`` refers to systematics by their unique name up until the workspace building stage.
For statistical inference, information contained in the workspace is used and thus the original systematics names are replaced by the values set in ``ModifierName`` (if that option is used).
