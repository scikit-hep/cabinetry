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
- a string with the name of the template being processed: ``Nominal``, ``Up`` or ``Down``.

The function needs to return a `boost-histogram Histogram <https://boost-histogram.readthedocs.io/en/latest/usage/histogram.html>`_.
This histogram is then further processed in ``cabinetry``.

Example
^^^^^^^

The example below defines a function ``build_data_hist``.
The decorator specifies that this function should be applied to all histograms for samples with name ``Data``.
It is also possible to specify ``region_name``, ``systematic_name`` and ``template`` for the names of the region, systematic and template.
When no user-defined function matches a given histogram that has to be produced, ``cabinetry`` falls back to use the default histogram creation methods.

.. code-block:: python

    import boost_histogram as bh
    import cabinetry

    my_router = cabinetry.route.Router()

    # define a custom template builder function that is executed for data samples
    @my_router.register_template_builder(sample_name="Data")
    def build_data_hist(
        region: dict, sample: dict, systematic: dict, template: str
    ) -> bh.Histogram:
        hist = bh.Histogram(
            bh.axis.Variable(reg["Binning"], underflow=False, overflow=False),
            storage=bh.storage.Weight(),
        )
        yields = np.asarray([17, 12, 25, 20])
        variance = np.asarray([1.5, 1.2, 1.8, 1.6])
        hist[...] = np.stack([yields, variance], axis=-1)
        return hist


    cabinetry.template_builder.create_histograms(
        cabinetry_config, histo_folder, method="uproot", router=my_router
    )

The instance of ``cabinetry.route.Router`` is handed to ``cabinetry.template_builder.create_histograms`` to enable the use of ``build_data_hist``.

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
Omitting ``template`` from the arguments, or using the default ``template=None`` has the same result.


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
