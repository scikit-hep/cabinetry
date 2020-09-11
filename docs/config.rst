Configuration schema
====================

The configuration schema for ``cabinetry`` is given below.
It is defined via a `json schema <https://json-schema.org/>`_ that can be found at `src/cabinetry/schemas/config.json <https://github.com/alexander-held/cabinetry/blob/master/src/cabinetry/schemas/config.json>`_.

The ``General`` block holds general settings, followed by blocks that take lists of objects: regions, samples, normalization factors, and systematics.
The ``Regions``, ``Samples`` and ``NormFactors`` blocks are required, while ``Systematics`` is optional.
Settings shown in bold are required.

.. jsonschema:: ../src/cabinetry/schemas/config.json#/properties/General
.. jsonschema:: ../src/cabinetry/schemas/config.json#/properties/Regions
.. jsonschema:: ../src/cabinetry/schemas/config.json#/properties/Samples
.. jsonschema:: ../src/cabinetry/schemas/config.json#/properties/NormFactors
.. jsonschema:: ../src/cabinetry/schemas/config.json#/properties/Systematics

Details about the setting blocks:
---------------------------------

.. jsonschema:: ../src/cabinetry/schemas/config.json#/definitions/general

.. jsonschema:: ../src/cabinetry/schemas/config.json#/definitions/region

.. jsonschema:: ../src/cabinetry/schemas/config.json#/definitions/sample

.. jsonschema:: ../src/cabinetry/schemas/config.json#/definitions/normfactor

.. jsonschema:: ../src/cabinetry/schemas/config.json#/definitions/systematic

Common options:
---------------

.. jsonschema:: ../src/cabinetry/schemas/config.json#/definitions/template

.. jsonschema:: ../src/cabinetry/schemas/config.json#/definitions/samples_setting

.. jsonschema:: ../src/cabinetry/schemas/config.json#/definitions/samplepaths_setting


