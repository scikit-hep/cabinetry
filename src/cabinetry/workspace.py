"""Constructs HistFactory workspaces in JSON format."""

import json
import logging
import pathlib
from typing import Any, cast

import pyhf

from cabinetry import configuration
from cabinetry import histo

log = logging.getLogger(__name__)


class WorkspaceBuilder:
    """Collects functionality to build a ``pyhf`` workspace."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Creates a workspace corresponding to a cabinetry configuration.

        Args:
            config (Dict[str, Any]): ``cabinetry`` configuration
        """
        self.config = config
        self.histogram_folder = pathlib.Path(config["General"]["HistogramFolder"])

    def _data_sample(self) -> dict[str, Any]:
        """Returns the data sample dictionary.

        Returns:
            Dict[str, Any]: the data sample dictionary
        """
        data_samples = [
            sample for sample in self.config["Samples"] if sample.get("Data", False)
        ]
        if len(data_samples) != 1:
            raise ValueError("did not find exactly one data sample")
        return data_samples[0]

    def _constant_parameter_setting(self, par_name: str) -> float | None:
        """Determines whether a parameter should be set to constant, and to which value.

        For a given parameter, determines if it is supposed to be set to constant. If
        not, returns None, otherwise returns the value it should be fixed to. This only
        looks for the first occurrence of the parameter in the list.

        Args:
            par_name (str): name of parameter to check

        Returns:
            float | None: returns None if parameter is not supposed to be held constant,
            or the value it has to be fixed to
        """
        fixed_parameters = self.config["General"].get("Fixed", [])
        fixed_value = next(
            (
                fixed_par["Value"]
                for fixed_par in fixed_parameters
                if fixed_par["Name"] == par_name
            ),
            None,
        )
        return fixed_value

    def normfactor_modifiers(
        self, region: dict[str, Any], sample: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Returns the list of NormFactor modifiers acting on a sample in a region.

        Args:
            region (Dict[str, Any]): specific region to get NormFactor modifiers for
            sample (Dict[str, Any]): specific sample to get NormFactor modifiers for

        Returns:
            List[Dict[str, Any]]: NormFactor modifiers for sample
        """
        modifiers = []
        for norm_factor in self.config["NormFactors"]:
            # check that region and sample are both not excluded by modifier
            if configuration.region_contains_modifier(
                region, norm_factor
            ) and configuration.sample_contains_modifier(sample, norm_factor):
                log.debug(
                    f"adding NormFactor {norm_factor['Name']} to sample "
                    f"{sample['Name']} in region {region['Name']}"
                )
                modifiers.append(
                    {"data": None, "name": norm_factor["Name"], "type": "normfactor"}
                )
        return modifiers

    @staticmethod
    def normalization_modifier(systematic: dict[str, Any]) -> dict[str, Any]:
        """Returns a normalization modifier (OverallSys in `HistFactory`).

        Args:
            systematic (Dict[str, Any]): systematic for which modifier is constructed

        Returns:
            Dict[str, Any]: single `normsys` modifier for ``pyhf`` workspace
        """
        # take name of modifier from ModifierName if set, default to systematic name
        modifier_name = systematic.get("ModifierName", systematic["Name"])

        modifier = {}
        modifier.update({"name": modifier_name})
        modifier.update({"type": "normsys"})
        modifier.update(
            {
                "data": {
                    "hi": 1 + systematic["Up"]["Normalization"],
                    "lo": 1 + systematic["Down"]["Normalization"],
                }
            }
        )
        return modifier

    def normplusshape_modifiers(
        self, region: dict[str, Any], sample: dict[str, Any], systematic: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Returns modifiers for a correlated shape + normalization effect.

        For a variation including a correlated shape + normalization effect, this
        provides the `histosys` and `normsys` modifiers for ``pyhf`` (in `HistFactory`
        language, this corresponds to a `HistoSys` and an `OverallSys`). Symmetrization
        could happen either at this stage (this is the case currently), or somewhere
        earlier, such as during template postprocessing. A `histosys` modifier is not
        created for single-bin channels, as it has no effect in this case (everything is
        handled by the `normsys` modifier already).

        Args:
            region (Dict[str, Any]): region the systematic variation acts in
            sample (Dict[str, Any]): sample the systematic variation acts on
            systematic (Dict[str, Any]): the systematic variation under consideration

        Raises:
            ValueError: when both up and down variation specify symmetrization

        Returns:
            List[Dict[str, Any]]: a list with a ``pyhf`` `normsys` modifier and a
            `histosys` modifier
        """
        # ensure that not both up and down variations are built by symmetrization
        symmetrize_up = systematic.get("Up", {}).get("Symmetrize", False)
        symmetrize_down = systematic.get("Down", {}).get("Symmetrize", False)
        if symmetrize_up and symmetrize_down:
            raise ValueError(
                f"up and down variation of systematic {systematic['Name']} cannot both "
                "be symmetrized"
            )

        # load the nominal histogram
        histogram_nominal = histo.Histogram.from_config(
            self.histogram_folder, region, sample, {}, modified=True
        )

        if not symmetrize_up:
            # load the systematic variation histogram for the up variation
            histogram_up = histo.Histogram.from_config(
                self.histogram_folder,
                region,
                sample,
                systematic,
                template="Up",
                modified=True,
            )

        if not symmetrize_down:
            # load the systematic variation histogram for the down variation
            histogram_down = histo.Histogram.from_config(
                self.histogram_folder,
                region,
                sample,
                systematic,
                template="Down",
                modified=True,
            )

        if symmetrize_down:
            histo_yield_up, histo_yield_down, norm_effect_up, norm_effect_down = (
                _symmetrized_templates_and_norm(histogram_up, histogram_nominal)
            )
        elif symmetrize_up:
            histo_yield_down, histo_yield_up, norm_effect_down, norm_effect_up = (
                _symmetrized_templates_and_norm(histogram_down, histogram_nominal)
            )
        else:
            norm_effect_up = histogram_up.normalize_to_yield(histogram_nominal)
            norm_effect_down = histogram_down.normalize_to_yield(histogram_nominal)
            # manually cast due to https://github.com/numpy/numpy/issues/27944
            histo_yield_up = cast(list[float], histogram_up.yields.tolist())
            histo_yield_down = cast(list[float], histogram_down.yields.tolist())

        log.debug(
            f"normalization impact of systematic {systematic['Name']} on sample "
            f"{sample['Name']} in region {region['Name']} is {norm_effect_up:.3f} "
            f"(up) {norm_effect_down:.3f} (down)"
        )

        # take name of modifier from ModifierName if set, default to systematic name
        modifier_name = systematic.get("ModifierName", systematic["Name"])

        # add the normsys
        modifiers = []
        norm_modifier = {}
        norm_modifier.update({"name": modifier_name})
        norm_modifier.update({"type": "normsys"})
        norm_modifier.update({"data": {"hi": norm_effect_up, "lo": norm_effect_down}})
        modifiers.append(norm_modifier)

        # add the shape part in a histosys
        if len(histogram_nominal.yields) > 1:
            # only relevant if there is more than one bin, otherwise there is no "shape"
            shape_modifier = {}
            shape_modifier.update({"name": modifier_name})
            shape_modifier.update({"type": "histosys"})
            shape_modifier.update(
                {"data": {"hi_data": histo_yield_up, "lo_data": histo_yield_down}}
            )
            modifiers.append(shape_modifier)
        return modifiers

    def sys_modifiers(
        self, region: dict[str, Any], sample: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Returns the list of all systematic modifiers acting on a sample in a region.

        Args:
            region (Dict[str, Any]): region considered
            sample (Dict[str, Any]): specific sample to get modifiers for

        Raises:
            NotImplementedError: when unsupported modifiers act on sample

        Returns:
            List[Dict[str, Any]]: modifiers for ``pyhf`` workspace
        """
        modifiers = []
        for systematic in self.config.get("Systematics", []):
            # check that region and sample both do not exclude modifier
            if configuration.region_contains_modifier(
                region, systematic
            ) and configuration.sample_contains_modifier(sample, systematic):
                if systematic["Type"] == "Normalization":
                    # OverallSys (norm uncertainty with Gaussian constraint)
                    log.debug(
                        f"adding OverallSys {systematic['Name']} to sample"
                        f" {sample['Name']} in region {region['Name']}"
                    )
                    modifiers.append(self.normalization_modifier(systematic))
                elif systematic["Type"] == "NormPlusShape":
                    # two modifiers are needed - an OverallSys for the norm effect,
                    # and a HistoSys for the shape variation
                    log.debug(
                        f"adding OverallSys and HistoSys {systematic['Name']} to sample"
                        f" {sample['Name']} in region {region['Name']}"
                    )
                    modifiers += self.normplusshape_modifiers(
                        region, sample, systematic
                    )
                else:
                    raise NotImplementedError(
                        "not supporting other systematic types yet"
                    )
        return modifiers

    def channels(self) -> list[dict[str, Any]]:
        """Returns the channel information: yields per sample and modifiers.

        Returns:
            List[Dict[str, Any]]: channels for ``pyhf`` workspace
        """
        channels = []
        for region in self.config["Regions"]:
            channel = {}
            channel.update({"name": region["Name"]})
            samples = []
            for sample in self.config["Samples"]:
                if sample.get("Data", False):
                    # skip the data sample, it goes into the observations instead
                    continue

                if not configuration.region_contains_sample(region, sample):
                    # current region does not contain this sample, so skip it
                    continue

                sample_hist = histo.Histogram.from_config(
                    self.histogram_folder, region, sample, {}, modified=True
                )

                # yield of the sample
                current_sample = {}
                current_sample.update({"name": sample["Name"]})
                current_sample.update({"data": sample_hist.yields.tolist()})

                # collect all modifiers for the sample
                modifiers = []

                # gammas
                if not sample.get("DisableStaterror", False):
                    # staterror modifiers are added unless DisableStaterror is True
                    gammas = {}
                    gammas.update(
                        {"name": "staterror_" + region["Name"].replace(" ", "-")}
                    )
                    gammas.update({"type": "staterror"})
                    gammas.update({"data": sample_hist.stdev.tolist()})
                    modifiers.append(gammas)

                # modifiers can have region and sample dependence, which is checked
                # check if normfactors affect sample in region, add modifiers as needed
                nf_modifier_list = self.normfactor_modifiers(region, sample)
                modifiers += nf_modifier_list

                # check if systematics affect sample in region, add modifiers as needed
                sys_modifier_list = self.sys_modifiers(region, sample)
                modifiers += sys_modifier_list

                current_sample.update({"modifiers": modifiers})
                samples.append(current_sample)
            channel.update({"samples": samples})
            channels.append(channel)
        return channels

    def measurements(self) -> list[dict[str, Any]]:
        """Returns the measurements object for the workspace.

        Constructs the measurements, including POI setting and parameter bounds, initial
        values and whether they are set to constant. Only supports a single measurement
        so far.

        Returns:
            List[Dict[str, Any]]: measurements for ``pyhf`` workspace
        """
        measurements = []
        measurement = {}
        measurement.update({"name": self.config["General"]["Measurement"]})
        config_dict = {}

        # get the norm factor initial values / bounds / constant setting
        parameters_list = []
        for nf in self.config.get("NormFactors", []):
            nf_name = nf["Name"]  # every NormFactor has a name
            init = nf.get("Nominal", None)
            bounds = nf.get("Bounds", None)
            fixed = self._constant_parameter_setting(nf_name)
            if (init is None) and (fixed is not None):
                # if no initial value is specified, but a constant setting
                # is requested, set the initial value to the constant value
                init = fixed

            parameter = {"name": nf_name}
            if init is not None:
                parameter.update({"inits": [init]})
            if bounds is not None:
                parameter.update({"bounds": [bounds]})
            if fixed is not None:
                parameter.update({"fixed": True})

            parameters_list.append(parameter)

        for sys in self.config.get("Systematics", []):
            # when there are many more systematics than NormFactors, it would be more
            # efficient to loop over fixed parameters and exclude all NormFactor related
            # ones to set all the remaining ones to constant (which are systematics)
            sys_name = sys["Name"]  # every systematic has a name
            fixed = self._constant_parameter_setting(sys_name)
            if fixed is not None:
                parameter = {"name": sys_name}
                parameter.update({"inits": [fixed]})
                parameter.update({"fixed": True})
                parameters_list.append(parameter)

        parameters = {"parameters": parameters_list}
        config_dict.update(parameters)
        # POI defaults to "" (interpreted as "no POI" by pyhf) if not specified
        config_dict.update({"poi": self.config["General"].get("POI", "")})
        measurement.update({"config": config_dict})
        measurements.append(measurement)
        return measurements

    def observations(self) -> list[dict[str, Any]]:
        """Returns the observations object (with data yields) for the workspace.

        Returns:
            List[Dict[str, Any]]: observations for ``pyhf`` workspace
        """
        data_sample = self._data_sample()
        observations = []
        for region in self.config["Regions"]:
            observation = {}
            histo_yield = histo.Histogram.from_config(
                self.histogram_folder, region, data_sample, {}, modified=True
            ).yields.tolist()
            observation.update({"name": region["Name"]})
            observation.update({"data": histo_yield})
            observations.append(observation)
        return observations

    def build(self) -> dict[str, Any]:
        """Constructs a `HistFactory` workspace in ``pyhf`` format.

        Returns:
            Dict[str, Any]: ``pyhf``-compatible `HistFactory` workspace
        """
        ws: dict[str, Any] = {}  # the workspace

        # channels
        channels = self.channels()
        ws.update({"channels": channels})

        # measurements
        measurements = self.measurements()
        ws.update({"measurements": measurements})

        # build observations
        observations = self.observations()
        ws.update({"observations": observations})

        # workspace version
        ws.update({"version": "1.0.0"})

        return ws


def build(config: dict[str, Any], *, with_validation: bool = True) -> dict[str, Any]:
    """Returns a `HistFactory` workspace in ``pyhf`` format.

    Args:
        config (Dict[str, Any]): cabinetry configuration
        with_validation (bool, optional): validate workspace validity with pyhf,
            defaults to True

    Returns:
        Dict[str, Any]: ``pyhf``-compatible `HistFactory` workspace
    """
    log.info("building workspace")

    ws_builder = WorkspaceBuilder(config)
    ws = ws_builder.build()

    if with_validation:
        validate(ws)
    return ws


def validate(ws: dict[str, Any]) -> None:
    """Validates a workspace with ``pyhf``.

    Args:
        ws (Dict[str, Any]): the workspace to validate
    """
    pyhf.Workspace(ws)


def save(ws: dict[str, Any], file_path_string: str | pathlib.Path) -> None:
    """Serializes a workspace to a file.

    Args:
        ws (Dict[str, Any]): ``pyhf``-compatible `HistFactory` workspace
        file_path_string (Union[str, pathlib.Path]): path to the file to save the
            workspace in
    """
    file_path = pathlib.Path(file_path_string)
    log.debug(f"saving workspace to {file_path}")
    # create output directory if it does not exist yet
    file_path.parent.mkdir(parents=True, exist_ok=True)
    # save workspace to file
    file_path.write_text(json.dumps(ws, sort_keys=True, indent=4))


def load(file_path_string: str | pathlib.Path) -> dict[str, Any]:
    """Loads a workspace from a file.

    Args:
        file_path_string (Union[str, pathlib.Path]): path to the file to load the
            workspace from

    Returns:
        Dict[str, Any]: ``pyhf``-compatible `HistFactory` workspace
    """
    file_path = pathlib.Path(file_path_string)
    ws = json.loads(file_path.read_text())
    return ws


def _symmetrized_templates_and_norm(
    variation: histo.Histogram, reference: histo.Histogram
) -> tuple[list[float], list[float], float, float]:
    """Returns symmetrized normalized templates and normalization factors.

    The ``variation`` input will be normalized in the process.

    Args:
        variation (histo.Histogram): variation histogram to symmetrize
        reference (histo.Histogram): reference nominal histogram

    Returns:
        Tuple[List[float], List[float], float, float]: yields for variation, symmetrized
        variation, normalization factor for variation, and symmetrized version of the
        normalization factor
    """
    # if symmetrization is desired, should support different implementations
    # symmetrization according to "method 1" from issue #26:
    # first normalization, then symmetrization

    # normalize the variation to the same yield as nominal
    norm_effect_var = variation.normalize_to_yield(reference)
    norm_effect_sym = 2 - norm_effect_var
    # manually cast due to https://github.com/numpy/numpy/issues/27944
    histo_yield_var = cast(list[float], variation.yields.tolist())
    # need another histogram that corresponds to the symmetrized variation,
    # which is 2*nominal - variation
    histo_yield_sym = cast(
        list[float], (2 * reference.yields - variation.yields).tolist()
    )

    return histo_yield_var, histo_yield_sym, norm_effect_var, norm_effect_sym
