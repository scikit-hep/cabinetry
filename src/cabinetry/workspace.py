import json
import logging
import pathlib
from typing import Any, Dict, List, Optional, Union

import pyhf

from . import configuration
from . import histo


log = logging.getLogger(__name__)


class WorkspaceBuilder:
    """Collects functionality to build a ``pyhf`` workspace."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Creates a workspace corresponding to a cabinetry configuration.

        Args:
            config (Dict[str, Any]): ``cabinetry`` configuration
        """
        self.config = config
        self.histogram_folder = pathlib.Path(config["General"]["HistogramFolder"])

    def _get_data_sample(self) -> Dict[str, Any]:
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

    def _get_constant_parameter_setting(self, par_name: str) -> Optional[float]:
        """Determines whether a parameter should be set to constant, and to which value.

        For a given parameter, determines if it is supposed to be set to constant. If
        not, returns None, otherwise returns the value it should be fixed to. This only
        looks for the first occurrence of the parameter in the list.

        Args:
            par_name (str): name of parameter to check

        Returns:
            Optional[float]: returns None if parameter is not supposed to be held
            constant, or the value it has to be fixed to
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

    def get_yield_for_sample(
        self,
        region: Dict[str, Any],
        sample: Dict[str, Any],
        systematic: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        """Returns the yield for a specific sample.

        Args:
            region (Dict[str, Any]): specific region to use
            sample (Dict[str, Any]): specific sample to use
            systematic (Optional[Dict[str, Any]], optional): specific systematic
                variation to use, defaults to None -> {"Name": "Nominal"}

        Returns:
            List[float]: yields per bin for the sample
        """
        if systematic is None:
            systematic = {"Name": "Nominal"}

        histogram = histo.Histogram.from_config(
            self.histogram_folder, region, sample, systematic, modified=True
        )
        histo_yield = histogram.yields.tolist()
        return histo_yield

    def get_unc_for_sample(
        self,
        region: Dict[str, Any],
        sample: Dict[str, Any],
        systematic: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        """Returns the MC stat. uncertainty for a specific sample.

        Args:
            region (Dict[str, Any]): specific region to use
            sample (Dict[str, Any]): specific sample to use
            systematic (Optional[Dict[str, Any]], optional): specific systematic
                variation to use, defaults to None -> {"Name": "Nominal"}

        Returns:
            List[float]: statistical uncertainty of yield per bin for the sample
        """
        if systematic is None:
            systematic = {"Name": "Nominal"}

        histogram = histo.Histogram.from_config(
            self.histogram_folder, region, sample, systematic, modified=True
        )
        histo_stdev = histogram.stdev.tolist()
        return histo_stdev

    def get_NF_modifiers(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Returns the list of NormFactor modifiers acting on a sample.

        Args:
            sample (Dict[str, Any]): specific sample to get NormFactor modifiers for

        Returns:
            List[Dict[str, Any]]: NormFactor modifiers for sample
        """
        modifiers = []
        for NormFactor in self.config["NormFactors"]:
            if configuration.sample_affected_by_modifier(sample, NormFactor):
                log.debug(
                    f"adding NormFactor {NormFactor['Name']} to sample {sample['Name']}"
                )
                modifiers.append(
                    {"data": None, "name": NormFactor["Name"], "type": "normfactor"}
                )
        return modifiers

    @staticmethod
    def get_Normalization_modifier(systematic: Dict[str, Any]) -> Dict[str, Any]:
        """Returns a normalization modifier (OverallSys in `HistFactory`).

        Args:
            systematic (Dict[str, Any]): systematic for which modifier is constructed

        Returns:
            Dict[str, Any]: single `normsys` modifier for ``pyhf`` workspace
        """
        modifier = {}
        modifier.update({"name": systematic["Name"]})
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

    def get_NormPlusShape_modifiers(
        self,
        region: Dict[str, Any],
        sample: Dict[str, Any],
        systematic: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Returns modifiers for a correlated shape + normalization effect.

        For a variation including a correlated shape + normalization effect, this
        provides the `histosys` and `normsys` modifiers for ``pyhf`` (in `HistFactory`
        language, this corresponds to a `HistoSys` and an `OverallSys`). Symmetrization
        could happen either at this stage (this is the case currently), or somewhere
        earlier, such as during template postprocessing.

        Args:
            region (Dict[str, Any]): region the systematic variation acts in
            sample (Dict[str, Any]): sample the systematic variation acts on
            systematic (Dict[str, Any]): the systematic variation under consideration

        Returns:
            List[Dict[str, Any]]: a list with a ``pyhf`` `normsys` modifier and a
            `histosys` modifier
        """
        # load the systematic variation histogram
        histogram_up = histo.Histogram.from_config(
            self.histogram_folder,
            region,
            sample,
            systematic,
            modified=True,
            template="Up",
        )

        # also need the nominal histogram
        histogram_nominal = histo.Histogram.from_config(
            self.histogram_folder, region, sample, {"Name": "Nominal"}, modified=True
        )

        if systematic.get("Down", {}).get("Symmetrize", False):
            # add support for two-sided variations that do not require symmetrization
            # if symmetrization is desired, should support different implementations

            # symmetrization according to "method 1" from issue #26:
            # first normalization, then symmetrization

            # normalize the variation to the same yield as nominal
            norm_effect = histogram_up.normalize_to_yield(histogram_nominal)
            norm_effect_up = norm_effect
            norm_effect_down = 2 - norm_effect
            histo_yield_up = histogram_up.yields.tolist()
            log.debug(
                f"normalization impact of systematic {systematic['Name']} on sample "
                f"{sample['Name']} in region {region['Name']} is {norm_effect:.3f}"
            )
            # need another histogram that corresponds to the "down" variation, which is
            # 2*nominal - up
            histo_yield_down = (
                2 * histogram_nominal.yields - histogram_up.yields
            ).tolist()
        else:
            histogram_down = histo.Histogram.from_config(
                self.histogram_folder,
                region,
                sample,
                systematic,
                modified=True,
                template="Down",
            )
            norm_effect_up = sum(histogram_up.yields) / sum(histogram_nominal.yields)
            norm_effect_down = sum(histogram_down.yields) / sum(
                histogram_nominal.yields
            )
            # normalize templates to same yield as nominal
            histo_yield_up = list(histogram_up.yields / norm_effect_up)
            histo_yield_down = list(histogram_down.yields / norm_effect_down)

        # add the normsys
        modifiers = []
        norm_modifier = {}
        norm_modifier.update({"name": systematic["Name"]})
        norm_modifier.update({"type": "normsys"})
        norm_modifier.update({"data": {"hi": norm_effect_up, "lo": norm_effect_down}})
        modifiers.append(norm_modifier)

        # add the shape part in a histosys
        shape_modifier = {}
        shape_modifier.update({"name": systematic["Name"]})
        shape_modifier.update({"type": "histosys"})
        shape_modifier.update(
            {"data": {"hi_data": histo_yield_up, "lo_data": histo_yield_down}}
        )
        modifiers.append(shape_modifier)
        return modifiers

    def get_sys_modifiers(
        self, region: Dict[str, Any], sample: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Returns the list of all systematic modifiers acting on a sample.

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
            if configuration.sample_affected_by_modifier(sample, systematic):
                if systematic["Type"] == "Normalization":
                    # OverallSys (norm uncertainty with Gaussian constraint)
                    log.debug(
                        f"adding OverallSys {systematic['Name']} to sample"
                        f" {sample['Name']}",
                    )
                    modifiers.append(self.get_Normalization_modifier(systematic))
                elif systematic["Type"] == "NormPlusShape":
                    # two modifiers are needed - an OverallSys for the norm effect,
                    # and a HistoSys for the shape variation
                    log.debug(
                        f"adding OverallSys and HistoSys {systematic['Name']} to sample"
                        f" {sample['Name']}",
                    )
                    modifiers += self.get_NormPlusShape_modifiers(
                        region, sample, systematic
                    )
                else:
                    raise NotImplementedError(
                        "not supporting other systematic types yet"
                    )
        return modifiers

    def get_channels(self) -> List[Dict[str, Any]]:
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

                # yield of the samples
                histo_yield = self.get_yield_for_sample(region, sample)
                current_sample = {}
                current_sample.update({"name": sample["Name"]})
                current_sample.update({"data": histo_yield})

                # collect all modifiers for the sample
                modifiers = []

                # gammas
                stat_unc = self.get_unc_for_sample(region, sample)
                gammas = {}
                gammas.update({"name": "staterror_" + region["Name"].replace(" ", "-")})
                gammas.update({"type": "staterror"})
                gammas.update({"data": stat_unc})
                modifiers.append(gammas)

                # check if normfactor affect the sample and add modifiers as needed
                nf_modifier_list = self.get_NF_modifiers(sample)
                modifiers += nf_modifier_list

                # check if systematic uncertainties affect the samples, add modifiers
                sys_modifier_list = self.get_sys_modifiers(region, sample)
                modifiers += sys_modifier_list

                current_sample.update({"modifiers": modifiers})
                samples.append(current_sample)
            channel.update({"samples": samples})
            channels.append(channel)
        return channels

    def get_measurements(self) -> List[Dict[str, Any]]:
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
            fixed = self._get_constant_parameter_setting(nf_name)
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
            fixed = self._get_constant_parameter_setting(sys_name)
            if fixed is not None:
                parameter = {"name": sys_name}
                parameter.update({"inits": [fixed]})
                parameter.update({"fixed": True})
                parameters_list.append(parameter)

        parameters = {"parameters": parameters_list}
        config_dict.update(parameters)
        config_dict.update({"poi": self.config["General"]["POI"]})
        measurement.update({"config": config_dict})
        measurements.append(measurement)
        return measurements

    def get_observations(self) -> List[Dict[str, Any]]:
        """Returns the observations (data yields) for the workspace.

        Returns:
            List[Dict[str, Any]]: observations for ``pyhf`` workspace
        """
        data_sample = self._get_data_sample()
        observations = []
        for region in self.config["Regions"]:
            observation = {}
            histo_yield = self.get_yield_for_sample(region, data_sample)
            observation.update({"name": region["Name"]})
            observation.update({"data": histo_yield})
            observations.append(observation)
        return observations

    def build(self) -> Dict[str, Any]:
        """Constructs a `HistFactory` workspace in ``pyhf`` format.

        Returns:
            Dict[str, Any]: ``pyhf``-compatible `HistFactory` workspace
        """
        ws: Dict[str, Any] = {}  # the workspace

        # channels
        channels = self.get_channels()
        ws.update({"channels": channels})

        # measurements
        measurements = self.get_measurements()
        ws.update({"measurements": measurements})

        # build observations
        observations = self.get_observations()
        ws.update({"observations": observations})

        # workspace version
        ws.update({"version": "1.0.0"})

        return ws


def build(config: Dict[str, Any], with_validation: bool = True) -> Dict[str, Any]:
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


def validate(ws: Dict[str, Any]) -> None:
    """Validates a workspace with ``pyhf``.

    Args:
        ws (Dict[str, Any]): the workspace to validate
    """
    pyhf.Workspace(ws)


def save(ws: Dict[str, Any], file_path_string: Union[str, pathlib.Path]) -> None:
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


def load(file_path_string: Union[str, pathlib.Path]) -> Dict[str, Any]:
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
