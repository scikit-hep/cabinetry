"""
need to save the histogram bins for visualization purposes still,
as well as cosmetics such as axis labels and region names
"""
import logging

from . import histo


log = logging.getLogger(__name__)


def _build_figure_name(region, is_prefit):
    """
    construct a name for the file a figure is saved as
    """
    figure_name = region.replace(" ", "-")
    if is_prefit:
        figure_name += "_" + "prefit"
    else:
        figure_name += "_" + "prefit"
    figure_name += ".pdf"
    return figure_name


def data_MC(config, histogram_folder, figure_folder, prefit=True, method="matplotlib"):
    """
    draw a data/MC histogram, control whether it is pre- or postfit with a flag
    """
    log.info("visualizing histogram")
    for region in config["Regions"]:
        histogram_dict_list = []
        for sample in config["Samples"]:
            for systematic in [{"Name": "nominal"}]:
                is_data = sample.get("Data", False)
                histogram_name = histo.build_name(sample, region, systematic)
                histogram = histo.load(histogram_folder, histogram_name, modified=True)
                histogram_dict_list.append(
                    {
                        "label": sample["Name"],
                        "isData": is_data,
                        "hist": histogram,
                        "variable": region["Variable"],
                    }
                )

        figure_name = _build_figure_name(region["Name"], prefit)

        if prefit:
            if method == "matplotlib":
                from cabinetry.contrib import histogram_drawing

                histogram_drawing.data_MC_matplotlib(
                    histogram_dict_list, figure_folder, figure_name
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError("only prefit implemented so far")
