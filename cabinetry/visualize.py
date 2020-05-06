"""
need to save the histogram bins for visualization purposes still,
as well as cosmetics such as axis labels and region names
"""

from . import histogram_wrapper


def _data_MC_matplotlib(histogram_dict_list, figure_folder, figure_name):
    """
    draw a data/MC histogram with matplotlib
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    mc_histograms = []
    mc_labels = []
    for h in histogram_dict_list:
        if h["isData"]:
            data_histogram = h["hist"]
            data_label = h["label"]
        else:
            mc_histograms.append(h["hist"])
            mc_labels.append(h["label"])

    # get the highest single bin from the sum of MC
    y_max = np.max(np.sum([h["hist"] for h in histogram_dict_list if not h["isData"]], axis=0))

    # if data is higher in any bin, the maximum y axis range should take that into account
    y_max = max(y_max, np.max([h["hist"] for h in histogram_dict_list if h["isData"]]))

    # plot MC stacked together
    total_yield = np.zeros_like(mc_histograms[0])
    x_dummy = np.arange(0, len(mc_histograms[0]))
    for i_sample in range(len(mc_histograms)):
        plt.bar(x_dummy, mc_histograms[i_sample], width=1.0, bottom=total_yield, label=mc_labels[i_sample])
        total_yield += mc_histograms[i_sample]

    # plot data
    plt.plot(data_histogram, "x", c="k", label=data_label)

    plt.legend()
    plt.ylim([0, y_max*1.1])  # 10% headroom
    plt.plot()

    if not os.path.exists(figure_folder):
        os.mkdir(figure_folder)
    print("- saving", figure_name, "to", figure_folder)
    plt.savefig(figure_folder + figure_name)


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
    print("# visualizing histogram")
    for region in config["Regions"]:
        histogram_dict_list = []
        for sample in config["Samples"]:
            for systematic in [{"Name": "nominal"}]:
                is_data = sample.get("Data", False)
                histogram_name = histogram_wrapper._build_histogram_name(sample, region, systematic)
                histogram = histogram_wrapper.load_histogram(histogram_folder, histogram_name)
                histogram_dict_list.append({"label": sample["Name"], "isData": is_data, "hist": histogram})

        figure_name = _build_figure_name(region["Name"], prefit)

        if prefit:
            if method == "matplotlib":
                _data_MC_matplotlib(histogram_dict_list, figure_folder, figure_name)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError("only prefit implemented so far")
