import logging
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


log = logging.getLogger(__name__)


def _total_yield_uncertainty(sumw2_list):
    """calculate the absolute statistical uncertainty of a stack of MC
    via sum in quadrature

    Args:
        sumw2_list (list): list of absolute stat. uncertainty per sample

    Returns:
        np.array: absolute stat. uncertainty of stack of samples
    """
    tot_unc = np.sqrt(np.sum(np.power(sumw2_list, 2), axis=0))
    return tot_unc


def data_MC_matplotlib(histogram_dict_list, figure_path):
    """draw a data/MC histogram with matplotlib

    Args:
        histogram_dict_list (list[dict]): list of samples (with info stored in one dict per sample)
        figure_path (pathlib.Path): path where figure should be saved
    """
    mc_histograms_yields = []
    mc_histograms_sumw2 = []
    mc_labels = []
    for h in histogram_dict_list:
        if h["isData"]:
            data_histogram_yields = h["hist"]["yields"]
            data_histogram_sumw2 = h["hist"]["sumw2"]
            data_label = h["label"]
        else:
            mc_histograms_yields.append(h["hist"]["yields"])
            mc_histograms_sumw2.append(h["hist"]["sumw2"])
            mc_labels.append(h["label"])

    # get the highest single bin from the sum of MC
    y_max = np.max(
        np.sum(
            [h["hist"]["yields"] for h in histogram_dict_list if not h["isData"]],
            axis=0,
        )
    )

    # if data is higher in any bin, the maximum y axis range should take that into account
    y_max = max(
        y_max, np.max([h["hist"]["yields"] for h in histogram_dict_list if h["isData"]])
    )

    mpl.style.use("seaborn-colorblind")

    # plot MC stacked together
    total_yield = np.zeros_like(mc_histograms_yields[0])
    bins = histogram_dict_list[0]["hist"]["bins"]
    bin_right_edges = bins[1:]
    bin_left_edges = bins[:-1]
    bin_width = bin_right_edges - bin_left_edges
    bin_centers = 0.5 * (bin_left_edges + bin_right_edges)
    for i_sample, mc_sample_yield in enumerate(mc_histograms_yields):
        plt.bar(
            bin_centers,
            mc_sample_yield,
            width=bin_width,
            bottom=total_yield,
            label=mc_labels[i_sample],
        )
        total_yield += mc_sample_yield

    # add total MC uncertainty
    mc_stack_unc = _total_yield_uncertainty(mc_histograms_sumw2)
    plt.bar(
        bin_centers,
        2 * mc_stack_unc,
        width=bin_width,
        bottom=total_yield - mc_stack_unc,
        label="Stat. uncertainty",
        fill=False,
        linewidth=0,
        edgecolor="gray",
        hatch=3 * "/",
    )

    # plot data
    plt.errorbar(
        bin_centers,
        data_histogram_yields,
        yerr=data_histogram_sumw2,
        fmt="o",
        c="k",
        label=data_label,
    )

    plt.legend(frameon=False)
    plt.xlabel(histogram_dict_list[0]["variable"])
    plt.ylabel("events")
    plt.xlim(bin_left_edges[0], bin_right_edges[-1])
    plt.ylim([0, y_max * 1.1])  # 10% headroom
    plt.plot()

    if not os.path.exists(figure_path.parent):
        os.mkdir(figure_path.parent)
    log.debug("saving figure as %s", figure_path)
    plt.savefig(figure_path)
    plt.clf()
