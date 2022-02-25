from collections import defaultdict
import json
import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyhf
from pyhf.contrib.utils import download


def modifier_grid(model: pyhf.pdf.Model) -> None:
    modifier_dict = defaultdict(list)
    for channel in model.spec["channels"]:
        for sample in channel["samples"]:
            for modifier in sample["modifiers"]:
                modifier_dict[
                    (channel["name"], sample["name"], modifier["name"])
                ].append(modifier["type"])

    NUM_CHANNELS = len(model.config.channels)
    NUM_SAMPLES = len(model.config.samples)
    NUM_PARAMS = len(model.config.par_order)

    NUM_CATEGORIES = 7  # no modifier + 6 modifier categories

    # build 2d grid for each sample
    grids = [np.zeros(shape=(NUM_CHANNELS, NUM_PARAMS)) for _ in model.config.samples]
    for i, sample in enumerate(model.config.samples):
        for j, chan in enumerate(model.config.channels):
            for k, par in enumerate(model.config.par_order):
                modifiers = modifier_dict[(chan, sample, par)]
                # define categories
                if modifiers == []:
                    val = 0
                elif modifiers == ["normsys"]:
                    val = 1
                elif modifiers == ["histosys"]:
                    val = 2
                elif sorted(modifiers) == ["histosys", "normsys"]:
                    val = 3
                elif modifiers == ["staterror"]:
                    val = 4
                elif modifiers == ["lumi"]:
                    val = 5
                elif modifiers == ["normfactor"]:
                    val = 6
                else:
                    raise NotImplementedError(chan, par, modifiers)
                grids[i][j, k] = val

    # colors of categories
    cmap = mpl.colors.ListedColormap(
        [
            "white",
            "lightskyblue",
            "gold",
            "olivedrab",
            "rebeccapurple",
            "lightslategray",
            "firebrick",
        ]
    )
    bounds = np.arange(-0.5, NUM_CATEGORIES + 0.5)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # very rough heuristics for figure size, need to be tuned
    fig_width = 3 + NUM_PARAMS / 3.8
    fig_height = 3 + (NUM_CHANNELS * NUM_SAMPLES) / 3.2
    fig, ax = plt.subplots(
        NUM_SAMPLES,
        sharex=True,
        constrained_layout=True,
        figsize=(fig_width, fig_height),
    )

    for i, sample in enumerate(model.config.samples):
        im = ax[i].imshow(grids[i], cmap=cmap, norm=norm)
        ax[i].set_title(sample)

        ax[i].set_xticks(np.arange(NUM_PARAMS))
        ax[i].set_yticks(np.arange(NUM_CHANNELS))
        ax[i].set_xticklabels(model.config.par_order)
        ax[i].set_yticklabels(model.config.channels)
        for tick in ax[i].get_xticklabels():
            tick.set_rotation(45)
            tick.set_horizontalalignment("right")

    # labels for categories
    ticks = {
        0: "none",
        1: "normsys",
        2: "histosys",
        3: "normsys + histosys",
        4: "staterror",
        5: "lumi",
        6: "normfactor",
    }
    formatter = plt.FuncFormatter(lambda val, _: ticks[val])
    _ = fig.colorbar(
        im, ax=ax.ravel().tolist(), ticks=np.arange(NUM_CATEGORIES), format=formatter
    )
    plt.savefig("modifier_grid.png")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ws = json.loads(pathlib.Path(sys.argv[-1]).read_text())
    else:
        download(
            "https://www.hepdata.net/record/resource/1935437?view=true",
            "bottom-squarks",
        )
        ws = pyhf.Workspace(json.load(open("bottom-squarks/RegionC/BkgOnly.json")))
        patchset = pyhf.PatchSet(
            json.load(open("bottom-squarks/RegionC/patchset.json"))
        )
        ws = patchset.apply(ws, "sbottom_600_280_150")

    model = pyhf.Workspace(ws).model()
    modifier_grid(model)
