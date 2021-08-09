import matplotlib.pyplot as plt

from cabinetry.visualize import utils


def test__save_figure(tmp_path):
    fig = plt.figure()
    fname = tmp_path / "fig.pdf"
    utils._save_figure(fig, fname)
    assert fname.is_file()  # file was saved
    assert len(plt.get_fignums()) == 1  # figure is open

    fig = plt.figure()  # new figure to test closing
    utils._save_figure(fig, fname, close_figure=True)
    assert len(plt.get_fignums()) == 1  # previous figure still open
    plt.close("all")
