from cabinetry import visualize


def test__build_figure_name():
    assert visualize._build_figure_name("SR", True) == "SR_prefit.pdf"
    assert visualize._build_figure_name("SR", False) == "SR_postfit.pdf"
    assert visualize._build_figure_name("SR 1", True) == "SR-1_prefit.pdf"
    assert visualize._build_figure_name("SR 1", False) == "SR-1_postfit.pdf"
