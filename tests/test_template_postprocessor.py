import numpy as np

from cabinetry import template_postprocessor


def test__fix_stat_unc():
    histo = {"sumw2": [float("nan"), 0.2]}
    name = "test_histo"
    fixed_sumw2 = [0.0, 0.2]
    assert np.allclose(
        template_postprocessor._fix_stat_unc(histo, name)["sumw2"], fixed_sumw2
    )


def test__fix_stat_unc_no_fix():
    histo = {"sumw2": [0.1, 0.2]}
    name = "test_histo"
    fixed_sumw2 = [0.1, 0.2]
    assert np.allclose(
        template_postprocessor._fix_stat_unc(histo, name)["sumw2"], fixed_sumw2
    )


def test_adjust_histogram():
    histo = {"sumw2": [float("nan"), 0.2]}
    name = "test_histo"
    fixed_sumw2 = [0.0, 0.2]
    assert np.allclose(
        template_postprocessor.adjust_histogram(histo, name)["sumw2"], fixed_sumw2
    )


def test_run(tmp_path):
    # needs to be expanded into a proper test
    config = {"Samples": {}, "Regions": {}, "Systematics": {}}
    template_postprocessor.run(config, tmp_path)
