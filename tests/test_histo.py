import logging

import pytest

from cabinetry import histo


def _example_hist():
    yields = [1, 2]
    sumw2 = [0.1, 0.2]
    bins = [1, 2]
    return histo.to_dict(yields, sumw2, bins)


def _example_hist_single_bin():
    yields = [1]
    sumw2 = [0.1]
    bins = [1]
    return histo.to_dict(yields, sumw2, bins)


def _example_hist_empty_bin():
    yields = [1, 0]
    sumw2 = [0.1, 0.2]
    bins = [1, 2]
    return histo.to_dict(yields, sumw2, bins)


def _example_hist_nan_sumw2_empty_bin():
    yields = [1, 0]
    sumw2 = [0.1, float("NaN")]
    bins = [1, 2]
    return histo.to_dict(yields, sumw2, bins)


def _example_hist_nan_sumw2_nonempty_bin():
    yields = [1, 2]
    sumw2 = [0.1, float("NaN")]
    bins = [1, 2]
    return histo.to_dict(yields, sumw2, bins)


def test_to_dict():
    yields = [1, 2]
    sumw2 = [0.1, 0.2]
    bins = [1, 2]
    assert histo.to_dict(yields, sumw2, bins) == {
        "yields": yields,
        "sumw2": sumw2,
        "bins": bins,
    }


def test_save(tmpdir):
    pass


def test_load():
    pass


def test_build_name():
    sample = {"Name": "Sample"}
    region = {"Name": "Region"}
    systematic = {"Name": "Systematic"}
    assert histo.build_name(sample, region, systematic) == "Sample_Region_Systematic"

    sample = {"Name": "Sample 1"}
    region = {"Name": "Region 1"}
    systematic = {"Name": "Systematic 1"}
    assert (
        histo.build_name(sample, region, systematic) == "Sample-1_Region-1_Systematic-1"
    )


def test_validate(caplog):
    caplog.set_level(logging.DEBUG)
    name = "test_histo"

    # should cause no warnings
    histo.validate(_example_hist(), name)
    histo.validate(_example_hist_single_bin(), name)

    # check for empty bin warning
    histo.validate(_example_hist_empty_bin(), name)
    assert "test_histo has empty bins: [1]" in [rec.message for rec in caplog.records]
    caplog.clear()

    # check for ill-defined stat uncertainty warning
    histo.validate(_example_hist_nan_sumw2_empty_bin(), name)
    assert "test_histo has empty bins: [1]" in [rec.message for rec in caplog.records]
    assert "test_histo has bins with ill-defined stat. unc.: [1]" in [
        rec.message for rec in caplog.records
    ]
    caplog.clear()

    # check for ill-defined stat uncertainty warning with non-zero bin
    histo.validate(_example_hist_nan_sumw2_nonempty_bin(), name)
    assert "test_histo has bins with ill-defined stat. unc.: [1]" in [
        rec.message for rec in caplog.records
    ]
    assert "test_histo has non-empty bins with ill-defined stat. unc.: [1]" in [
        rec.message for rec in caplog.records
    ]
    caplog.clear()
