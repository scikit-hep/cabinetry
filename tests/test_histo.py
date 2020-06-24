import logging
from pathlib import Path

import numpy as np

from cabinetry import histo


def _example_hist():
    yields = np.asarray([1.0, 2.0])
    sumw2 = np.asarray([0.1, 0.2])
    bins = np.asarray([1, 2, 3])
    return histo.to_dict(yields, sumw2, bins)


def _example_hist_single_bin():
    yields = np.asarray([1.0])
    sumw2 = np.asarray([0.1])
    bins = np.asarray([1, 2])
    return histo.to_dict(yields, sumw2, bins)


def _example_hist_empty_bin():
    yields = np.asarray([1.0, 0.0])
    sumw2 = np.asarray([0.1, 0.2])
    bins = np.asarray([1, 2, 3])
    return histo.to_dict(yields, sumw2, bins)


def _example_hist_nan_sumw2_empty_bin():
    yields = np.asarray([1.0, 0.0])
    sumw2 = np.asarray([0.1, float("NaN")])
    bins = np.asarray([1, 2, 3])
    return histo.to_dict(yields, sumw2, bins)


def _example_hist_nan_sumw2_nonempty_bin():
    yields = np.asarray([1.0, 2.0])
    sumw2 = np.asarray([0.1, float("NaN")])
    bins = np.asarray([1, 2, 3])
    return histo.to_dict(yields, sumw2, bins)


def test_to_dict():
    yields = np.asarray([1.0, 2.0])
    sumw2 = np.asarray([0.1, 0.2])
    bins = np.asarray([1, 2, 3])
    assert histo.to_dict(yields, sumw2, bins) == {
        "yields": yields,
        "sumw2": sumw2,
        "bins": bins,
    }


def test_save(tmp_path):
    hist = _example_hist()
    histo.save(hist, tmp_path)
    np.testing.assert_equal(histo._load(tmp_path, modified=False), hist)


def test_save_new_dir(tmp_path):
    hist = _example_hist()
    # add a subdirectory that needs to be created for histogram saving
    fname = tmp_path / "subdir" / "file"
    histo.save(hist, fname)
    np.testing.assert_equal(histo._load(fname, modified=False), hist)


def test__load(tmp_path, caplog):
    hist = _example_hist()
    histo.save(hist, tmp_path)

    # try loading the original histogram
    np.testing.assert_equal(histo._load(tmp_path, modified=False), hist)

    # try loading the modified histograms, without success
    np.testing.assert_equal(histo._load(tmp_path, modified=True), hist)
    expected_warning = (
        "the modified histogram " + str(tmp_path) + "_modified.npz " + "does not exist"
    )
    assert expected_warning in [rec.message for rec in caplog.records]
    assert "loading the un-modified histogram instead!" in [
        rec.message for rec in caplog.records
    ]
    caplog.clear()


def test__load_modified(tmp_path, caplog):
    hist = _example_hist()
    histo_name = "histo"
    fname_modified = tmp_path / (histo_name + "_modified")
    fname_original = tmp_path / histo_name
    histo.save(hist, fname_modified)
    # load the modified histogram by specifying the original name, which should produce no warning
    np.testing.assert_equal(histo._load(fname_original, modified=True), hist)
    assert [rec.message for rec in caplog.records] == []
    caplog.clear()


def test_load_from_config(tmp_path):
    hist = _example_hist()
    expected_name = "Sample_Region_Systematic"
    fname = tmp_path / expected_name
    histo.save(hist, fname)
    sample = {"Name": "Sample"}
    region = {"Name": "Region"}
    systematic = {"Name": "Systematic"}
    loaded_histo, loaded_name = histo.load_from_config(
        tmp_path, sample, region, systematic, modified=False
    )
    np.testing.assert_equal(loaded_histo, hist)
    assert loaded_name == expected_name


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


def test_normalize_to_yield():
    hist = _example_hist_empty_bin()
    var_hist = _example_hist()
    modified_hist, factor = histo.normalize_to_yield(var_hist, hist)
    assert factor == 3.0
    np.testing.assert_equal(modified_hist["yields"], np.asarray([1 / 3, 2 / 3]))
