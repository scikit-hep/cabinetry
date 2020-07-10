import logging
from pathlib import Path

import numpy as np
import pytest

from cabinetry import histo


class ExampleHistograms:
    """a collection of different kinds of histograms
    """

    @staticmethod
    def normal():
        yields = np.asarray([1.0, 2.0])
        sumw2 = np.asarray([0.1, 0.2])
        bins = np.asarray([1, 2, 3])
        return yields, sumw2, bins

    @staticmethod
    def single_bin():
        yields = np.asarray([1.0])
        sumw2 = np.asarray([0.1])
        bins = np.asarray([1, 2])
        return yields, sumw2, bins

    @staticmethod
    def empty_bin():
        yields = np.asarray([1.0, 0.0])
        sumw2 = np.asarray([0.1, 0.2])
        bins = np.asarray([1, 2, 3])
        return yields, sumw2, bins

    @staticmethod
    def nan_sumw2_empty_bin():
        yields = np.asarray([1.0, 0.0])
        sumw2 = np.asarray([0.1, float("NaN")])
        bins = np.asarray([1, 2, 3])
        return yields, sumw2, bins

    @staticmethod
    def nan_sumw2_nonempty_bin():
        yields = np.asarray([1.0, 2.0])
        sumw2 = np.asarray([0.1, float("NaN")])
        bins = np.asarray([1, 2, 3])
        return yields, sumw2, bins


@pytest.fixture
def example_histograms():
    return ExampleHistograms


class HistogramHelpers:
    @staticmethod
    def assert_equal(h1, h2):
        assert np.allclose(h1.yields, h2.yields)
        assert np.allclose(h1.sumw2, h2.sumw2)
        assert np.allclose(h1.bins, h2.bins)


@pytest.fixture
def histogram_helpers():
    return HistogramHelpers


def test_Histogram(example_histograms):
    yields, sumw2, bins = example_histograms.normal()
    h = histo.Histogram(yields, sumw2, bins)
    assert np.allclose(h.yields, yields)
    assert np.allclose(h.sumw2, sumw2)
    assert np.allclose(h.bins, bins)


def test_Histogram_from_path(tmp_path, caplog, example_histograms, histogram_helpers):
    h_ref = histo.Histogram(*example_histograms.normal())
    h_ref.save(tmp_path)

    # load the original histogram
    h_from_path = histo.Histogram.from_path(tmp_path, modified=False)
    histogram_helpers.assert_equal(h_ref, h_from_path)

    # try loading a modified one, without success since it does not exist
    h_from_path_modified = histo.Histogram.from_path(tmp_path, modified=True)
    expected_warning = (
        "the modified histogram " + str(tmp_path) + "_modified.npz " + "does not exist"
    )
    assert expected_warning in [rec.message for rec in caplog.records]
    assert "loading the un-modified histogram instead!" in [
        rec.message for rec in caplog.records
    ]
    caplog.clear()

    # successfully load a modified histogram
    histo_name = "histo"
    fname_modified = tmp_path / (histo_name + "_modified")
    fname_original = tmp_path / histo_name
    h_ref.save(fname_modified)
    # load the modified histogram by specifying the original name, which should produce no warning
    h_from_path_modified = histo.Histogram.from_path(fname_original, modified=True)
    histogram_helpers.assert_equal(h_ref, h_from_path_modified)
    assert [rec.message for rec in caplog.records] == []
    caplog.clear()


def test_Histogram_from_config(tmp_path, example_histograms, histogram_helpers):
    h_ref = histo.Histogram(*example_histograms.normal())
    histo_path = tmp_path / "region_sample_nominal.npz"
    h_ref.save(histo_path)

    region = {"Name": "region"}
    sample = {"Name": "sample"}
    systematic = {"Name": "nominal"}
    h_from_path = histo.Histogram.from_config(
        tmp_path, region, sample, systematic, modified=False
    )
    histogram_helpers.assert_equal(h_ref, h_from_path)


def test_Histogram_save(tmp_path, example_histograms, histogram_helpers):
    hist = histo.Histogram(*example_histograms.normal())
    hist.save(tmp_path)
    hist_loaded = histo.Histogram.from_path(tmp_path, modified=False)
    histogram_helpers.assert_equal(hist, hist_loaded)
    # add a subdirectory that needs to be created for histogram saving
    fname = tmp_path / "subdir" / "file"
    hist.save(fname)
    hist_loaded = histo.Histogram.from_path(fname, modified=False)
    histogram_helpers.assert_equal(hist, hist_loaded)


def test_Histogram_validate(caplog, example_histograms):
    caplog.set_level(logging.DEBUG)
    name = "test_histo"

    # should cause no warnings
    histo.Histogram(*example_histograms.normal()).validate(name)
    histo.Histogram(*example_histograms.single_bin()).validate(name)
    assert [rec.message for rec in caplog.records] == []
    caplog.clear()

    # check for empty bin warning
    histo.Histogram(*example_histograms.empty_bin()).validate(name)
    assert "test_histo has empty bins: [1]" in [rec.message for rec in caplog.records]
    caplog.clear()

    # check for ill-defined stat uncertainty warning
    histo.Histogram(*example_histograms.nan_sumw2_empty_bin()).validate(name)
    assert "test_histo has empty bins: [1]" in [rec.message for rec in caplog.records]
    assert "test_histo has bins with ill-defined stat. unc.: [1]" in [
        rec.message for rec in caplog.records
    ]
    caplog.clear()

    # check for ill-defined stat uncertainty warning with non-zero bin
    histo.Histogram(*example_histograms.nan_sumw2_nonempty_bin()).validate(name)
    assert "test_histo has bins with ill-defined stat. unc.: [1]" in [
        rec.message for rec in caplog.records
    ]
    assert "test_histo has non-empty bins with ill-defined stat. unc.: [1]" in [
        rec.message for rec in caplog.records
    ]
    caplog.clear()


def test_Histogram_normalize_to_yield(example_histograms):
    hist = histo.Histogram(*example_histograms.empty_bin())
    var_hist = histo.Histogram(*example_histograms.normal())
    factor = var_hist.normalize_to_yield(hist)
    assert factor == 3.0
    np.testing.assert_equal(var_hist.yields, np.asarray([1 / 3, 2 / 3]))


def test_build_name():
    region = {"Name": "Region"}
    sample = {"Name": "Sample"}
    systematic = {"Name": "Systematic"}
    assert histo.build_name(region, sample, systematic) == "Region_Sample_Systematic"

    region = {"Name": "Region 1"}
    sample = {"Name": "Sample 1"}
    systematic = {"Name": "Systematic 1"}
    assert (
        histo.build_name(region, sample, systematic) == "Region-1_Sample-1_Systematic-1"
    )
