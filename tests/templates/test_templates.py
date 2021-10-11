import logging
import pathlib
from unittest import mock

from cabinetry import templates


@mock.patch("cabinetry.route.apply_to_all_templates")
@mock.patch("cabinetry.templates.builder._Builder")
def test_build(mock_builder, mock_apply):
    config = {"General": {"HistogramFolder": "path/", "InputPath": "file.root"}}
    method = "uproot"

    # no router
    templates.build(config, method=method)
    assert mock_builder.call_args_list == [
        ((pathlib.Path("path/"), "file.root", method), {})
    ]
    assert mock_apply.call_count == 1
    config_call, func_call = mock_apply.call_args[0]
    assert config_call == config
    assert func_call._extract_mock_name() == "_Builder()._create_histogram"
    assert mock_apply.call_args[1] == {"match_func": None}

    # including a router
    mock_router = mock.MagicMock()
    templates.build(config, method=method, router=mock_router)

    # verify wrapper was set
    assert (
        mock_router.template_builder_wrapper._extract_mock_name()
        == "_Builder()._wrap_custom_template_builder"
    )

    assert mock_apply.call_count == 2  # 1 from before
    config_call, func_call = mock_apply.call_args[0]
    assert config_call == config
    assert func_call._extract_mock_name() == "_Builder()._create_histogram"
    assert mock_apply.call_args[1] == {
        "match_func": mock_router._find_template_builder_match
    }


@mock.patch("cabinetry.route.apply_to_all_templates")
@mock.patch("cabinetry.templates.collector._collector", return_value="func")
def test_collect(mock_collector, mock_apply, caplog):
    caplog.set_level(logging.DEBUG)
    config = {
        "General": {
            "HistogramFolder": "path/",
            "InputPath": "f.root:{VariationPath}",
            "VariationPath": "nominal",
        }
    }
    method = "uproot"

    templates.collect(config, method=method)
    assert mock_collector.call_args_list == [
        ((pathlib.Path("path/"), "f.root:{VariationPath}", "nominal", method), {})
    ]
    assert mock_apply.call_args_list == [((config, "func"), {})]

    caplog.clear()

    # no VariationPath in general settings
    config = {
        "General": {"HistogramFolder": "path/", "InputPath": "f.root:{VariationPath}"}
    }
    templates.collect(config, method=method)
    assert 'no VariationPath specified in general settings, defaulting to ""' in [
        rec.message for rec in caplog.records
    ]
    assert mock_collector.call_args == (
        (pathlib.Path("path/"), "f.root:{VariationPath}", "", method),
        {},
    )
    caplog.set_level(logging.DEBUG)


@mock.patch("cabinetry.route.apply_to_all_templates")
@mock.patch("cabinetry.templates.postprocessor._postprocessor", return_value="func")
def test_run(mock_postprocessor, mock_apply):
    config = {"General": {"HistogramFolder": "path/"}}

    templates.postprocess(config)
    assert mock_postprocessor.call_args_list == [((pathlib.Path("path/"),), {})]
    assert mock_apply.call_args_list == [((config, "func"), {})]
