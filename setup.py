from setuptools import setup

extras_require = {"contrib": ["uproot>=4.1.5"]}  # file writing bug-fixes
extras_require["test"] = sorted(
    set(
        extras_require["contrib"]
        + [
            "pytest",
            "pytest-cov>=2.6.1",  # no_cover support
            "pydocstyle",
            "check-manifest",
            "flake8",
            "flake8-bugbear",
            "flake8-import-order",
            "flake8-print",
            "mypy",
            "types-tabulate",
            "types-PyYAML",
            "typeguard>=2.13.0",  # typing.NamedTuple compatibility in Python 3.7
            "black",
        ]
    )
)
extras_require["pyhf_backends"] = ["pyhf[backends]"]
extras_require["docs"] = sorted(
    {
        "sphinx!=5.2.0.post0",  # broken due to version parsing in RTD theme
        "sphinx-click",
        "sphinx-copybutton",
        "sphinx-jsonschema",
        "sphinx-rtd-theme",
    }
)

extras_require["develop"] = sorted(
    set(extras_require["test"] + extras_require["docs"] + ["pre-commit", "twine"])
)
extras_require["complete"] = sorted(set(sum(extras_require.values(), [])))

setup(extras_require=extras_require)
