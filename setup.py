from setuptools import setup

extras_require = {"contrib": ["matplotlib", "uproot3", "uproot>=4.0"]}
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
            "typeguard>=2.12.1",  # click 8.0 compatibility
            "black",
            "pyhf[backends]",
        ]
    )
)
extras_require["docs"] = sorted(
    {
        "sphinx",
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

setup(
    extras_require=extras_require,
)
