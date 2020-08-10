from setuptools import find_packages, setup

extras_require = {"contrib": ["matplotlib", "uproot", "uproot4", "awkward1"]}
extras_require["test"] = sorted(
    set(
        extras_require["contrib"]
        + [
            "pytest",
            "pytest-cov>=2.5.1",
            "pydocstyle",
            "check-manifest",
            "flake8",
            "flake8-bugbear",
            "flake8-import-order",
            "flake8-print",
            "mypy",
            "typeguard",
            "black;python_version>='3.6'",  # Black is Python3 only
        ]
    )
)
extras_require["docs"] = sorted(
    set(
        [
            "sphinx",
            "sphinx-click",
            "sphinx-copybutton",
            "sphinx-jsonschema",
            "sphinx-rtd-theme",
        ]
    )
)

extras_require["develop"] = sorted(
    set(extras_require["test"] + extras_require["docs"] + ["pre-commit", "twine"])
)
extras_require["complete"] = sorted(set(sum(extras_require.values(), [])))


with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="cabinetry",
    version="0.1.1",
    author="Alexander Held",
    description="design and steer profile likelihood fits",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="BSD 3-Clause",
    url="https://github.com/alexander-held/cabinetry",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"cabinetry": ["py.typed", "schemas/config.json"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: BSD License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "pyyaml",
        "pyhf>=0.5.1",  # fixed parameter bookkeeping #989
        "iminuit>1.4.0",
        "boost_histogram",
        "jsonschema",
        "click",
    ],
    extras_require=extras_require,
)
