from setuptools import setup, find_packages

extras_require = {"contrib": ["matplotlib", "uproot", "scipy", "iminuit", "numexpr"]}
extras_require["test"] = sorted(
    set(
        extras_require["contrib"]
        + [
            "pytest",
            "pytest-cov>=2.5.1",
            "pydocstyle",
            "check-manifest",
            "flake8",
            "black;python_version>='3.6'",  # Black is Python3 only
        ]
    )
)
extras_require["develop"] = sorted(
    set(extras_require["test"] + ["pre-commit", "twine"])
)
extras_require["complete"] = sorted(set(sum(extras_require.values(), [])))


with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="cabinetry",
    version="0.0.4",
    author="cabinetry developers",
    description="design and steer profile likelihood fits",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="BSD 3-Clause",
    url="https://github.com/alexander-held/cabinetry",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: BSD License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.6",
    install_requires=["numpy", "pyyaml", "pyhf>=0.3.2", "iminuit"],
    extras_require=extras_require,
)
