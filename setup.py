from setuptools import setup, find_packages


with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="cabinetry",
    version="0.0.3",
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
)
