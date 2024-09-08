from setuptools import find_packages, setup

setup(
    name="pyadps",
    version="0.0.2",
    install_requires=[
        "streamlit>=1.36.0",
        "numpy>=1.26.4",
        "matplotlib>=3.8.4",
        "scipy>=1.14.0",
        "wmm2020>=1.1.1",
        "cmake>=3.30.2",
        "pandas>=2.2.2",
        "netCDF4>=1.7.1",
        "plotly>=5.22.0",
        "plotly-resampler>=0.10.0",
        "meson>=1.4.1",
    ],
    packages=find_packages(),
)
