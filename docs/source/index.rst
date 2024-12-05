.. pyadps documentation master file, created by
   sphinx-quickstart on Wed Dec  4 18:12:19 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyadps's documentation!
==================================

**pyadps** is a Python package for processing moored Acoustic Doppler Current Profiler (ADCP) 
data. It provides various functionalities such as data reading, quality control tests, NetCDF 
file creation, and visualization.

This software offers both a graphical interface (`Streamlit`) for those new to Python and 
direct Python package access for experienced users. Please note that `pyadps` is primarily 
designed for Teledyne RDI workhorse ADCPs. Other company's ADCP files are not compatible, 
and while some other RDI models may work, they might require additional considerations.

.. note::

   This project is under active development.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   streamlit
   example.ipynb


**Stay Connected**
------------------
- **GitHub Repository**: Explore the source code and contribute at `pyadps on GitHub <https://github.com/p-amol/pyadps>`_
- **Issue Tracker**: Report issues or request features

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
