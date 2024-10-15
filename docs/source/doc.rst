**Documentation**
=================

Welcome to the ADPS documentation!

This guide will help you understand how to use the `pyadps` package for working with RDI ADCP binary files.

Features
--------
- **Access RDI ADCP Binary Files**: Easily interact with RDI ADCP binary files using Python 3.
- **Convert to NetCDF**: Convert RDI binary files to NetCDF format for seamless integration with other tools.
- **Publication Quality Processing**: Process ADCP data for high-quality output ready for publication.

Installation
------------
To use `pyadps`, you need Python 3.12.3. We recommend setting up a new Conda environment with this specific version of Python. Here's how:

1. **Create a Conda Environment**:

   .. code-block:: bash
      :linenos:

      conda create -n pyadps-env python=3.12.3
      conda activate pyadps-env

2. **Install pyadps**:

   .. code-block:: bash
      :linenos:

      pip install pyadps

Quickstart
----------
For a quick start, you have two main options:

1. **GUI Version**:
   Run the `pyadps` GUI to process data using a user-friendly Streamlit interface:
   
   .. code-block:: bash
      :linenos:

      run-pyadps

2. **Command-Line and Professional Use**:
   For more advanced use, import `pyadps` in your Python scripts and use its classes and functions:
   
   .. code-block:: python
      :linenos:

      import pyadps

      # Example usage
      from pyadps import FileHeader, FixedLeader, VariableLeader
      


Contribute
----------
We welcome contributions! If you'd like to help improve `pyadps`, please check out our resources:

- **Issue Tracker**: `<https://github.com/adps/issues>`_
- **Source Code**: `<https://github.com/p-amol/adps>`_

Support
-------
If you encounter issues or have questions, we are here to help. Reach out through our mailing list:

- **Mailing List**: `adps-python@google-groups.com`

License
-------
`pyadps` is licensed under the MIT License. See the `LICENSE` file for more details.

