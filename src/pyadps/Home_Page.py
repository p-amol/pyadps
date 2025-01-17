import streamlit as st


def main():
    st.set_page_config(
        page_title="ADCP Data Processing Software",
        page_icon=":world_map:️",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={
            "Get Help": "https://github.com/p-amol/pyadps",
            "Report a bug": "https://github.com/p-amol/pyadps/issues",
            "About": "# Python ADCP Data Processing Software (PyADPS)",
        },
    )

    """
    # **Python ADCP Data Processing Software (pyadps)**
  
    `pyadps` is a Python package for processing moored Acoustic Doppler Current Profiler (ADCP) data. It provides various functionalities such as data reading, quality control tests, NetCDF file creation, and visualization.

    This software offers both a graphical interface (Streamlit) for those new to Python and direct Python package access for experienced users. Please note that pyadps is primarily designed for Teledyne RDI workhorse ADCPs. Other company's ADCP files are not compatible, and while some other RDI models may work, they might require additional considerations.

    * Documentation: https://pyadps.readthedocs.io
    * Source code: https://github.com/p-amol/pyadps
    * Bug reports: https://github.com/p-amol/pyadps/issues
    ## Features

    * Access RDI ADCP binary files using Python 3
    * Convert RDI binary files to netcdf
    * Process ADCP data 

    ## Contribute
    Issue Tracker: https://github.com/p-amol/pyadps/issues
    Source Code: https://github.com/p-amol/pyadps

    ## Support
    If you are having issues, please let us know.
    We have a mailing list located at: adps-python@google-groups.com
    
    ## License
    The project is licensed under the MIT license.

    """


if __name__ == "__main__":
    main()
