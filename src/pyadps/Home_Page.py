import streamlit as st


def main():
    st.set_page_config(
        page_title="ADCP Data Processing Software",
        page_icon=":world_map:️",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={
            "Get Help": "http://github.com/p-amol/adps",
            "Report a bug": "http://github.com/adps/issues",
            "About": "# Python ADCP Data Processing Software (PyADPS)",
        },
    )

    """
    # **Python ADCP Data Processing Software (pyadps)**
    `pyadps` is a software for processing Teledyne RDI Acoustic Doppler Current Profiler (ADCP) PD0 files. Currently the software can process the data from Workhorse ADCPs.

    ## Features

    * Access RDI ADCP binary files using Python 3
    * Convert RDI binary files to netcdf
    * Process ADCP data 

    ## Contribute
    Issue Tracker: http://github.com/adps/issues
    Source Code: http://github.com/p-amol/adps

    ## Support
    If you are having issues, please let us know.
    We have a mailing list located at: adps-python@google-groups.com
    
    ## License
    The project is licensed under the MIT license.

    """


if __name__ == "__main__":
    main()
