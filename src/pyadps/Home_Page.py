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
            "About": "# Python ADCP Data Processing Software (pyadps)",
        },
    )

    """
    # **Python ADCP Data Processing Software (pyadps)**

    `pyadps` is a Python package for processing moored Acoustic Doppler Current Profiler (ADCP) data. This interface walks you through reading a file, checking it for problems, and exporting the result — no programming required.

    `pyadps` was built for moored ADCP deployments that lack navigation (GPS) data, developed primarily for the COSINE and ECO-IOD mooring programs in the north Indian Ocean. Each processing page below shows you exactly what will be masked or corrected before you commit to it, so you can review the results yourself rather than trust the defaults blindly. This version currently works with data recorded in Earth coordinates only; Beam-coordinate support is planned for a future release.

    Please note that pyadps reads PD0 files from Teledyne RDI instruments; other manufacturers' ADCP files are not compatible. PD0 files from Workhorse, Ocean Surveyor, and DVS ADCPs can all be read, but take extra care when processing data from non-Workhorse models, since the pipeline's defaults were tuned against Workhorse deployments.

    * Documentation: https://pyadps.readthedocs.io
    * Source code: https://github.com/p-amol/pyadps
    * Bug reports: https://github.com/p-amol/pyadps/issues

    ## What you can do here

    * Read RDI PD0 binary files — including damaged or truncated ones, with automatic checksum verification and partial-data recovery
    * Step through quality control one page at a time: sensor health, signal quality, profile operations (trimming, side-lobe cut, regridding), and velocity checks, with a preview of the effect before you apply it
    * Visualize raw and processed data at every step
    * Export processed data to NetCDF (CF Convention compliant) or CSV
    * Save your processing choices to a configuration file, so a run can be reproduced or repeated later with minor changes

    ## Contribute
    Issue Tracker: https://github.com/p-amol/pyadps/issues
    Source Code: https://github.com/p-amol/pyadps

    ## Support
    If you are having issues, please open a ticket on the [issue tracker](https://github.com/p-amol/pyadps/issues).

    ## License
    The project is licensed under the MIT license.

    ---
    This software was developed with extensive use of Claude (Anthropic) as a coding assistant.
    """


if __name__ == "__main__":
    main()
