import streamlit as st

def main():
        st.set_page_config(
            page_title="ADCP Data Processing Software",
            page_icon=":world_map:️",
            layout="wide",
            initial_sidebar_state="auto",
            menu_items={
                'Get Help': 'https://www.extremelycoolapp.com/help',
                'Report a bug': "https://www.extremelycoolapp.com/bug",
                'About': "# This is a header. This is an *extremely* cool app!"
            }
        )

        "# Python ADCP Data Processing Software (PyADPS)"


        """
        PyADPS reads and processes raw RDI ADCP files. Currently the software accepts workhorse ADCP files.

        """

        """
        There are three modules:\\
            - `Read module`: Reads binary file and provides file summary \\
            - `Write module`: Creates netcdf file \\
            - `Process module`: For processing ADCP Data 
        """
if __name__ == "__main__":
    main()