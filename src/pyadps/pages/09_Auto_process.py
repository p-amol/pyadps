import os
import tempfile

import configparser
import json
import streamlit as st
from utils.autoprocess import autoprocess


@st.cache_data
def file_access(uploaded_file):
    """
    Function creates temporary directory to store the uploaded file.
    The path of the file is returned

    Args:
        uploaded_file (string): Name of the uploaded file

    Returns:
        path (string): Path of the uploaded file
    """
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return path


def display_config_as_json(config_file):
    config = configparser.ConfigParser()
    config.read_string(config_file.getvalue().decode("utf-8"))
    st.json({section: dict(config[section]) for section in config.sections()})


def main():
    st.title("ADCP Data Auto Processing Tool")
    st.write("Upload a binary input file and config.ini file for processing.")

    # File Upload Section
    uploaded_binary_file = st.file_uploader(
        "Upload ADCP Binary File", type=["000", "bin"]
    )
    uploaded_config_file = st.file_uploader(
        "Upload Config File (config.ini)", type=["ini"]
    )

    if uploaded_binary_file and uploaded_config_file:
        st.success("Files uploaded successfully!")

        # Display config.ini file content as JSON
        display_config_as_json(uploaded_config_file)

        fpath = file_access(uploaded_binary_file)
        # Process files
        with st.spinner("Processing files. Please wait..."):
            autoprocess(uploaded_config_file, binary_file_path=fpath)
            st.success("Processing completed successfully!")
            st.write("Processed file written.")


if __name__ == "__main__":
    main()
