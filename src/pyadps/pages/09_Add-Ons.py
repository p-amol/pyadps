import os
import tempfile
from pathlib import Path

import re
import io
import contextlib

import configparser
import streamlit as st
from utils.autoprocess import autoprocess
from utils.multifile import ADCPBinFileCombiner

# To make the page wider if the user presses the reload button.
st.set_page_config(layout="wide")


def ansi_to_html(text):
    """
    Function to convert ANSI (console color) to HTML.
    To display the text, map the output to st.markdown
    """
    text = re.sub(r"\x1b\[31m", "<span style='color:red'><br>", text)  # red
    text = re.sub(r"\x1b\[32m", "<span style='color:green'><br>", text)  # red
    text = re.sub(r"\x1b\[33m", "<span style='color:orange'><br>", text)  # green
    text = re.sub(r"\x1b\[0m", "</span>", text)  # reset
    return text


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
    st.title("🧰 Add-Ons")
    st.header("🔧 Auto Processing Tool", divider=True)
    st.write(
        "You can use a configuration file from `pyadps` to re-process ADCP data by simply adjusting threshold values within the file. "
        "This allows you to fine-tune the output without repeating the full processing workflow in the software."
    )
    st.write(
        "To begin, upload both a binary ADCP file and a `config.ini` file for processing."
    )

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

    st.header("🔗 Binary File Combiner", divider=True)
    st.write(
        "ADCPs  may produce multiple binary segments instead of a single continuous file. "
        "This tool scans each uploaded binary file for the `7f7f` header, removes any broken ensembles at the beginning or the end, and combines all valid segments into a single file. "
        "To ensure correct order during concatenation, please rename the files using sequential numbering. "
        "For example: `KKS_000.000`, `KKS_001.000`, `KKS_002.000`."
    )
    output_cat_filename = "merged_000.000"
    st.info(f"Current file name: **{output_cat_filename}**")
    output_cat_filename_radio = st.radio(
        "Would you like to edit the output filename?",
        ["No", "Yes"],
        horizontal=True,
    )
    if output_cat_filename_radio == "Yes":
        output_cat_filename = st.text_input(
            "Enter file name (e.g., GD10A000)",
            value=output_cat_filename,
        )

    display_log = st.radio(
        "Display log from console:",
        ["No", "Yes"],
        horizontal=True,
    )

    uploaded_files = st.file_uploader(
        "Upload multiple binary files", type=["bin", "000"], accept_multiple_files=True
    )

    if uploaded_files:
        st.info("Saving uploaded files to temporary disk files...")

        # Save files to temporary path
        temp_file_paths = []
        for uploaded_file in uploaded_files:
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                temp_path = Path(tmp.name)
                temp_file_paths.append(temp_path)

        st.divider()
        st.subheader("🛠 Processing and Combining...")

        if display_log == "Yes":
            # The `buffer` is used to display console output to streamlit
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                adcpcat = ADCPBinFileCombiner()
                combined_data = adcpcat.combine_files(temp_file_paths)
            st.markdown(ansi_to_html(buffer.getvalue()), unsafe_allow_html=True)
        else:
            adcpcat = ADCPBinFileCombiner()
            combined_data = adcpcat.combine_files(temp_file_paths)

        if combined_data:
            st.success("✅ Valid binary data has been combined successfully.")
            st.warning(
                "⚠️ Note: The time axis in the final file may be irregular due to missing ensembles during concatenation."
            )
            st.download_button(
                label="📥 Download Combined Binary File",
                data=bytes(combined_data),
                file_name=output_cat_filename,
                mime="application/octet-stream",
            )
        else:
            st.warning("⚠️ No valid data found to combine.")

        # Optional: Clean up temporary files
        for path in temp_file_paths:
            try:
                os.remove(path)
            except Exception as e:
                st.warning(f"Failed to delete temp file {path}: {e}")
    else:
        st.info("Please upload binary files to begin.")


if __name__ == "__main__":
    main()
