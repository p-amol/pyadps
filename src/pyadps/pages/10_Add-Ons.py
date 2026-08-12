"""
Streamlit Page 09: Add-Ons

This page provides supplementary tools for ADCP data processing:
1. Auto Processing Tool - Reprocess ADCP data using a config.ini file
2. Binary File Combiner - Combine multiple ADCP binary files into one

Uses pyadps v1.0.0 API:
- autoprocess() for config-based processing
- combine_file_list() for binary file combination
"""

import os
import io
import re
import tempfile
import contextlib
from pathlib import Path
from typing import List

import streamlit as st

# pyadps v1.0.0 imports
from pyadps.processing.autoprocess import autoprocess
from pyadps.processing.multifile import (
    combine_file_list,
    validate_adcp_file,
    CombineResult,
)

# Set page configuration
st.set_page_config(layout="wide", page_title="ADCP Add-Ons")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def ansi_to_html(text: str) -> str:
    """
    Convert ANSI color codes to HTML for Streamlit display.

    Parameters
    ----------
    text : str
        Text containing ANSI escape codes

    Returns
    -------
    str
        HTML-formatted text
    """
    # Red - errors
    text = re.sub(r"\x1b\[31m", "<span style='color:red'><br>", text)
    # Green - success
    text = re.sub(r"\x1b\[32m", "<span style='color:green'><br>", text)
    # Yellow/Orange - warnings
    text = re.sub(r"\x1b\[33m", "<span style='color:orange'><br>", text)
    # Reset
    text = re.sub(r"\x1b\[0m", "</span>", text)
    return text


@st.cache_data
def save_uploaded_file(uploaded_file) -> str:
    """
    Save uploaded file to a temporary directory.

    Parameters
    ----------
    uploaded_file : UploadedFile
        Streamlit uploaded file object

    Returns
    -------
    str
        Path to the saved temporary file
    """
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return path


def parse_config_to_dict(config_content: bytes) -> dict:
    """
    Parse config.ini content to a dictionary for display.

    Parameters
    ----------
    config_content : bytes
        Binary content of the config file

    Returns
    -------
    dict
        Nested dictionary of config sections and values
    """
    import configparser

    config = configparser.ConfigParser()
    config.read_string(config_content.decode("utf-8"))
    return {section: dict(config[section]) for section in config.sections()}


def format_bytes(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


# =============================================================================
# AUTO PROCESSING TOOL
# =============================================================================


def render_autoprocess_tool():
    """Render the Auto Processing Tool section."""
    st.header("🔧 Auto Processing Tool", divider=True)

    st.write("""
    Use a configuration file from `pyadps` to re-process ADCP data by simply 
    adjusting threshold values within the file. This allows you to fine-tune 
    the output without repeating the full processing workflow in the software.
    """)

    st.info("""
    **How it works:**
    1. Upload an ADCP binary file (.000 or .bin)
    2. Upload a config.ini file (generated from a previous processing session)
    3. The tool will apply all settings from the config file automatically
    """)

    # File upload section
    col1, col2 = st.columns(2)

    with col1:
        uploaded_binary = st.file_uploader(
            "Upload ADCP Binary File",
            type=["000", "bin", "ENS", "ENX"],
            key="autoprocess_binary",
            help="Select an RDI ADCP binary file",
        )

    with col2:
        uploaded_config = st.file_uploader(
            "Upload Config File (config.ini)",
            type=["ini"],
            key="autoprocess_config",
            help="Configuration file from previous processing",
        )

    # Process when both files are uploaded
    if uploaded_binary and uploaded_config:
        st.success("✅ Files uploaded successfully!")

        # Display config file content
        with st.expander("View Configuration File Contents", expanded=False):
            try:
                config_dict = parse_config_to_dict(uploaded_config.getvalue())
                st.json(config_dict)
            except Exception as e:
                st.error(f"Error parsing config file: {e}")

        # Processing options
        st.subheader("Processing Options")

        save_netcdf = st.checkbox(
            "Save NetCDF output",
            value=True,
            help="Save processed data to NetCDF file",
        )

        use_config_export_settings = st.checkbox(
            "Use export settings from config.ini",
            value=True,
            help="If this config.ini was saved after an export on the Write "
            "File page, reuse that same component selection automatically. "
            "Uncheck to choose different components for this run. If the "
            "config predates this feature (no [ExportOptions] section) and "
            "nothing is chosen below, the entire dataset is saved instead.",
        )

        include_velocity = include_echo = include_correlation = None
        include_percent_good = include_mask = apply_mask = None
        velocity_units = None

        if not use_config_export_settings:
            st.write("Select Data Components to Export")
            col1, col2, col3 = st.columns(3)
            with col1:
                include_velocity = st.checkbox("Velocity", value=True)
                include_echo = st.checkbox("Echo Intensity", value=False)
            with col2:
                include_correlation = st.checkbox("Correlation", value=False)
                include_percent_good = st.checkbox("Percent Good", value=False)
            with col3:
                include_mask = st.checkbox(
                    "QC Mask",
                    value=False,
                    help="The raw QC mask (1=invalid, 0=valid). Export it "
                    "alongside the raw Echo Intensity/Correlation/Percent "
                    "Good values above to see which cells were flagged "
                    "without losing the diagnostic data that explains why.",
                )

            col1, col2 = st.columns(2)
            with col1:
                apply_mask = st.checkbox(
                    "Apply QC mask to exported data",
                    value=True,
                    help="Masked cells become NaN in Velocity only - never "
                    "in Echo Intensity/Correlation/Percent Good, which are "
                    "indexed by physical beam, not the mask's velocity-"
                    "derived U/V/W/combined slots.",
                )
            with col2:
                velocity_units = st.selectbox(
                    "Velocity units",
                    options=["cm/s", "mm/s", "m/s"],
                    index=0,
                    help="Units for velocity output",
                )

        # Process button
        if st.button("🚀 Process Data", type="primary", use_container_width=True):
            # Save files to temp directory
            binary_path = save_uploaded_file(uploaded_binary)

            # Save config to temp file
            config_temp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".ini", delete=False
            )
            config_temp.write(uploaded_config.getvalue().decode("utf-8"))
            config_temp.close()

            try:
                with st.spinner("Processing files. Please wait..."):
                    # Capture console output
                    buffer = io.StringIO()
                    with contextlib.redirect_stdout(buffer):
                        result = autoprocess(
                            config_file_or_object=config_temp.name,
                            binary_file_path=binary_path,
                            save_netcdf=save_netcdf,
                            include_velocity=include_velocity,
                            include_echo=include_echo,
                            include_correlation=include_correlation,
                            include_percent_good=include_percent_good,
                            include_mask=include_mask,
                            apply_mask=apply_mask,
                            velocity_units=velocity_units,
                            print_summary=True,
                        )

                    # Display output
                    console_output = buffer.getvalue()
                    if console_output:
                        st.text_area(
                            "Processing Log",
                            value=console_output,
                            height=200,
                        )

                st.success("✅ Processing completed successfully!")

                # Display result summary
                st.subheader("Result Summary")
                col1, col2, col3 = st.columns(3)

                with col1:
                    n_time = result.sizes.get("time", 0)
                    st.metric("Ensembles", f"{n_time:,}")

                with col2:
                    n_cells = result.sizes.get("cell", result.sizes.get("depth", 0))
                    st.metric("Depth Cells", n_cells)

                with col3:
                    if "mask" in result.data_vars:
                        mask = result["mask"].values
                        masked_pct = (mask == 1).mean() * 100
                        st.metric("Data Masked", f"{masked_pct:.1f}%")
                    else:
                        st.metric("Data Masked", "N/A")

                # Provide download if NetCDF was saved. autoprocess() picks
                # the output filename/suffix itself based on which
                # components ended up included (entire dataset, velocity
                # only, or any other combination) - so look for whatever it
                # actually wrote rather than re-deriving the same choice
                # here and risking the two falling out of sync.
                if save_netcdf:
                    candidates = sorted(
                        Path(binary_path).parent.glob(
                            f"{Path(binary_path).stem}*.nc"
                        )
                    )
                    output_path = candidates[0] if candidates else None

                    if output_path is not None and output_path.exists():
                        output_filename = output_path.name
                        with open(output_path, "rb") as f:
                            st.download_button(
                                label="📥 Download Processed NetCDF",
                                data=f.read(),
                                file_name=output_filename,
                                mime="application/x-netcdf",
                            )

            except FileNotFoundError as e:
                st.error(f"❌ File not found: {e}")
            except ValueError as e:
                st.error(f"❌ Configuration error: {e}")
            except Exception as e:
                st.error(f"❌ Processing error: {e}")
                st.exception(e)

            finally:
                # Cleanup temp config file
                try:
                    os.unlink(config_temp.name)
                except Exception:
                    pass

    else:
        st.info(
            "👆 Please upload both an ADCP binary file and a config.ini file to begin."
        )


# =============================================================================
# BINARY FILE COMBINER
# =============================================================================


def render_file_combiner_tool():
    """Render the Binary File Combiner section."""
    st.header("🔗 Binary File Combiner", divider=True)

    st.write("""
    ADCPs may produce multiple binary segments instead of a single continuous file. 
    This tool scans each uploaded binary file for the `7f7f` header, removes any 
    broken ensembles at the beginning or the end, and combines all valid segments 
    into a single file.
    """)

    st.warning("""
    ⚠️ **Important:** To ensure correct order during concatenation, please rename 
    the files using sequential numbering. For example: `KKS_000.000`, `KKS_001.000`, 
    `KKS_002.000`.
    """)

    # Output filename configuration
    st.subheader("Output Configuration")

    col1, col2 = st.columns([2, 1])

    with col1:
        output_filename = st.text_input(
            "Output filename",
            value="merged_000.000",
            help="Name for the combined output file",
        )

    with col2:
        display_log = st.checkbox(
            "Display processing log",
            value=False,
            help="Show detailed console output during processing",
        )

    # Advanced options
    with st.expander("Advanced Options", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            skip_invalid = st.checkbox(
                "Skip invalid files",
                value=True,
                help="Continue processing if some files are invalid",
            )

        with col2:
            check_ensemble_size = st.checkbox(
                "Validate ensemble size consistency",
                value=True,
                help="Ensure all files have the same ensemble size",
            )

    # File upload
    st.subheader("Upload Files")
    uploaded_files = st.file_uploader(
        "Upload multiple binary files",
        type=["bin", "000", "ENS", "ENX"],
        accept_multiple_files=True,
        key="combiner_files",
        help="Select multiple ADCP binary files to combine",
    )

    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} files selected")

        # Display file list
        with st.expander("View Uploaded Files", expanded=False):
            for i, f in enumerate(uploaded_files, 1):
                st.write(f"{i}. {f.name} ({format_bytes(f.size)})")

        # Validate button
        if st.button("🔍 Validate Files", use_container_width=True):
            st.subheader("Validation Results")

            # Save files to temp and validate
            temp_paths: List[Path] = []
            validation_results = []

            for uploaded_file in uploaded_files:
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    temp_path = Path(tmp.name)
                    temp_paths.append(temp_path)

                    # Reset file pointer for later use
                    uploaded_file.seek(0)

                    # Validate
                    result = validate_adcp_file(temp_path)
                    validation_results.append(
                        {
                            "File": uploaded_file.name,
                            "Valid": "✅" if result.is_valid else "❌",
                            "Ensembles": result.valid_ensembles
                            if result.is_valid
                            else 0,
                            "Ensemble Size": result.ensemble_size
                            if result.is_valid
                            else 0,
                            "Truncated": "Yes" if result.is_truncated else "No",
                            "Error": result.error_message
                            if not result.is_valid
                            else "",
                        }
                    )

            # Display validation table
            import pandas as pd

            df = pd.DataFrame(validation_results)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Cleanup temp files
            for path in temp_paths:
                try:
                    os.unlink(path)
                except Exception:
                    pass

        # Process button
        st.divider()
        if st.button(
            "🔗 Combine Files",
            type="primary",
            use_container_width=True,
        ):
            st.subheader("🛠 Processing and Combining...")

            # Save files to temporary path
            temp_file_paths: List[Path] = []
            for uploaded_file in uploaded_files:
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    temp_file_paths.append(Path(tmp.name))

            # Create output path
            output_dir = tempfile.mkdtemp()
            output_path = Path(output_dir) / output_filename

            try:
                if display_log:
                    # Capture and display console output
                    import logging

                    # Set up logging to capture multifile output
                    logging.basicConfig(level=logging.INFO)
                    buffer = io.StringIO()
                    handler = logging.StreamHandler(buffer)
                    handler.setLevel(logging.INFO)
                    logging.getLogger("pyadps.processing.multifile").addHandler(handler)

                # Combine files
                result: CombineResult = combine_file_list(
                    files=temp_file_paths,
                    output_file=output_path,
                    skip_invalid=skip_invalid,
                    require_matching_ensemble_size=check_ensemble_size,
                )

                if display_log:
                    log_output = buffer.getvalue()
                    if log_output:
                        st.markdown(
                            ansi_to_html(log_output),
                            unsafe_allow_html=True,
                        )

                # Display results
                if result.success:
                    st.success("✅ Valid binary data has been combined successfully!")

                    # Result metrics
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Files Processed",
                            f"{result.files_processed}/{result.files_total}",
                        )

                    with col2:
                        st.metric(
                            "Total Ensembles",
                            f"{result.total_ensembles:,}",
                        )

                    with col3:
                        st.metric(
                            "Output Size",
                            format_bytes(result.total_bytes),
                        )

                    # Warnings
                    if result.skipped_files:
                        st.warning(
                            f"⚠️ Skipped {len(result.skipped_files)} files: "
                            f"{', '.join(result.skipped_files)}"
                        )

                    st.warning(
                        "⚠️ **Note:** The time axis in the final file may be irregular "
                        "due to missing ensembles during concatenation."
                    )

                    # Download button
                    with open(output_path, "rb") as f:
                        combined_data = f.read()

                    st.download_button(
                        label="📥 Download Combined Binary File",
                        data=combined_data,
                        file_name=output_filename,
                        mime="application/octet-stream",
                    )

                else:
                    st.error(f"❌ Combination failed: {result.error_message}")

                    if result.skipped_files:
                        st.write("**Skipped files:**")
                        for name in result.skipped_files:
                            st.write(f"  - {name}")

            except Exception as e:
                st.error(f"❌ Error during processing: {e}")
                st.exception(e)

            finally:
                # Cleanup temporary files
                for path in temp_file_paths:
                    try:
                        os.unlink(path)
                    except Exception:
                        pass

    else:
        st.info("👆 Please upload binary files to begin.")


# =============================================================================
# MAIN APPLICATION
# =============================================================================


def main():
    """Main application entry point."""
    st.title("🧰 Add-Ons")
    st.write("*Supplementary tools for ADCP data processing*")

    # Create tabs for the two main tools
    tab1, tab2 = st.tabs(["🔧 Auto Processing", "🔗 File Combiner"])

    with tab1:
        render_autoprocess_tool()

    with tab2:
        render_file_combiner_tool()

    # Footer
    st.divider()
    st.caption(
        "These tools use the pyadps v1.0.0 API for ADCP data processing. "
        "For more information, see the pyadps documentation."
    )


if __name__ == "__main__":
    main()
