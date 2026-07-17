# pyadps/processing/autoprocess.py
"""
Automated ADCP data processing module.

This module provides a unified entry point for processing ADCP data using
either configuration files (from Streamlit UI) or ProcessingConfig objects
(from programmatic use).

The autoprocess function coordinates the full pipeline:
1. Load configuration
2. Read binary ADCP data
3. Process through ProcessedDataset orchestrator
4. Save outputs

Example usage:
    # From config file (Streamlit workflow)
    result = autoprocess('config.ini')

    # From ProcessingConfig object (programmatic workflow)
    config = ProcessingConfig(...)
    result = autoprocess(config, binary_file_path='data.000')

    # With output saving
    result = autoprocess('config.ini', save_netcdf=True, output_dir='output/')
"""

from pathlib import Path
from typing import Optional, Union

import xarray as xr

from .config import ProcessingConfig
from .core import ProcessedDataset


def autoprocess(
    config_file_or_object: Union[str, Path, ProcessingConfig],
    binary_file_path: Optional[Union[str, Path]] = None,
    save_netcdf: bool = False,
    save_velocity_only: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
    output_filename: Optional[str] = None,
    velocity_units: str = "cm/s",
    ensure_depth_ascending: bool = True,
    print_summary: bool = True,
) -> xr.Dataset:
    """
    Unified processing entry point supporting both config files and objects.

    Parameters
    ----------
    config_file_or_object : str, Path, or ProcessingConfig
        Path to .ini configuration file OR ProcessingConfig object.
    binary_file_path : str or Path, optional
        Path to ADCP binary file. If not provided, reads from config.
    save_netcdf : bool, default False
        If True, save processed dataset to NetCDF file.
    save_velocity_only : bool, default False
        If True and save_netcdf is True, save only velocity components
        (u, v, w) as separate 2D variables.
    output_dir : str or Path, optional
        Directory for output files. Defaults to same directory as input.
    output_filename : str, optional
        Output filename. Defaults to input filename with '_processed.nc' suffix.
    velocity_units : str, default 'cm/s'
        Units for velocity output when save_velocity_only=True.
        Options: 'mm/s', 'cm/s', 'm/s'.
    ensure_depth_ascending : bool, default True
        If True, ensure depth is in ascending order in output.
    print_summary : bool, default True
        If True, print processing summary to console.

    Returns
    -------
    xr.Dataset
        Processed dataset with QC mask applied.

    Examples
    --------
    >>> # Basic usage with config file
    >>> result = autoprocess('config.ini')

    >>> # With ProcessingConfig object
    >>> config = ProcessingConfig.from_ini('config.ini')
    >>> result = autoprocess(config, binary_file_path='data.000')

    >>> # Save full processed dataset
    >>> result = autoprocess('config.ini', save_netcdf=True)

    >>> # Save velocity components only in m/s
    >>> result = autoprocess(
    ...     'config.ini',
    ...     save_netcdf=True,
    ...     save_velocity_only=True,
    ...     velocity_units='m/s'
    ... )

    >>> # Custom output location
    >>> result = autoprocess(
    ...     'config.ini',
    ...     save_netcdf=True,
    ...     output_dir='processed/',
    ...     output_filename='deployment_2024_qc.nc'
    ... )

    >>> # Silent processing (no console output)
    >>> result = autoprocess('config.ini', print_summary=False)
    """
    # =========================================================================
    # PARSE CONFIGURATION
    # =========================================================================

    if isinstance(config_file_or_object, (str, Path)):
        # Load from INI file (Streamlit path)
        config_path = Path(config_file_or_object)
        config = ProcessingConfig.from_ini(str(config_path))

        # Get binary file path from config if not overridden
        if binary_file_path is None:
            if config.input_file_path and config.input_file_name:
                binary_file_path = Path(config.input_file_path) / config.input_file_name
            else:
                raise ValueError(
                    "binary_file_path must be provided or specified in config file"
                )
    else:
        # Already a ProcessingConfig object
        config = config_file_or_object
        if binary_file_path is None:
            raise ValueError(
                "binary_file_path must be provided when using ProcessingConfig object"
            )

    binary_file_path = Path(binary_file_path)

    if not binary_file_path.exists():
        raise FileNotFoundError(f"Binary file not found: {binary_file_path}")

    # =========================================================================
    # LOAD DATA
    # =========================================================================

    import pyadps  # noqa: PLC0415 — lazy import avoids numpy double-load at package boundary

    ds = pyadps.read(str(binary_file_path))

    # =========================================================================
    # PROCESS DATA
    # =========================================================================

    # Create ProcessedDataset and apply configuration
    proc = ProcessedDataset(ds)
    proc.apply_config(config)

    # Finalize with depth ordering option
    result = proc.finalize(ensure_depth_ascending=ensure_depth_ascending)

    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================

    if save_netcdf:
        # Determine output directory
        if output_dir is None:
            output_dir = binary_file_path.parent
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine output filename
        if output_filename is None:
            suffix = "_velocity.nc" if save_velocity_only else "_processed.nc"
            output_filename = binary_file_path.stem + suffix

        output_path = output_dir / output_filename

        if save_velocity_only:
            # Save velocity components only
            proc.velocity_to_netcdf(
                output_path,
                units=velocity_units,
                ensure_depth_ascending=ensure_depth_ascending,
            )
        else:
            # Save full dataset
            result.to_netcdf(output_path)

        if print_summary:
            print(f"Output saved to: {output_path}")

    # Print summary
    if print_summary:
        proc.print_summary()

    return result


def main() -> None:
    """
    Command-line entry point (``pyadps-auto``) for config-driven processing.

    Wraps :func:`autoprocess` for non-interactive use, e.g. reprocessing a
    deployment with a previously exported ``config.ini`` outside the
    Streamlit app. Always writes a NetCDF output, since a CLI run that
    produces nothing to disk isn't useful.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="pyadps-auto",
        description="Process an ADCP binary file using a pyadps config.ini file.",
    )
    parser.add_argument(
        "config", help="Path to the config.ini file (from export_config() or the Write File page)."
    )
    parser.add_argument(
        "-b",
        "--binary",
        dest="binary_file_path",
        default=None,
        help="Path to the ADCP binary file. Defaults to the path recorded in the config file.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Directory for the output NetCDF file. Defaults to the binary file's directory.",
    )
    parser.add_argument(
        "--output-filename",
        dest="output_filename",
        default=None,
        help="Output filename. Defaults to '<input>_processed.nc' (or '_velocity.nc' with --velocity-only).",
    )
    parser.add_argument(
        "--velocity-only",
        dest="save_velocity_only",
        action="store_true",
        help="Save only the velocity components (u, v, w) instead of the full dataset.",
    )
    parser.add_argument(
        "--velocity-units",
        dest="velocity_units",
        default="cm/s",
        choices=["mm/s", "cm/s", "m/s"],
        help="Units for velocity output when --velocity-only is set (default: cm/s).",
    )
    parser.add_argument(
        "--no-depth-ascending",
        dest="ensure_depth_ascending",
        action="store_false",
        help="Do not force ascending depth order in the output.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        dest="print_summary",
        action="store_false",
        help="Suppress the processing summary printed to the console.",
    )
    args = parser.parse_args()

    autoprocess(
        args.config,
        binary_file_path=args.binary_file_path,
        save_netcdf=True,
        save_velocity_only=args.save_velocity_only,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        velocity_units=args.velocity_units,
        ensure_depth_ascending=args.ensure_depth_ascending,
        print_summary=args.print_summary,
    )


if __name__ == "__main__":
    main()
