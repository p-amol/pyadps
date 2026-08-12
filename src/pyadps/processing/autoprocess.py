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
    include_velocity: Optional[bool] = None,
    include_echo: Optional[bool] = None,
    include_correlation: Optional[bool] = None,
    include_percent_good: Optional[bool] = None,
    include_mask: Optional[bool] = None,
    apply_mask: Optional[bool] = None,
    output_dir: Optional[Union[str, Path]] = None,
    output_filename: Optional[str] = None,
    velocity_units: Optional[str] = None,
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
        Shorthand for include_velocity=True with every other include_*
        False. Kept for backward compatibility; prefer the include_*
        parameters below for anything more specific.
    include_velocity, include_echo, include_correlation, include_percent_good, include_mask : bool, optional
        Which components to save - the same choice as the Write File
        page's "Select Data Components to Export" / ``export_to_netcdf()``.
        Any left as None (the default) falls back to whatever was last
        exported via that page or ``ProcessedDataset`` (config.ini's
        ``[ExportOptions]`` section, if present) - so a config.ini saved
        after exporting Velocity + Echo Intensity will, by default,
        reproduce that same combination here. If the config predates this
        section (or nothing was ever exported), and no include_*/
        save_velocity_only argument is given either, the entire dataset
        is saved instead - the original behavior of this function.
    apply_mask : bool, optional
        If True, masked cells become NaN in velocity only (never in
        Echo Intensity/Correlation/Percent Good - see
        ``get_export_dataset``). None falls back to the config's stored
        choice, or True if there isn't one.
    output_dir : str or Path, optional
        Directory for output files. Defaults to same directory as input.
    output_filename : str, optional
        Output filename. Defaults to input filename with a suffix based
        on which components were included.
    velocity_units : str, optional
        Units for velocity output. None falls back to the config's
        stored choice, or 'cm/s' if there isn't one.
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

    >>> # Save velocity + echo intensity, mask included as its own variable
    >>> result = autoprocess(
    ...     'config.ini',
    ...     save_netcdf=True,
    ...     include_velocity=True,
    ...     include_echo=True,
    ...     include_mask=True,
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

        # Use the component-selection export path (get_export_dataset() /
        # export_to_netcdf()) whenever the caller explicitly asked for it
        # (save_velocity_only, or any include_* argument), or when the
        # loaded config.ini itself has an [ExportOptions] section (i.e.
        # it was saved after an actual export, so it has a real "last
        # exported" choice to reproduce). Old configs with no such
        # section, and no explicit override either, fall through to the
        # original full-dataset behavior unchanged - this function's
        # long-standing default.
        _explicit_include = any(
            v is not None
            for v in (
                include_velocity,
                include_echo,
                include_correlation,
                include_percent_good,
                include_mask,
            )
        )
        _use_component_export = (
            save_velocity_only or _explicit_include or config.isExportOptions
        )

        if _use_component_export:
            if save_velocity_only:
                _default_includes = {
                    "velocity": True,
                    "echo": False,
                    "correlation": False,
                    "percent_good": False,
                    "mask": False,
                }
            else:
                _default_includes = {
                    "velocity": config.export_include_velocity,
                    "echo": config.export_include_echo,
                    "correlation": config.export_include_correlation,
                    "percent_good": config.export_include_percent_good,
                    "mask": config.export_include_mask,
                }

            _include_velocity = (
                _default_includes["velocity"]
                if include_velocity is None
                else include_velocity
            )
            _include_echo = (
                _default_includes["echo"] if include_echo is None else include_echo
            )
            _include_correlation = (
                _default_includes["correlation"]
                if include_correlation is None
                else include_correlation
            )
            _include_percent_good = (
                _default_includes["percent_good"]
                if include_percent_good is None
                else include_percent_good
            )
            _include_mask = (
                _default_includes["mask"] if include_mask is None else include_mask
            )
            _apply_mask = config.export_apply_mask if apply_mask is None else apply_mask
            _velocity_units = (
                config.export_velocity_units if velocity_units is None else velocity_units
            )

            if output_filename is None:
                if _include_velocity and not any(
                    [_include_echo, _include_correlation, _include_percent_good, _include_mask]
                ):
                    suffix = "_velocity.nc"
                else:
                    suffix = "_export.nc"
                output_filename = binary_file_path.stem + suffix

            output_path = output_dir / output_filename

            proc.export_to_netcdf(
                output_path,
                include_velocity=_include_velocity,
                include_echo=_include_echo,
                include_correlation=_include_correlation,
                include_percent_good=_include_percent_good,
                include_mask=_include_mask,
                apply_mask=_apply_mask,
                ensure_depth_ascending=ensure_depth_ascending,
                velocity_units=_velocity_units,
            )
        else:
            # Save full dataset
            if output_filename is None:
                output_filename = binary_file_path.stem + "_processed.nc"
            output_path = output_dir / output_filename
            proc.to_netcdf(output_path, ensure_depth_ascending=ensure_depth_ascending)

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
        help="Output filename. Defaults to a name based on which components "
        "were included (e.g. '<input>_processed.nc' for the entire dataset, "
        "'<input>_velocity.nc' for velocity only, '<input>_export.nc' for "
        "any other combination).",
    )
    parser.add_argument(
        "--velocity-only",
        dest="save_velocity_only",
        action="store_true",
        help="Save only the velocity components (u, v, w) instead of the full "
        "dataset. Shorthand for velocity with no other --include-* flag.",
    )
    parser.add_argument(
        "--no-velocity",
        dest="include_velocity",
        action="store_false",
        default=None,
        help="Exclude velocity (on by default). Combine with --include-echo/"
        "--include-correlation/--include-percent-good/--include-mask to "
        "export only diagnostics, no velocity.",
    )
    parser.add_argument(
        "--include-echo",
        dest="include_echo",
        action="store_true",
        default=None,
        help="Include Echo Intensity (raw, unmasked). Defaults to whatever "
        "was last exported via the config.ini, if it has an [ExportOptions] "
        "section.",
    )
    parser.add_argument(
        "--include-correlation",
        dest="include_correlation",
        action="store_true",
        default=None,
        help="Include Correlation (raw, unmasked). Same default rule as --include-echo.",
    )
    parser.add_argument(
        "--include-percent-good",
        dest="include_percent_good",
        action="store_true",
        default=None,
        help="Include Percent Good (raw, unmasked). Same default rule as --include-echo.",
    )
    parser.add_argument(
        "--include-mask",
        dest="include_mask",
        action="store_true",
        default=None,
        help="Include the raw QC mask (1=invalid, 0=valid) as its own variable. "
        "Same default rule as --include-echo.",
    )
    parser.add_argument(
        "--no-mask",
        dest="apply_mask",
        action="store_false",
        default=None,
        help="Don't NaN-out masked velocity cells (mask is never applied to "
        "Echo Intensity/Correlation/Percent Good regardless - see "
        "get_export_dataset()). Defaults to the config.ini's stored choice, or True.",
    )
    parser.add_argument(
        "--velocity-units",
        dest="velocity_units",
        default=None,
        choices=["mm/s", "cm/s", "m/s"],
        help="Units for velocity output. Defaults to the config.ini's stored "
        "choice, or 'cm/s'.",
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
        include_velocity=args.include_velocity,
        include_echo=args.include_echo,
        include_correlation=args.include_correlation,
        include_percent_good=args.include_percent_good,
        include_mask=args.include_mask,
        apply_mask=args.apply_mask,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        velocity_units=args.velocity_units,
        ensure_depth_ascending=args.ensure_depth_ascending,
        print_summary=args.print_summary,
    )


if __name__ == "__main__":
    main()
