# Quick Start

Get started with pyadps in 5 minutes.

## Loading Data

The simplest way to load an ADCP file:

```python
import pyadps
import pyadps.accessors

# Load complete dataset
ds = pyadps.read('deployment.000')

# What did we get?
print(ds)
```

## Inspecting the Data

### File Structure

```python
# Check file integrity
ds.header.check_file()

# View header summary
ds.header.summary()
```

### System Configuration

```python
# Get configuration
config = ds.fixed_leader.system_configuration()
print(f"Frequency: {config['Frequency']}")
print(f"Beam Direction: {config['Beam Direction']}")

# Full configuration summary
ds.fixed_leader.summary()
```

### Sensor Data

```python
# Check sensor readings
ds.variable_leader.summary()

# Time interval
interval = ds.variable_leader.get_time_interval()
print(f"Sampling interval: {interval}")
```

## Accessing Measurements

### Velocity Data

```python
# Get velocity array
velocity = ds['velocity']
print(f"Shape: {velocity.shape}")  # (beam, cell, time)

# Plot a time series
velocity.sel(beam=0, cell=10).plot()
```

### Quality Metrics

```python
# Correlation (data quality indicator)
correlation = ds['correlation']

# Echo intensity
echo = ds['echo_intensity']

# Percent good
percent_good = ds['percent_good']
```

## Quality Control

### Built-in Test Results

```python
# Check BIT results
bit_summary = ds.variable_leader.bit_result_summary()

if not bit_summary['all_passed']:
    print(f"BIT errors: {bit_summary['error_count']} ensembles")
```

### Data Continuity

```python
# Check for gaps
continuity = ds.variable_leader.ensemble_continuity_check()

if not continuity['is_continuous']:
    print(f"Found {continuity['gap_count']} gaps")
```

## Exporting Data

### To NetCDF

```python
ds.to_netcdf('processed.nc')
```

### Selective Export

```python
# Export only velocity
ds[['velocity']].to_netcdf('velocity_only.nc')
```

## Using the Web Interface

Launch the interactive interface:

```bash
pyadps
```

Or with streamlit directly:

```bash
streamlit run src/pyadps/pages/Home_Page.py
```

## Complete Example

```python
"""Complete pyadps workflow example."""
import pyadps
import pyadps.accessors

# Load data
ds = pyadps.read('deployment.000')

# Check file integrity
check = ds.header.check_file()
if check['File Size Match']:
    print("✓ File structure valid")

# Get configuration
config = ds.fixed_leader.system_configuration()
print(f"Frequency: {config['Frequency']}")
print(f"Direction: {config['Beam Direction']}")

# Check data quality
bit_summary = ds.variable_leader.bit_result_summary()
if bit_summary['all_passed']:
    print("✓ All BIT tests passed")

# Check timestamps
ts_valid = ds.variable_leader.validate_timestamps()
print(f"Time range: {ts_valid['time_range'][0]} to {ts_valid['time_range'][1]}")

# Access velocity data
velocity = ds['velocity']
print(f"Velocity shape: {velocity.shape}")

# Export to NetCDF
ds.to_netcdf('processed.nc')
print("✓ Exported to NetCDF")
```

## Next Steps

- {doc}`io/index` — Learn more about data loading
- {doc}`io/accessors` — Explore accessor methods
- {doc}`processing/index` — Apply quality control
- {doc}`tutorials/index` — Follow detailed tutorials
