# Web Application

pyadps includes an interactive web interface built with Streamlit for processing
RDI ADCP data without writing any code.

## Launching the Application

```bash
run-pyadps
```

The application opens in your default web browser at `http://localhost:8501`.

---

## Processing Workflow

The sidebar lists all pages in order. **Processing must be carried out
sequentially** — each page builds on the state saved by the previous one.
Always start at Page 1 and work downward through the sidebar.

```
 1. Read File            →  Upload file and inspect metadata
 2. View Raw Data        →  Visualize unprocessed data
 3. Download Raw File    →  Export raw data (optional)
 4. Time Diagnostics     →  Correct time axis before processing
 5. Sensor Health        →  Validate and correct environmental sensors
 6. Signal Quality       →  Apply QC thresholds
 7. Profile Operations   →  Trim, cut bins, and regrid
 8. Velocity Processing  →  Velocity checks and magnetic correction
 9. Write File           →  Export processed data and config
10. Add-Ons              →  Auto processing and file combiner
```

---

## Page Reference

### Home Page

```{image} ../_static/images/webapp/home_page.png
:alt: Home Page
:width: 100%
```

---

### Page 1: Read File

Upload an ADCP binary file and inspect its metadata. This page initialises
the `ProcessedDataset` that all subsequent pages use — **always start here**.

```{image} ../_static/images/webapp/01_read_file.png
:alt: Read File page
:width: 100%
```

**Tabs**

| Tab | Description |
|-----|-------------|
| File Header | Binary file structure and integrity check |
| Fixed Leader | System configuration, sensors, coordinate transform |
| Variable Leader | Time analysis, motion sensors, environmental sensors |
| Data Overview | Available data arrays and dimensions |

---

### Page 2: View Raw Data

Visualise the raw dataset before any processing is applied.

```{image} ../_static/images/webapp/02_view_raw_data.png
:alt: View Raw Data page
:width: 100%
```

**Tabs**

| Tab | Description |
|-----|-------------|
| Primary Data | Velocity, echo intensity, correlation, percent good heatmaps |
| Variable Leader | Dynamic sensor measurements (heading, pitch, roll, temperature) |
| Fixed Leader | Static configuration values |
| Advanced | BIT results, ADC channels, Error Status Words |

---

### Page 3: Download Raw File

Export the raw dataset to NetCDF or CSV without any processing applied.
This step is optional and can be skipped if you only need the processed output.

```{image} ../_static/images/webapp/03_download_raw_file.png
:alt: Download Raw File page
:width: 100%
```

---

### Page 4: Time Diagnostics

Diagnose and correct the time axis **before** any QC processing begins.

```{image} ../_static/images/webapp/04_time_diagnostics.png
:alt: Time Diagnostics page
:width: 100%
```

**Tabs**

| Tab | Description |
|-----|-------------|
| Diagnose | Time interval statistics, gap detection, and component plots |
| Snap Time Axis | Round drifted timestamps to the intended recording interval |
| Fill Time Gaps | Insert synthetic ensembles to make the time axis uniform |
| Reset | Undo corrections and restore the original time axis |

---

### Page 5: Sensor Health

Validate environmental sensors and optionally replace pressure, salinity,
and temperature with external data (e.g. from a co-deployed CTD).

```{image} ../_static/images/webapp/05_sensor_health.png
:alt: Sensor Health page
:width: 100%
```

**Tabs**

| Tab | Description |
|-----|-------------|
| 🌊 Pressure | Transducer depth sensor with drift analysis |
| 🧂 Salinity | Salinity sensor — replace with fixed value or CSV |
| 🌡️ Temperature | Temperature sensor with drift analysis |
| 🧭 Heading | Heading sensor visualisation |
| 📐 Pitch | Pitch sensor with threshold indicator |
| 🔄 Roll | Roll sensor with threshold indicator |
| ⚙️ Apply Checks | Configure roll/pitch thresholds and sound speed correction |
| 💾 Save/Reset | Commit or undo changes |

---

### Page 6: Signal Quality

Apply signal quality thresholds to mask low-quality data.

```{image} ../_static/images/webapp/06_signal_quality.png
:alt: Signal Quality page
:width: 100%
```

**Tabs**

| Tab | Description |
|-----|-------------|
| 📊 Noise Floor | Echo profiles for identifying the noise floor |
| 🎯 PG Threshold Advisor | Recommend a percent-good cutoff for a target precision |
| ⚙️ QC Tests | Configure correlation, echo, error velocity, percent good, false target |
| 🗺️ Mask Preview | Before/after mask comparison |
| 🔄 Fix Orientation | Correct beam direction (Up/Down) |
| 💾 Save/Reset | Commit or undo changes |

---

### Page 7: Profile Operations

Modify the profile structure — trim deployment/recovery periods, remove
side-lobe contaminated bins, and optionally regrid to a regular depth grid.

```{note}
Apply profile operations **after** signal quality checks. Regridding changes
the dataset structure and invalidates cell-based masks from earlier steps.
```

```{image} ../_static/images/webapp/07_profile_operations.png
:alt: Profile Operations page
:width: 100%
```

**Tabs**

| Tab | Description |
|-----|-------------|
| ✂️ Trim Ends | Remove ensembles from deployment/recovery periods |
| 📡 Side Lobe | Physics-based side lobe contamination removal |
| 🔧 Manual Cut | Cut arbitrary rectangular regions of cells and ensembles |
| 📐 Regrid | Interpolate to a regular depth grid |
| 💾 Save/Reset | Commit or undo changes |

---

### Page 8: Velocity Processing

Apply velocity-specific quality control and magnetic declination correction.

```{image} ../_static/images/webapp/08_velocity_processing.png
:alt: Velocity Processing page
:width: 100%
```

**Tabs**

| Tab | Description |
|-----|-------------|
| Magnetic Declination | Apply declination correction to U/V components |
| Velocity Thresholds | Set per-component cutoffs (U, V, W) |
| Despike Data | Median-filter spike detection |
| Flatline Detection | Detect frozen/stuck sensor values |
| Preview | View mask impact before committing |
| Save & Reset | Commit or undo changes |

---

### Page 9: Write File

Export the processed dataset and save the processing configuration.

```{image} ../_static/images/webapp/09_write_file.png
:alt: Write File page
:width: 100%
```

**Tabs**

| Tab | Description |
|-----|-------------|
| 📊 Preview Data | Visualise processed data with QC mask applied |
| 📝 Attributes | Add custom metadata (cruise number, location, etc.) |
| 💾 Export Data | Download NetCDF or CSV output |
| ⚙️ Config File | Download `config.ini` capturing all processing settings |

The exported `config.ini` can be used with the Auto Processing tool (Page 10)
to reprocess data with adjusted parameters without repeating the full workflow.

---

### Page 10: Add-Ons

Supplementary tools for batch reprocessing and combining multi-segment files.

**Auto Processing**

Reprocess an ADCP file using a previously saved `config.ini`.

```{image} ../_static/images/webapp/10_addons_autoprocess.png
:alt: Add-Ons — Auto Processing tab
:width: 100%
```

**File Combiner**

Merge multiple sequential ADCP binary files into a single file.

```{image} ../_static/images/webapp/10_addons_file_combiner.png
:alt: Add-Ons — File Combiner tab
:width: 100%
```

```{note}
When combining files, rename them with sequential numbering before uploading
(e.g. `KKS_000.000`, `KKS_001.000`, `KKS_002.000`) to ensure correct ordering.
```

---

## See Also

- {doc}`/quickstart` — Python API quick start
- {doc}`/processing/index` — Processing module reference
