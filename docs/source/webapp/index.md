# Web Application

Interactive Streamlit interface for ADCP data processing.

## Overview

The pyadps web application provides a graphical interface for processing RDI ADCP 
data. Built with Streamlit, it guides users through the complete processing workflow 
from file upload to data export.

```{note}
The web application uses the same `ProcessedDataset` orchestrator as the Python API,
ensuring consistency between interactive and programmatic processing.
```

## Quick Start

### Running the Application

```bash
# Navigate to the pages directory
cd pyadps/pages

# Run with Streamlit
streamlit run 01_Read_File.py
```

The application opens in your default web browser at `http://localhost:8501`.

### Processing Workflow

The application follows a sequential workflow through 9 pages:

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Read File      →  Upload and inspect ADCP binary file       │
│  2. View Raw Data  →  Visualize unprocessed data                │
│  3. Download Raw   →  Export raw data (NetCDF/CSV)              │
│  4. Sensor Health  →  Check/replace environmental sensors       │
│  5. QC Test        →  Apply signal quality thresholds           │
│  6. Profile Test   →  Trim ensembles, cut bins, regrid          │
│  7. Velocity Test  →  Magnetic correction, despike, thresholds  │
│  8. Write File     →  Export processed data                     │
│  9. Add-Ons        →  Batch processing, file combination        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Page Reference

### Page 1: Read File

**Purpose:** Upload ADCP binary files and inspect metadata.

**Features:**
- File upload for RDI binary formats (.000, .ENS, .ENX, .LTA, .STA)
- File health check (integrity verification)
- Fixed Leader inspection (static configuration)
- Variable Leader inspection (dynamic measurements)
- Time axis diagnostics
- Data overview with array shapes and valid data percentages

**Tabs:**
| Tab | Description |
|-----|-------------|
| File Header | Binary file structure and health check |
| Fixed Leader | System configuration, sensors, coordinate transform |
| Variable Leader | Time analysis, motion sensors, environmental sensors |
| Time Diagnostics | Interval plots and time component analysis |
| Data Overview | Available data arrays and dimensions |

**Key Actions:**
- **Check File Health** — Verify file integrity (size match, byte uniformity)
- **Show Data Types** — List available data types in file
- **Reset Processor** — Clear all processing and start fresh

---

### Page 2: View Raw Data

**Purpose:** Visualize raw (unprocessed) ADCP data.

**Features:**
- 2D heatmaps for velocity, echo intensity, correlation, percent good
- Time series plots for individual cells
- Variable Leader time series (heading, pitch, roll, temperature, etc.)
- Fixed Leader verification plots
- Advanced diagnostics (BIT results, ADC channels, Error Status Words)

**Tabs:**
| Tab | Description |
|-----|-------------|
| Primary Data | Velocity, Echo, Correlation, Percent Good heatmaps |
| Variable Leader | Dynamic sensor measurements |
| Fixed Leader | Static configuration values |
| Advanced | BIT results, ADC channels, ESW diagnostics |

**Controls:**
- X-axis toggle: Time or Ensemble number
- Beam selection: 1, 2, 3, or 4
- Cell selection for time series extraction

---

### Page 3: Download Raw File

**Purpose:** Export raw (unprocessed) data to standard formats.

**Features:**
- NetCDF export with selectable components
- CSV export for individual variables
- Custom metadata attributes
- Configurable file naming

**Export Options:**
| Component | Description |
|-----------|-------------|
| Fixed Leader | Static configuration variables |
| Variable Leader | Time-varying sensor data |
| Velocity | 4-beam velocity arrays |
| Echo Intensity | Backscatter strength |
| Correlation | Signal correlation |
| Percent Good | Valid ping percentage |

**Metadata Attributes:**
- Cruise number, ship name, project number
- Water depth, deployment depth
- Latitude, longitude
- Deployment and recovery dates
- Custom user-defined attributes

---

### Page 4: Sensor Health

**Purpose:** Validate and correct environmental sensor data.

**Features:**
- Pressure (depth) sensor inspection and replacement
- Salinity sensor inspection and replacement
- Temperature sensor inspection and replacement
- Heading, pitch, roll visualization
- Sound speed correction using corrected T/S values
- Roll/pitch threshold checks

**Tabs:**
| Tab | Description |
|-----|-------------|
| 🌊 Pressure | Transducer depth sensor with drift analysis |
| 🧂 Salinity | Salinity sensor with fixed value or CSV replacement |
| 🌡️ Temperature | Temperature sensor with drift analysis |
| 🧭 Heading | Heading sensor visualization |
| 📐 Pitch | Pitch sensor with threshold indicator |
| 🔄 Roll | Roll sensor with threshold indicator |
| ⚙️ Apply Checks | Configure and preview all checks |
| 💾 Save/Reset | Commit or reset processing |

**Correction Methods:**
- **Fixed Value** — Apply constant value across all ensembles
- **File Upload** — Replace with external CSV data (e.g., CTD)

**Sound Speed Correction:**
When temperature or salinity is modified, sound speed can be recalculated 
and optionally applied to velocity data using the Urick (1983) formula.

---

### Page 5: QC Test

**Purpose:** Apply signal quality control thresholds.

**Features:**
- Noise floor identification from echo intensity profiles
- Configurable QC thresholds with preview
- Three-beam mode for problematic beams
- Mask comparison visualization
- Beam orientation fix

**Tabs:**
| Tab | Description |
|-----|-------------|
| 📊 Noise Floor | Echo profiles for deployment/recovery ensembles |
| ⚙️ QC Tests | Configure all threshold values |
| 🗺️ Mask Preview | Before/after mask comparison |
| 🔄 Fix Orientation | Correct beam direction (Up/Down) |
| 💾 Save/Reset | Commit or reset QC processing |

**Available Tests:**
| Test | Default | Description |
|------|---------|-------------|
| Correlation | 64 | Minimum correlation threshold |
| Echo Intensity | 0 | Minimum echo intensity |
| Error Velocity | 2000 mm/s | Maximum error velocity |
| Percent Good | 0% | Minimum percent good |
| False Target | 50 | Maximum echo difference |

**Three-Beam Mode:**
When one beam is known to be problematic, enable three-beam mode and 
select the beam to ignore. QC checks will use remaining beams only.

---

### Page 6: Profile Test

**Purpose:** Spatial operations on the data profile.

**Features:**
- Trim deployment/recovery ensembles
- Cut bins affected by side lobe contamination
- Manual region cutting (cells and/or ensembles)
- Regrid to regular depth intervals
- Preview before committing

```{warning}
Profile operations should be applied **after** QC checks. Regridding changes 
the dataset structure and invalidates cell-based masks.
```

**Tabs:**
| Tab | Description |
|-----|-------------|
| ✂️ Trim Ends | Remove deployment/recovery ensembles |
| 📡 Side Lobe | Physics-based side lobe removal |
| 🔧 Manual Cut | Cut rectangular regions |
| 📏 Regrid | Transform to regular depth grid |
| 💾 Save/Reset | Commit or reset profile operations |

**Side Lobe Parameters:**
- **Orientation** — Up or Down (auto-detected from data)
- **Water Depth** — Required for downward-looking ADCP
- **Extra Cells** — Additional margin beyond calculated contamination

**Regrid Options:**
- **Method** — nearest, linear, or cubic interpolation
- **End Cell Option** — cell (last valid), surface, or manual depth

---

### Page 7: Velocity Test

**Purpose:** Velocity-specific quality control and corrections.

**Features:**
- Magnetic declination correction (local calculation or NOAA API)
- Component-specific velocity thresholds (U, V, W)
- Despike filtering with visualization
- Flatline detection
- Preview with mask comparison

**Tabs:**
| Tab | Description |
|-----|-------------|
| Magnetic Declination | Apply declination correction to U/V |
| Velocity Thresholds | Set component-specific cutoffs |
| Despike Data | Median filter spike detection |
| Flatline Detection | Detect frozen sensor values |
| Preview | View mask impact before saving |
| Save & Reset | Commit or reset velocity processing |

**Magnetic Declination Methods:**
| Method | Description |
|--------|-------------|
| pygeomag | Local calculation using WMM coefficients (2010-2030) |
| API | NOAA online magnetic declination service |
| Manual | Direct entry of known declination value |

**Velocity Thresholds:**
| Component | Default | Description |
|-----------|---------|-------------|
| U (East) | 2500 mm/s | Zonal velocity threshold |
| V (North) | 2500 mm/s | Meridional velocity threshold |
| W (Vertical) | 500 mm/s | Vertical velocity threshold |

**Despike Parameters:**
- **Kernel Size** — Window size for rolling median (default: 13)
- **Cutoff** — Standard deviations from median (default: 3.0)

**Flatline Parameters:**
- **Kernel Size** — Minimum consecutive constant values (default: 4)
- **Cutoff** — Maximum variation to consider "constant" (default: 1.0 mm/s)

---

### Page 8: Write File

**Purpose:** Export processed data to standard formats.

**Features:**
- Preview processed data with mask applied
- NetCDF export (velocity-only or full dataset)
- CSV export for velocity components
- Custom metadata attributes
- Configuration file generation for reproducibility

**Tabs:**
| Tab | Description |
|-----|-------------|
| 📊 Preview Data | Visualize processed data with mask |
| 📝 Attributes | Add custom metadata |
| 💾 Export Data | Generate downloadable files |
| ⚙️ Config File | Generate processing configuration |

**Export Types:**
| Type | Description |
|------|-------------|
| Velocity Only | U, V, W components with QC mask applied (recommended) |
| Full Dataset | Complete dataset including all variables |

**Velocity Units:**
- mm/s (original)
- cm/s (default output)
- m/s

**Configuration File:**
Generates a `config.ini` file capturing all processing settings. This file 
can be used with `pyadps.autoprocess()` for batch processing.

---

### Page 9: Add-Ons

**Purpose:** Supplementary processing tools.

**Features:**
- Auto Processing Tool — Reprocess using config.ini files
- Binary File Combiner — Merge multiple ADCP files

**Tabs:**
| Tab | Description |
|-----|-------------|
| 🔧 Auto Processing | Config-based reprocessing |
| 🔗 File Combiner | Combine multiple binary files |

**Auto Processing:**
1. Upload ADCP binary file
2. Upload config.ini from previous processing
3. Optionally adjust velocity units
4. Process and download results

**File Combiner:**
1. Upload multiple binary files
2. Validate file compatibility
3. Combine into single file
4. Download merged binary

```{note}
When combining files, ensure they are named sequentially 
(e.g., `KKS_000.000`, `KKS_001.000`) for correct ordering.
```

---

## Session State Architecture

The application uses Streamlit's session state to maintain data between pages:

```python
# Core state variables
st.session_state.processor    # ProcessedDataset orchestrator
st.session_state.ds           # xarray Dataset (raw data)
st.session_state.ds_header    # Header dataset for file checks
st.session_state.fname        # Current filename
st.session_state.fpath        # Path to temporary file

# Processing step tracking
st.session_state.processing_step  # Current step (0-5)

# Page-specific state (examples)
st.session_state.sensor_health_applied
st.session_state.qc_applied
st.session_state.profile_applied
st.session_state.velocity_applied
```

**Staging Processor Pattern:**

Processing pages use a "staging processor" for safe preview:

```python
# Main processor (committed changes)
proc = st.session_state.processor

# Staging processor (preview changes)
preview_proc = st.session_state.preview_qc_proc

# Preview without affecting main processor
runner = preview_proc.get_signal_quality_runner()
runner.correlation(cutoff=64)
preview_proc.commit_runner(runner)  # Only affects staging

# When satisfied, apply to main processor
runner = proc.get_signal_quality_runner()
runner.correlation(cutoff=64)
proc.commit_runner(runner)  # Commits to main processor
```

---

## Sidebar Information

Each processing page displays real-time statistics in the sidebar:

- **Total Cells** — Total data cells in dataset
- **Valid Cells** — Cells passing all QC checks
- **Masked Cells** — Cells flagged by QC checks
- **Processing Log** — Recent processing steps applied

---

## Best Practices

### Recommended Workflow Order

1. **Read File** — Always start here to load data
2. **View Raw Data** — Inspect data quality before processing
3. **Sensor Health** — Fix environmental sensor issues first
4. **QC Test** — Apply signal quality thresholds
5. **Profile Test** — Trim and cut bins (regrid last if needed)
6. **Velocity Test** — Apply velocity-specific checks
7. **Write File** — Export processed data

### Tips

- **Preview before saving** — Use preview features to check impact
- **Check the sidebar** — Monitor valid/masked percentages
- **Reset if needed** — Each page has reset functionality
- **Export config** — Generate config.ini for reproducibility
- **Velocity-only export** — Recommended for most use cases

---

## Troubleshooting

### Common Issues

**"No data loaded" error:**
- Navigate to Page 1 (Read File) and upload a file first

**Processing seems slow:**
- Large files (>100MB) may take time to process
- Regridding is computationally intensive

**Mask preview doesn't update:**
- Click "Preview" button after changing settings
- Check that the staging processor was updated

**Export file is empty:**
- Verify that not all data is masked
- Check the "Valid Cells" count in sidebar

---

## See Also

- {doc}`/processing/core` — ProcessedDataset Python API
- {doc}`/processing/autoprocess` — Automated batch processing
- {doc}`/quickstart` — Getting started guide
