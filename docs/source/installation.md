# Installation

This guide covers how to install pyadps and its dependencies.

## Requirements

- Python 3.12
- NumPy
- xarray
- pandas
- Streamlit (for web interface)

## Setting Up a Virtual Environment

It is strongly recommended to install pyadps in a dedicated virtual environment to avoid dependency conflicts.

### Using `venv`

```bash
python3.12 -m venv pyadps-env
source pyadps-env/bin/activate  # On Windows: pyadps-env\Scripts\activate
```

### Using `conda`

```bash
conda create -n pyadps-env python=3.12
conda activate pyadps-env
conda install pip
```

## Installation Methods

### From PyPI (Recommended)

```bash
pip install pyadps
```

### From Source

```bash
git clone https://github.com/p-amol/pyadps.git
cd pyadps
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

## Verifying Installation

```python
import pyadps
print(pyadps.__version__)
```

## Optional Dependencies

### For Documentation Building

```bash
pip install sphinx myst-nb sphinx-rtd-theme sphinx-autoapi
```

### For Testing

```bash
pip install pytest pytest-cov
```

## Troubleshooting

### Import Errors

If you get import errors, ensure all dependencies are installed:

```bash
pip install numpy xarray pandas
```

### Accessor Registration

Accessor methods are registered automatically when you import pyadps — no additional import is needed:

```python
import pyadps

# include_header=True is required for ds.header.* — it defaults to False
ds = pyadps.read('file.000', include_header=True)
ds.header.summary()
```

## Next Steps

- {doc}`quickstart` — Get started in 5 minutes
- {doc}`io/index` — Learn about data loading
- {doc}`processing/index` — Explore quality control
