# Installation

This guide covers how to install pyadps and its dependencies.

## Requirements

- Python 3.9 or higher
- NumPy
- xarray
- pandas
- Streamlit (for web interface)

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

Remember to import accessors before using domain-specific methods:

```python
import pyadps
import pyadps.accessors  # Required!

ds = pyadps.read('file.000')
ds.header.summary()  # Now this works
```

## Next Steps

- {doc}`quickstart` — Get started in 5 minutes
- {doc}`io/index` — Learn about data loading
- {doc}`processing/index` — Explore quality control
