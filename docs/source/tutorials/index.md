# Tutorials

Step-by-step tutorials for common pyadps workflows.

## Available Tutorials

```{toctree}
:maxdepth: 1

basic_reading
quality_control
batch_processing
visualization
```

## Tutorial Overview

| Tutorial | Description | Difficulty |
|----------|-------------|------------|
| {doc}`basic_reading` | Reading RDI files and accessing data | Beginner |
| {doc}`quality_control` | Applying QC tests to ADCP data | Intermediate |
| {doc}`batch_processing` | Processing multiple files | Intermediate |
| {doc}`visualization` | Creating diagnostic plots | Beginner |

## Prerequisites

Before starting these tutorials, ensure you have:

1. pyadps installed (`pip install pyadps`)
2. Sample ADCP data files
3. Basic Python knowledge

## Getting Sample Data

```python
# Check if you have test data
import pyadps

# Most tutorials use 'deployment.000' as an example
# Replace with your own file path
```

## Tutorial Format

Each tutorial includes:

- **Overview**: What you'll learn
- **Prerequisites**: Required knowledge and files
- **Step-by-step instructions**: With code examples
- **Expected output**: What you should see
- **Troubleshooting**: Common issues and solutions
