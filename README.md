# pyadps

`pyadps` is a Python package for processing moored Acoustic Doppler Current Profiler (ADCP) data. It provides various functionalities such as data reading, quality control tests, NetCDF file creation, and visualization.

This software offers both a graphical interface (`Streamlit`) for those new to Python and direct Python package access for experienced users. Please note that `pyadps` is primarily designed for Teledyne RDI workhorse ADCPs. Other company's ADCP files are not compatible, and while some other RDI models may work, they might require additional considerations.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [License](#license)

## Installation

We recommend installing the package within a virtual environment. At present, the package is compatible exclusively with Python version 3.12.
You can create a Python environment using tools like `venv` or `conda`. Below are instructions for both methods.

### 1. Using `venv` (Built-in Python Tool)

#### Step 1: Install Python version 3.12 (if not already installed)
Ensure you have Python installed. You can download the latest version from [python.org](https://www.python.org/downloads/).

#### Step 2: Create a Virtual Environment 
- Open your terminal or command prompt.
- Navigate to your project folder:
```bash
cd /path/to/your/project
```
- Run the following command to create a virtual environment (replace adpsenv with your preferred environment name):

```bash
python -m venv adpsenv
```

#### Step 3: Activate the Environment
- On Windows:
```bash
adpsenv\Scripts\activate
```

- On macOS/Linux:
```bash
source adpsenv/bin/activate
```
You’ll see the environment name in your terminal prompt indicating the environment is active.

#### Step 4: Install Dependencies

Now you can install packages like this:

```bash
pip install pyadps
```

#### Step 5: Deactivate the Environment
When you’re done working in the environment, deactivate it by running:

```bash
deactivate
```




### 2. Using `conda` (Anaconda/Miniconda):

#### Step 1: Install Conda
First, you need to have Conda installed on your system. You can either install:

- [Anaconda (Full Distribution)](https://www.anaconda.com/products/individual)
- [Miniconda (Lightweight Version)](https://docs.conda.io/en/latest/miniconda.html)

#### Step 2: Create a Conda Environment with Python 3.12
Once Conda is installed, open a terminal or command prompt and run the following to create a new environment (replace `adpsenv` with your preferred environment name):

```bash
conda create --name adpsenv python=3.12
```

#### Step 3: Activate the Environment
```bash
conda activate adpsenv
```

#### Step 4: Install Dependencies
You can install packages with pip inside Conda environments. 
```bash
pip install pyadps
```

#### Step 5: Deactivate the Environment
When done, deactivate the environment by running:
```
conda deactivate
```

## Quick Start

### Streamlit web interface

Open a terminal or command prompt, activate the environment, and run the command.
```bash
run-pyadps
```


## License

This project is licensed under the MIT License. See the LICENSE file for details.
