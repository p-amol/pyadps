Installation
------------

We recommend installing the package within a virtual environment. At
present, the package is compatible exclusively with Python version 3.12.
You can create a Python environment using tools like ``venv`` or
``conda``. Below are instructions for both methods.

1. Using ``venv`` (Built-in Python Tool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


**Step 1: Install Python version 3.12 (if not already installed)**
Ensure you have Python installed. You can download the latest version
from `python.org <https://www.python.org/downloads/>`__.

**Step 2: Create a Virtual Environment**

-  Open your terminal or command prompt.
-  Navigate to your project folder:

.. code:: bash

   cd /path/to/your/project

-  Run the following command to create a virtual environment (replace
   adpsenv with your preferred environment name):

.. code:: bash

   python -m venv adpsenv

**Step 3: Activate the Environment**

-  On Windows:

.. code:: bash

   adpsenv\Scripts\activate

-  On macOS/Linux:

.. code:: bash

   source adpsenv/bin/activate

You’ll see the environment name in your terminal prompt indicating the
environment is active.

**Step 4: Install Dependencies**

Now you can install packages like this:

.. code:: bash

   pip install pyadps

**Step 5: Deactivate the Environment**

When you’re done working in the environment, deactivate it by running:

.. code:: bash

   deactivate

2. Using ``conda`` (Anaconda/Miniconda):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1: Install Conda**

First, you need to have Conda installed on your system. You can either
install:

-  `Anaconda (Full
   Distribution) <https://www.anaconda.com/products/individual>`__
-  `Miniconda (Lightweight
   Version) <https://docs.conda.io/en/latest/miniconda.html>`__

**Step 2: Create a Conda Environment with Python 3.12**

Once Conda is installed, open a terminal or command prompt and run the
following to create a new environment (replace ``adpsenv`` with your
preferred environment name):

.. code:: bash

   conda create --name adpsenv python=3.12

.. _step-3-activate-the-environment-1:

**Step 3: Activate the Environment**

.. code:: bash

   conda activate adpsenv

.. _step-4-install-dependencies-1:

**Step 4: Install Dependencies**

You can install packages with pip inside Conda environments.

.. code:: bash

   pip install pyadps

.. _step-5-deactivate-the-environment-1:

**Step 5: Deactivate the Environment**

When done, deactivate the environment by running:

::

   conda deactivate
