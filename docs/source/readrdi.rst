Read Data Module
================

The `readrdi` module provides tools for reading and extracting data from RDI ADCP (Acoustic Doppler Current Profiler) binary files. This module supports both fixed and variable leader data extraction, along with other ADCP measurement variables.

Overview
--------

The `readrdi` module includes the following key components:

- **Classes**: `DotDict`, `FileHeader`, `FixedLeader`, `VariableLeader`, `ReadFile`
- **Functions**: `velocity`, `echo`, `correlation`, `percentgood`, `status`, `variables`

Classes
--------

DotDict
^^^^^^^

.. autoclass:: utils.readrdi.DotDict
   :members:
   :undoc-members:
   :show-inheritance:

File header
^^^^^^^^^^^

.. autoclass:: utils.readrdi.FileHeader
   :members:
   :undoc-members:
   :show-inheritance:

Fixed Leader
^^^^^^^^^^^^
   
.. autoclass:: utils.readrdi.FixedLeader
   :members:
   :undoc-members:
   :show-inheritance:

Variable Leader
^^^^^^^^^^^^^^^

.. autoclass:: utils.readrdi..VariableLeader
   :members:
   :undoc-members:
   :show-inheritance:

Velocity
^^^^^^^^

.. autoclass:: utils.readrdi.Velocity
   :members:
   :undoc-members:
   :show-inheritance:

Echo
^^^^

.. autoclass:: utils.readrdi.Echo
   :members:
   :undoc-members:
   :show-inheritance:

Correlation
^^^^^^^^^^^

.. autoclass:: utils.readrdi.Correlation
   :members:
   :undoc-members:
   :show-inheritance:


Percentgood
^^^^^^^^^^^

.. autoclass:: utils.readrdi.PercentGood
   :members:
   :undoc-members:
   :show-inheritance:


Status
^^^^^^

.. autoclass:: utils.readrdi.Status
   :members:
   :undoc-members:
   :show-inheritance:
   
ReadFile
^^^^^^^^

.. autoclass:: utils.readrdi.ReadFile
   :members:
   :undoc-members:
   :show-inheritance:

