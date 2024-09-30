Read Binary File Module
=======================

The ``pyreadrdi`` module provides functionality to read and process RDI (RD Instruments) ADCP binary files. It includes functions to extract header information and fixed leader data from ADCP files.

Classes
--------

Bcolors
^^^^^^^

.. autoclass:: utils.pyreadrdi.bcolors
   :members:
   :undoc-members:
   :show-inheritance:
   
   
ErrorCode
^^^^^^^^^

.. autoclass:: utils.pyreadrdi.ErrorCode
   :members:
   :undoc-members:
   :show-inheritance:
   
   
Functions
----------

Safe_Open
^^^^^^^^^

.. autofunction:: utils.pyreadrdi.safe_open

Safe_Read
^^^^^^^^^

.. autofunction:: utils.pyreadrdi.safe_open


File Header
^^^^^^^^^^^

.. autofunction:: utils.pyreadrdi.fileheader


Fixed Leader
^^^^^^^^^^^^

.. autofunction:: utils.pyreadrdi.fixedleader




Varibale Leader
^^^^^^^^^^^^^^^

.. autofunction:: utils.pyreadrdi.variableleader


Data_Type
^^^^^^^^^

.. autofunction:: utils.pyreadrdi.datatype

