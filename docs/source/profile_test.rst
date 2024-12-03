Profile Test
============

This module provides functionality to process Acoustic Doppler Current Profiler (ADCP)
data, specifically trimming and applying side-lobe corrections.


Class
-----

PLotEnds Class
^^^^^^^^^^^^^^
PlotEnds`: A class for interactively selecting the start and end ensembles for trimming ADCP data. 

.. autoclass:: utils.profile_test.PlotEnds

Functions
---------

Trims
^^^^^

.. autofunction:: utils.profile_test.trim_ends


Side-lobe interference correction based on Beam Angle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: utils.profile_test.side_lobe_beam_angle

Manual Cut Bins Delete
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: utils.profile_test.manual_cut_bins


Side-lobe Correction based on RSSI bump
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: utils.profile_test.side_lobe_rssi_bump

