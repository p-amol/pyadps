
Signal Quality Test
===================

The `signal_quality` module provides essential functions for performing quality control on Acoustic Doppler Current Profiler (ADCP) data. It includes functionalities to assess and mask various types of data, ensuring the accuracy and reliability of your measurements. The module focuses on error velocity, echo intensity, percent good, and other quality metrics.


Quality_Check
^^^^^^^^^^^^^
.. autofunction:: utils.signal_quality.qc_check

Error_Velocity_Check
^^^^^^^^^^^^^^^^^^^^
.. autofunction:: utils.signal_quality.ev_check

Percent_Good_Check
^^^^^^^^^^^^^^^^^^
.. autofunction:: utils.signal_quality.pg_check

Identifies_And_Masks_False_Targets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: utils.signal_quality.false_target

Default_Mask
^^^^^^^^^^^^
.. autofunction:: utils.signal_quality.default_mask

Quality_Control_Prompts
^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: utils.signal_quality.qc_prompt
