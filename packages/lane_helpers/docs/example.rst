Example
=======

Polyline Interpolation
----------------------

The example below samples a rectangle-shaped polyline at a handful of distances.

.. important::

   You can run the example using the script
   ``packages/lane_helpers/examples/polyline_interpolation/basic_usage.py``.

.. note-literalinclude:: ../examples/polyline_interpolation/basic_usage.py
   :language: python
   :caption: packages/lane_helpers/examples/polyline_interpolation/basic_usage.py
   :linenos:

Frenet Transformation
---------------------

The example below transforms Cartesian points on both sides of a straight reference lane into longitudinal
and signed lateral coordinates using :func:`~accvlab.lane_helpers.frenet.transform`.

.. important::

   You can run the example using the script ``packages/lane_helpers/examples/frenet/basic_usage.py``.

.. note-literalinclude:: ../examples/frenet/basic_usage.py
   :language: python
   :caption: packages/lane_helpers/examples/frenet/basic_usage.py
   :linenos:
