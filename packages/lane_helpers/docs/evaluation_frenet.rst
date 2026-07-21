Frenet Transformation Evaluation
================================

The Frenet runtime evaluation compares :func:`~accvlab.lane_helpers.frenet.transform` on CPU and CUDA against
a Shapely-based reference while varying the number of Cartesian points to transform. Two smooth,
low-amplitude sine reference lanes with 10 and 100 points are evaluated. Query points are generated away from
exact lane vertices so that each point has a unique projection inside one segment.

The Shapely reference first computes the longitudinal location of each closest point along the reference lane.
NumPy then selects the corresponding geometry-derived segment normal and computes the signed lateral
coordinate. The prepared Shapely measurement reuses the reference-lane geometry, while the unprepared
measurement rebuilds the Shapely line, segment lengths, cumulative lengths, and normals during every call.
Input conversion is completed before timing for both measurements.

The reference implementation is kept separate from the evaluation driver:

* ``packages/lane_helpers/evaluation/frenet/shapely_frenet.py`` implements the Shapely and NumPy transformation.
* ``packages/lane_helpers/evaluation/frenet/frenet_evaluation.py`` generates inputs, validates results, measures
  runtime, writes a CSV file, and generates the runtime plot.

.. note::

   The following measurements are intended as directional guidance. They were generated on a system with an
   ``NVIDIA RTX 5000 Ada Generation`` GPU and an ``AMD Ryzen 9 7950X`` 16-Core Processor.

Shapely uses double-precision coordinates. The solid blue line uses the prepared reference, and the dashed
blue line includes reference preparation. Solid ACCV-Lab lines show float64 execution, while dotted lines in
the same CPU or CUDA color show float32 execution.

.. figure:: _generated/frenet_runtime_evaluation/runtime_comparison.png
   :alt: Frenet runtime comparison for reference lanes with 10 and 100 points
   :align: center
   :width: 100%

   Runtime per transformation call while varying the number of Cartesian points. Each panel compares prepared
   and unprepared Shapely with float64 and float32 ACCV-Lab CPU and CUDA execution for one reference-lane size.

Run the default evaluation from the repository root:

.. code-block:: bash

   python packages/lane_helpers/evaluation/frenet/frenet_evaluation.py

Use ``--num-query-points`` and ``--num-reference-points`` to select comma-separated point-count sweeps. Other
command-line options control warmup and measurement work, validation tolerances, and CSV and plot output paths.
