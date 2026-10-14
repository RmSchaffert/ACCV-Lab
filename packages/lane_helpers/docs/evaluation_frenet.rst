Frenet Transformation Evaluation
================================

The Frenet runtime evaluation compares :func:`~accvlab.lane_helpers.frenet.transform` on CPU and CUDA against
a Shapely-based reference while varying the number of Cartesian points to transform. Two smooth,
low-amplitude sine reference lanes with 10 and 100 points are evaluated. Query points are generated away from
exact lane vertices so that each point has a unique projection inside one segment.

Setup
-----

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
   ``NVIDIA A100-SXM4-80GB`` GPU and two ``AMD EPYC 7J13`` 64-Core Processors.

Results
-------

Shapely uses double-precision coordinates. The solid blue line uses the prepared reference, and the dashed
blue line includes reference preparation. ACCV-Lab also uses ``float64`` by default for a matched-precision
comparison.

.. figure:: _generated/frenet_runtime_evaluation/runtime_comparison_float64.png
   :alt: Frenet runtime comparison for reference lanes with 10 and 100 points
   :align: center
   :width: 100%

   Runtime per transformation call while varying the number of Cartesian points. Each panel compares prepared
   and unprepared Shapely with ACCV-Lab ``float64`` CPU and CUDA execution for one reference-lane size.

Running the Experiments
-----------------------

Run the evaluation from the repository root in an environment with the evaluation dependencies installed,
including a CUDA-capable PyTorch installation and the built ``lane_helpers`` package:

.. code-block:: bash

   python packages/lane_helpers/evaluation/frenet/frenet_evaluation.py

The default sweep evaluates reference lanes with 10 and 100 points over several query-point counts. Use
``--help`` to list all options. ``--num-query-points`` and ``--num-reference-points`` accept comma-separated
point-count sweeps, while ``--num-points-per-measurement`` controls the amount of measured work. Use
``--accv-precision`` to evaluate ACCV-Lab with ``float32`` instead. CPU and CUDA results are validated against
Shapely for every configuration using the precision-appropriate tolerance.

For example, run a smaller sweep with:

.. code-block:: bash

   python packages/lane_helpers/evaluation/frenet/frenet_evaluation.py \
       --num-query-points 10,1000 \
       --num-reference-points 10 \
       --num-points-per-measurement 10000 \
       --output-path /tmp/frenet_evaluation/results_float64.csv \
       --plot-path /tmp/frenet_evaluation/runtime_comparison_float64.png

By default, the script writes
``packages/lane_helpers/evaluation/frenet/frenet_eval_results/point_count_results_float64.csv`` and
``packages/lane_helpers/evaluation/frenet/frenet_eval_results/runtime_comparison_float64.png``.
The default filenames use ``float32`` when that precision is selected; explicit output paths are used as given.
Inspect the plot for the runtime trends and the CSV file for the configuration, number of measured runs,
runtime, and maximum absolute difference of every result row.
