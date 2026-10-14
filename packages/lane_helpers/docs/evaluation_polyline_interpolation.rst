Polyline Interpolation Evaluation
=================================

The runtime evaluation compares batched interpolation for both CPU and CUDA against a Shapely LineString
reference over a grid of point counts, numbers of sampled distances, and batch sizes. Runtime plots report
milliseconds per interpolation call, while speedup plots report the x-fold improvement over the Shapely
reference.

Setup
-----

Shapely performs its geometry calculations in ``float64``. The evaluation also uses ``float64`` for ACCV-Lab
to provide a matched-precision comparison.

.. note::

   The following measurements are intended as directional guidance. Exact runtimes depend on the used system,
   with performance primarily influenced by the CPU and GPU.

   The plots shown here were generated on a system with an ``NVIDIA A100-SXM4-80GB`` GPU and two
   ``AMD EPYC 7J13`` 64-Core Processors.

Results
-------

Performance depends on the batch size for both CPU and CUDA execution. CUDA parallelism scales with the number
of polylines in the batch, so very small batch sizes may not fully utilize the GPU.

For practical problem sizes, it is recommended to choose the implementation based primarily on where the
tensors already live: CPU inputs should generally stay on CPU, and CUDA inputs should generally stay on CUDA.
Moving tensors only to use a different implementation can dominate the interpolation cost.

The plots below focus on batch sizes 1 and 64 as examples. The evaluation script runs for more batch sizes by
default, and other batch sizes can be easily added.

.. note::

   In the following runtime plots, markers highlight the smallest measured problem size, the largest measured
   problem size, and the 100-point/100-distance cell.

   In the speedup plots, markers highlight the smallest measured problem size and the largest speedup. If speedup
   is not above 1x everywhere, they also mark representative cells near the first matching point-count and
   distance-count configuration where speedup exceeds 1x.

Batch size 1 shows behavior for the smallest batch configuration in the benchmark:

.. figure:: _generated/polyline_runtime_evaluation/batch_1_runtime_comparison.png
   :alt: Runtime comparison heatmaps for batch size 1
   :align: center
   :width: 100%

   Runtime comparison between Shapely and ACCV-Lab ``float64`` for batch size 1. Rows vary the number of
   polyline points, and columns vary the number of sampled distances.

.. figure:: _generated/polyline_runtime_evaluation/batch_1_speedup_comparison.png
   :alt: Speedup comparison heatmaps for batch size 1
   :align: center
   :width: 100%

   Speedup comparison between Shapely and ACCV-Lab ``float64`` for batch size 1.

For larger batch sizes, CUDA can expose more parallel work and its speedup over the other methods typically
becomes more pronounced. Batch size 64 shows this behavior:

.. figure:: _generated/polyline_runtime_evaluation/batch_64_runtime_comparison.png
   :alt: Runtime comparison heatmaps for batch size 64
   :align: center
   :width: 100%

   Runtime comparison between Shapely and ACCV-Lab ``float64`` for batch size 64.

.. figure:: _generated/polyline_runtime_evaluation/batch_64_speedup_comparison.png
   :alt: Speedup comparison heatmaps for batch size 64
   :align: center
   :width: 100%

   Speedup comparison between Shapely and ACCV-Lab ``float64`` for batch size 64.

Running the Experiments
-----------------------

Run the evaluation from the repository root in an environment with the evaluation dependencies installed,
including a CUDA-capable PyTorch installation and the built ``lane_helpers`` package:

.. code-block:: bash

   python packages/lane_helpers/evaluation/polyline_interpolation/shapely_evaluation.py

The default sweep evaluates all configured point counts, sampled-distance counts, and batch sizes. It can take
a long time because every combination is measured on CPU, CUDA, and Shapely. Use ``--help`` to list all
options. In particular, ``--num-points``, ``--num-distances``, and ``--batch-sizes`` accept comma-separated
sweeps, while ``--num-polylines-per-measurement`` controls the amount of measured work. Use
``--accv-precision`` to evaluate ACCV-Lab with ``float32`` instead, and ``--assert-results`` to write numerical
differences after validating CPU and CUDA outputs against Shapely.

For example, run a smaller validation sweep with:

.. code-block:: bash

   python packages/lane_helpers/evaluation/polyline_interpolation/shapely_evaluation.py \
       --num-points 10,100 \
       --num-distances 10,100 \
       --batch-sizes 1 \
       --num-polylines-per-measurement 64 \
       --assert-results \
       --output-dir /tmp/polyline_evaluation

By default, outputs are written below
``packages/lane_helpers/evaluation/polyline_interpolation/polyline_shapely_eval_results``. The output directory
contains Markdown tables for each runtime, speedup, and requested validation metric, together with runtime and
speedup comparison plots for every batch size. Inspect the PNG files for an overview and the corresponding
``batch_<batch-size>_<metric>.md`` files for the measured values in each point-count and distance-count cell.
