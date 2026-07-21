Polyline Interpolation Evaluation
=================================

The runtime evaluation compares batched interpolation for both CPU and CUDA against a Shapely LineString
reference over a grid of point counts, numbers of sampled distances, and batch sizes. Runtime plots report
milliseconds per interpolation call, while speedup plots report the x-fold improvement over the Shapely
reference.

.. _polyline-interpolation-precision-discussion:

Shapely performs its geometry calculations in float64. The primary comparison uses ACCV-Lab float32 because
float32 provides sufficient precision for most applications, while Shapely does not offer an equivalent
float32 mode. For a precision-matched comparison, see the :ref:`polyline-interpolation-fp64-appendix`.

.. seealso::

   The evaluation script is available at
   ``packages/lane_helpers/evaluation/polyline_interpolation/shapely_evaluation.py``. It can be used to run the
   benchmark sweep for different problem sizes on your target system.

Performance depends on the batch size for both CPU and CUDA execution. CUDA parallelism scales with the number
of polylines in the batch, so very small batch sizes may not fully utilize the GPU.

For practical problem sizes, it is recommended to choose the implementation based primarily on where the
tensors already live: CPU inputs should generally stay on CPU, and CUDA inputs should generally stay on CUDA.
Moving tensors only to use a different implementation can dominate the interpolation cost.

The plots below focus on batch sizes 1 and 64 as examples. The evaluation script runs for more batch sizes by
default, and other batch sizes can be easily added.

.. note::

   The following measurements are intended as directional guidance. Exact runtimes depend on the used system,
   with performance primarily influenced by the CPU and GPU.

   The plots shown here were generated on a system with an ``NVIDIA RTX 5000 Ada Generation`` GPU and an
   ``AMD Ryzen 9 7950X`` 16-Core Processor.

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

   Runtime comparison between Shapely float64 and ACCV-Lab float32 for batch size 1. Rows vary the number of
   polyline points, and columns vary the number of sampled distances.

.. figure:: _generated/polyline_runtime_evaluation/batch_1_speedup_comparison.png
   :alt: Speedup comparison heatmaps for batch size 1
   :align: center
   :width: 100%

   Speedup comparison between Shapely float64 and ACCV-Lab float32 for batch size 1.

For larger batch sizes, CUDA can expose more parallel work and its speedup over the other methods typically
becomes more pronounced. Batch size 64 shows this behavior:

.. figure:: _generated/polyline_runtime_evaluation/batch_64_runtime_comparison.png
   :alt: Runtime comparison heatmaps for batch size 64
   :align: center
   :width: 100%

   Runtime comparison between Shapely float64 and ACCV-Lab float32 for batch size 64.

.. figure:: _generated/polyline_runtime_evaluation/batch_64_speedup_comparison.png
   :alt: Speedup comparison heatmaps for batch size 64
   :align: center
   :width: 100%

   Speedup comparison between Shapely float64 and ACCV-Lab float32 for batch size 64.

.. _polyline-interpolation-fp64-appendix:

Appendix: FP64 Comparison
-------------------------

The following plots compare Shapely and ACCV-Lab with both using float64. This removes the precision
difference from the primary comparison while retaining the same benchmark configurations.

.. note::

   This comparison is primarily useful for evaluating matched-precision performance. See the
   :ref:`discussion of the practical float32 comparison <polyline-interpolation-precision-discussion>` above
   for why the primary plots use ACCV-Lab float32.

.. figure:: _generated/polyline_runtime_evaluation/batch_1_runtime_float64_comparison.png
   :alt: Float64 runtime comparison heatmaps for batch size 1
   :align: center
   :width: 100%

   Runtime comparison between Shapely and ACCV-Lab float64 for batch size 1.

.. figure:: _generated/polyline_runtime_evaluation/batch_1_speedup_float64_comparison.png
   :alt: Float64 speedup comparison heatmaps for batch size 1
   :align: center
   :width: 100%

   Speedup comparison between Shapely and ACCV-Lab float64 for batch size 1.

.. figure:: _generated/polyline_runtime_evaluation/batch_64_runtime_float64_comparison.png
   :alt: Float64 runtime comparison heatmaps for batch size 64
   :align: center
   :width: 100%

   Runtime comparison between Shapely and ACCV-Lab float64 for batch size 64.

.. figure:: _generated/polyline_runtime_evaluation/batch_64_speedup_float64_comparison.png
   :alt: Float64 speedup comparison heatmaps for batch size 64
   :align: center
   :width: 100%

   Speedup comparison between Shapely and ACCV-Lab float64 for batch size 64.
