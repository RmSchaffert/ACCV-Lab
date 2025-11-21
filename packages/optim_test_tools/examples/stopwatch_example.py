# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
The stopwatch can be used to measure time for different steps in the implementation, across different parts of
the implementation. It supports warm-up handling and optional CPU usage measurement. It is a singleton, so you
can enable & configure it once (e.g. in the main script) and then use it anywhere else assuming that it is
enabled/configured if needed.

Overview:
  - The stopwatch is a singleton. Use ``Stopwatch()`` to retrieve the instance anywhere.
  - Call ``enable(...)`` once to activate measurements; otherwise calls become no-ops with minimal overhead.
  - You can measure repeated and one-time measurements across the code parts.
  - Fore repeated measurements, the name identifies the measurement (i.e. if the name is used multiple times,
    this is considered multiple instances of the same measurement).
  - Repeated measurements are performed after a warm-up phase is finished; One-time measurements can be
    performed at any time.
  - The stopwatch relies on tracking the number of finished iterations (for ending the warm-up phase,
    computing average times per iteration, knowing when to print outputs). To ensure that this information is
    correct, you need to call :meth:`finish_iter` at the end of each iteration of the outer loop of interest
    (e.g. training loop).
  - The output is printed in a formatted way with multiple sections. It contains the following information:

    - For repeated measurements, multiple values are displayed:

        - Average runtime per measured interval (an interval is started by calling :meth:`start_meas` and ended
          by calling :meth:`end_meas`)
        - Average iteration runtime (for the loop of interest). Here, if a measurement is performed multiple
          times in a single iteration, it is accumulated for the result. Similarly, if a measurement is
          performed in only some of the iterations, this will be reflected in the average iteration runtime
          (e.g. if a measurement is performed every 2nd iteration, the average iteration runtime will be half
          the average runtime per measured interval).
        - Total runtime
        - Mean CPU usage during the measurement which was selected (by calling :meth:`set_cpu_usage_meas_name`).

    - For one-time measurements

      - Values are displayed in own section
      - Measurements are not influenced by the warm-up phase, and can be performed at any time.
      - A measurement with a given name can be performed only once.
'''

import time
from accvlab.optim_test_tools import Stopwatch

# @NOTE
# In this example, the individual sections (such as Main Script, Code Part I, Code Part II, etc.),
# indicate parts of the code which, in the actual use case, would be in different files, and would rely
# on the `Stopwatch` being a singleton to enable its use throughout the files as shown here.


# ---------------------------- Main Script ----------------------------

stopwatch = Stopwatch()
# @NOTE
# To activate the stop watch, it needs to be enabled.
# If the following step is omitted, calls to method of the stopwatch will have no effect
# (and `get_num_nonwarmup_iters_measured()`) will always return `0``.
# Try commenting out the following line:
stopwatch.enable(num_warmup_iters=5, print_every_n_iters=2, do_cuda_sync=False)
# @NOTE
# Note that if the stopwatch is not enabled, calling its methods has minimal overhead
# (call to an empty method).

# ---------------------------------------------------------------------


# ---------------------------- Code Part I ----------------------------

# @NOTE
# Note that if a code part (such as this one) is used in isolation (meaning that there is no other code which
# already enabled the stopwatch), the stopwatch will be disabled and any related calls will have no effect.
# The overhead for the stopwatch is minimal in this case (call to an empty function).

# @NOTE
# Obtain the stopwatch instance (assuming that this is a separate file without access to the previously
# obtained one).
stopwatch = Stopwatch()

# @NOTE: Set up "meas1" to also measure mean CPU usage (can e.g. be the training loop in actual use case).
stopwatch.set_cpu_usage_meas_name("meas1")

# @NOTE
# One-time measurements can be performed at any point in the code. They are not affected by warm-up
# iterations and can be performed only once per used name, and are reported as such
# (i.e. in own section and without averages etc.).
stopwatch.start_one_time_measurement("complete_run")

# @NOTE: Another one-time measurement (start and end).
stopwatch.start_one_time_measurement("preparation")
time.sleep(0.1)
stopwatch.end_one_time_measurement("preparation")


num_iters = 16
for i in range(num_iters):

    # -------------------------- Code Part II --------------------------

    # @NOTE
    # Obtain the stopwatch instance and use it to start a measurement (ended in another part of the code).
    stopwatch = Stopwatch()
    stopwatch.start_meas("meas1")
    # Dummy workload to simulate some work.
    k = 0
    for j in range(1000000):
        k += float(j) ** 0.3 + 1.0 / (float(j) + 1.0)
    # ... continue and at some point call code part III

    # ------------------------------------------------------------------

    # -------------------------- Code Part III -------------------------

    # @NOTE: Here, we perform a measurement in another part of the code.
    stopwatch = Stopwatch()
    stopwatch.start_meas("meas2")
    time.sleep(0.05)
    stopwatch.end_meas("meas2")
    # ... continue and at some point call code part IV

    # ------------------------------------------------------------------

    # ------------------------- Code Part IV  --------------------------

    # @NOTE: End the measurement started in part II.
    stopwatch.end_meas("meas1")
    # ... continue and at some point call code part V

    # ------------------------------------------------------------------

    # --------------------------- Code Part V --------------------------

    stopwatch = Stopwatch()
    # @NOTE
    # Measurements may also be performed in only some of the iterations. In this case,
    # the average run-time per iteration will be lower than the average run-time
    # per measured interval.
    if i % 3 == 2:
        stopwatch.start_meas("meas3")
        time.sleep(0.01)
        stopwatch.end_meas("meas3")
    # @NOTE
    # We can perform a measurement with the same name multiple times in a
    # single iteration. This can be in the same or different code parts. In this case,
    # the average run-time per iteration will be higher than the average run-time
    # per measured interval.
    stopwatch.start_meas("meas2")
    time.sleep(0.01)
    stopwatch.end_meas("meas2")
    # ... continue and at some point call code part VI

    # ------------------------------------------------------------------

    # -------------------------- Code Part VI ---------------------------

    # @NOTE
    # Finish the iteration. This is very important to
    #   - Ensure that the warm-up iterations are counted down correctly and measurements are
    #     started after the warm-up iterations.
    #   - Ensure that the measurements are correctly accumulated and printed.
    stopwatch = Stopwatch()
    stopwatch.finish_iter()

    # ------------------------------------------------------------------


# ---------------------------- Code Part VI ----------------------------

# @NOTE
# End the one-time measurement
stopwatch.end_one_time_measurement("complete_run")
# @NOTE
# Print the final result
print("Final measurement:")
stopwatch.print_eval_times()

# ----------------------------------------------------------------------
