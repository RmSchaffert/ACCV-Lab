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

import copy
from argparse import Namespace


def add_stream_petr_arguments_to_parser(parser):
    parser = copy.deepcopy(parser)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        metavar="B",
        help="Size of the batches to be processed & output",
    )
    parser.add_argument(
        "--num_iterations", type=int, default=1000, metavar="S", help="Number of iterations to perform"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        metavar="P",
        help="Number of workers for the external source data loader (see DALI documentation of ExternalSource class)",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=4,
        metavar="T",
        help="Number of (CPU) threads to be used in DALI CPU & mixed operators",
    )
    parser.add_argument(
        "--prefetch_queue_depth",
        type=int,
        default=2,
        metavar="Q",
        help="Depth of the pre-fetch queue for the DALI pipeline",
    )
    parser.add_argument(
        "--num_warmup_iters",
        type=int,
        default=300,
        metavar="U",
        help="Number of warmup iterations (Measuring run time starts after that)",
    )
    parser.add_argument(
        "--convert_format_for_training",
        type=int,
        default=1,
        metavar="C",
        help="Whether to convert data format for training (1) or keep original format (0)",
    )
    return parser


def get_default_stream_petr_pipeline_config(num_epochs, num_iters_per_epoch, use_randomizations=False):

    def convert_to_namespace(as_dict):
        for k, v in as_dict.items():
            if isinstance(v, dict):
                as_dict[k] = convert_to_namespace(v)
        res = Namespace(**as_dict)
        return res

    if use_randomizations:
        affine_trafo_config = {
            "rand_flip": True,
            "resize_lim": (0.86, 1.25),
            "output_hw": (256, 704),
        }
        bev_bboxes_trafo_config = {
            "rotation_range": [-0.3925, 0.3925],
            "translation_range": [0, 0, 0],
            "scaling_range": [0.95, 1.05],
        }
    else:
        affine_trafo_config = {
            "rand_flip": False,
            "resize_lim": (1.0, 1.0),
            "output_hw": (256, 704),
        }
        bev_bboxes_trafo_config = {
            "rotation_range": [0.3925, 0.3925],
            "translation_range": [0, 0, 0],
            "scaling_range": [1.05, 1.05],
        }

    config = dict(
        image_size=[1600, 900],
        pipeline_config={"num_workers": 1, "num_threads": 4, "prefetch_queue_depth": 2},
        decoder_config={
            "hw_decoder_load": 0.65,
            "as_rgb": True,
        },
        point_cloud_range={"min": [-51.2, -51.2, None], "max": [51.2, 51.2, None]},
        img_norm_cfg=dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
        affine_trafo_config=affine_trafo_config,
        perform_3d_augmentation=True,
        bev_bboxes_trafo_config=bev_bboxes_trafo_config,
        use_reproducible_seed=True,
        randomize_sample_selection=use_randomizations,
        use_input_iterable=True,
        max_iterations_input_callable=num_iters_per_epoch * num_epochs,
    )

    config_as_namespace = convert_to_namespace(config)
    return config_as_namespace


def add_stream_petr_config_to_arguments(config, args):
    config = copy.deepcopy(config)
    config.pipeline_config.num_workers = args.num_workers
    config.pipeline_config.num_threads = args.num_threads
    config.pipeline_config.prefetch_queue_depth = args.prefetch_queue_depth
    config.convert_format_for_training = bool(args.convert_format_for_training)
    return config
