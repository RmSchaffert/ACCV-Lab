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


def add_object_detection_2d_arguments_to_parser(parser):
    parser = copy.deepcopy(parser)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
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
        default=8,
        metavar="T",
        help="Number of (CPU) threads to be used in DALI CPU & mixed operators",
    )
    parser.add_argument(
        "--prefetch_queue_depth",
        type=int,
        default=1,
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
    # parser.add_argument("--use_single_images", type=int, default=0, metavar="M", help="Whether to treat single images as samples (otherwise, one sample contains the images of all 6 cameras of the vehicle)")
    parser.add_argument(
        "--pipeline_use_gpu",
        type=int,
        default=1,
        metavar="G",
        help="Whether to use GPU for pre-processing. Note that this does not apply to all pre-processing operations as small operations are more efficient on the CPU",
    )
    parser.add_argument(
        "--hardware_decoder_load",
        type=float,
        default=0.65,
        metavar="D",
        help="Fraction of the decoding work which is performed by hardware decoder units (see parameters hw_decoder_load of fn.decoders.image nvidia.dali.fn.decoders.image)",
    )
    parser.add_argument(
        "--pipeline_use_hw_decoders",
        type=int,
        default=1,
        metavar="C",
        help="Whether to use GPU hardware decoders when decoding the input images",
    )
    parser.add_argument(
        "--use_single_images",
        type=int,
        default=1,
        metavar="M",
        help="Whether to treat single images as samples (otherwise, one sample contains the images of all 6 cameras of the vehicle)",
    )
    return parser


def get_default_object_detection_2d_pipeline_config(
    num_epochs, num_iters_per_epoch, use_randomizations=False, use_single_images=True
):

    def convert_to_namespace(as_dict):
        for k, v in as_dict.items():
            if isinstance(v, dict):
                as_dict[k] = convert_to_namespace(v)
        res = Namespace(**as_dict)
        return res

    if use_randomizations:
        pass
    else:
        pass

    config = dict(
        # Pipeline configuration
        pipeline_config={
            "num_workers": 1,
            "num_threads": 8,
            "prefetch_queue_depth": 1,
            "pipeline_use_gpu": True,
        },
        # Decoder configuration
        decoder_config={
            "hw_decoder_load": 0.65,
            "use_device_mixed": True,
        },
        # Image processing configuration
        image_config={
            "output_hw": [450, 800],
            "normalize": True,
            "use_single_images": use_single_images,
        },
        # Heatmap configuration
        heatmap_config={
            "heatmap_hw": [225, 400],
            "max_num_objects": 100,
            "num_categories": 6,
            "max_radius": 20,
        },
        # Augmentation configuration
        augmentation_config={
            "flip_probability": 0.5,
            "scaling_probabilities": [0.7, 0.3],
            "scaling_ranges": [(0.6, 1.4), (2.0, 2.0)],
            "translation_ranges": [(-100, -100, 100, 100), (-300, -300, 300, 300)],
        },
        # General configuration
        use_reproducible_seed=True,
        use_input_iterable=True,
        max_iterations_input_callable=num_iters_per_epoch * num_epochs,
    )

    config_as_namespace = convert_to_namespace(config)
    return config_as_namespace


def add_object_detection_2d_config_to_arguments(config, args):
    config = copy.deepcopy(config)
    config.pipeline_config.num_workers = args.num_workers
    config.pipeline_config.prefetch_queue_depth = args.prefetch_queue_depth
    config.pipeline_config.pipeline_use_gpu = bool(args.pipeline_use_gpu)
    config.pipeline_config.hardware_decoder_load = args.hardware_decoder_load
    config.pipeline_config.pipeline_use_hw_decoders = bool(args.pipeline_use_hw_decoders)
    config.image_config.use_single_images = bool(args.use_single_images)
    return config
