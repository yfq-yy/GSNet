# Copyright 2021 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=invalid-name
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import Masternet
    from evolution_search import get_latency
except ImportError:
    print('fail to import Masternet, evolution_search')


def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1, help='number of instances in one mini-batch.')
    parser.add_argument('--input_image_size', type=int, default=224,
                        help='resolution of input image, usually 32 for CIFAR and 224 for ImageNet.')
    parser.add_argument('--repeat_times', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=1000,
                        help='number of classes')
    parser.add_argument('--plain_structure', type=str, default=None,
                        help='model structure str')
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt

if __name__ == '__main__':
    opt = parse_cmd_options(sys.argv)

    gpu = opt.gpu

    any_plain_net = Masternet.MasterNet

    the_latency = get_latency(any_plain_net, opt.plain_structure, gpu, opt)

    print("latency: %s" % (the_latency))
