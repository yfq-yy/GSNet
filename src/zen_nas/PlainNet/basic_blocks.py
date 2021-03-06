'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.

define all needed basic layer class
'''
# pylint: disable=W0613,too-many-lines,too-many-arguments
import os
import sys
import uuid
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from PlainNet import _get_right_parentheses_index_, _create_netblock_list_from_str_
except ImportError:
    print('fail to import zen_nas modules')


# pylint: disable=no-self-use,invalid-name
class PlainNetBasicBlockClass(nn.Module):
    """BasicBlock base class"""

    def __init__(self, in_channels=None, out_channels=None, stride=1, no_create=False, block_name=None, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.no_create = no_create
        self.block_name = block_name
        if self.block_name is None:
            self.block_name = f'uuid{uuid.uuid4().hex}'

    def forward(self, input_):
        """subclass implementation"""
        raise RuntimeError('Not implemented')

    def __str__(self):
        return type(self).__name__ + f'({self.in_channels},{self.out_channels},{self.stride})'

    def __repr__(self):
        return type(self).__name__ + f'({self.block_name}|{self.in_channels},{self.out_channels},{self.stride})'

    def get_output_resolution(self, input_resolution):
        """subclass implementation"""
        raise RuntimeError('Not implemented')

    def get_FLOPs(self, input_resolution):
        """subclass implementation"""
        raise RuntimeError('Not implemented')

    def get_model_size(self):
        """subclass implementation"""
        raise RuntimeError('Not implemented')

    def set_in_channels(self, channels):
        """subclass implementation"""
        raise RuntimeError('Not implemented')

    @classmethod
    def create_from_str(cls, struct_str, no_create=False, **kwargs):
        """ class method

            :param s (str): basicblock str
            :return cls instance
        """
        assert PlainNetBasicBlockClass.is_instance_from_str(struct_str)
        idx = _get_right_parentheses_index_(struct_str)
        assert idx is not None
        param_str = struct_str[len(cls.__name__ + '('):idx]

        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = f'uuid{uuid.uuid4().hex}'
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        param_str_split = param_str.split(',')
        in_channels = int(param_str_split[0])
        out_channels = int(param_str_split[1])
        stride = int(param_str_split[2])
        return cls(in_channels=in_channels, out_channels=out_channels, stride=stride,
                   block_name=tmp_block_name, no_create=no_create), struct_str[idx + 1:]

    @classmethod
    def is_instance_from_str(cls, struct_str):
        if struct_str.startswith(cls.__name__ + '(') and struct_str[-1] == ')':
            return True
        return False


class GhostConv(PlainNetBasicBlockClass):
    def __init__(self, in_channels=None, out_channels=None, kernel_size=1, ratio=2, dw_size=3, 
                stride=1, copy_from=None, relu=None, no_create=False, **kwargs):
        super().__init__(**kwargs)
        self.no_create = no_create
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dw_size = dw_size
        self.relu = relu
        self.init_channels = math.ceil(out_channels / ratio)
        self.new_channels = self.init_channels * (ratio - 1)

        if no_create or self.in_channels == 0 or self.out_channels == 0 or self.kernel_size == 0 \
                or self.stride == 0:
            return

        self.primary_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.init_channels, self.kernel_size, self.stride, self.kernel_size//2, bias=False),
            # nn.BatchNorm2d(self.init_channels),
            # nn.ReLU(inplace=True) if self.relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(self.init_channels, self.new_channels, self.dw_size, 1, self.dw_size//2, groups=self.init_channels, bias=False),
            # nn.BatchNorm2d(self.new_channels),
            # nn.ReLU(inplace=True) if self.relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        t = self.out_channels
        return out[:, :self.out_channels, :, :]

    def __str__(self):
        return type(self).__name__ + f'({self.in_channels},{self.out_channels},{self.kernel_size},'\
                                        f'{self.stride})'

    def __repr__(self):
        return type(self).__name__ + f'({self.block_name}|{self.in_channels},{self.out_channels},'\
                                        f'{self.kernel_size},{self.stride})'

    def get_output_resolution(self, input_resolution):
        return input_resolution // self.stride

    def get_FLOPs(self, input_resolution):
        return self.in_channels * self.init_channels * self.kernel_size ** 2 * \
            input_resolution ** 2 // self.stride ** 2  + \
               self.init_channels * self.new_channels * self.dw_size ** 2 * \
            input_resolution ** 2 // self.init_channels

    def get_model_size(self):
        return self.in_channels * self.init_channels * self.kernel_size ** 2  + \
                self.init_channels * self.new_channels * self.dw_size ** 2 // self.init_channels
                
    def set_in_channels(self, channels):
        self.in_channels = channels
        if not self.no_create:
            self.primary_conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.init_channels, self.kernel_size, self.stride, self.kernel_size//2, bias=False),
                # nn.BatchNorm2d(self.init_channels),
                # nn.ReLU(inplace=True) if self.relu else nn.Sequential(),
            )

            self.cheap_operation = nn.Sequential(
                nn.Conv2d(self.init_channels, self.new_channels, self.dw_size, 1, self.dw_size//2, groups=self.init_channels, bias=False),
                # nn.BatchNorm2d(self.new_channels),
                # nn.ReLU(inplace=True) if self.relu else nn.Sequential(),
            )
            self.primary_conv.train()
            self.cheap_operation.train()
            self.primary_conv.requires_grad_(True)
            self.cheap_operation.requires_grad_(True)

    @classmethod
    def create_from_str(cls, struct_str, no_create=False, **kwargs):
        assert cls.is_instance_from_str(struct_str)
        idx = _get_right_parentheses_index_(struct_str)
        assert idx is not None
        param_str = struct_str[len(cls.__name__ + '('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = f'uuid{uuid.uuid4().hex}'
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        split_str = param_str.split(',')
        in_channels = int(split_str[0])
        out_channels = int(split_str[1])
        kernel_size = int(split_str[2])
        stride = int(split_str[3])
        #relu = True if str(split_str[4]) == 'relu' else False
        return GhostConv(in_channels=in_channels, out_channels=out_channels,
                   kernel_size=kernel_size, stride=stride, no_create=no_create,
                   block_name=tmp_block_name), struct_str[idx + 1:]

class GhostShuffleBlock(PlainNetBasicBlockClass):
    def __init__(self, block_list, in_channels=None, stride=None,
                 kernel_size=None, no_create=False, **kwargs):
        super().__init__(**kwargs)
        self.block_list = block_list
        self.stride = stride
        self.no_create = no_create
        self.group = 2 # group = 2
        if not no_create:
            self.module_list = nn.ModuleList(block_list)

        if in_channels is None:
            if self.stride == 1:
                self.in_channels = block_list[0].in_channels * 2
            else:
                self.in_channels = block_list[0].in_channels
        else:
            self.in_channels = in_channels * 2

        if self.stride == 1:
            self.mid_channels = self.in_channels // 2
        else:
            self.mid_channels = self.in_channels
        self.out_channels = block_list[-1].out_channels * 2

        self.kernel_size = kernel_size

        if self.stride is None:
            tmp_input_res = 1024
            tmp_output_res = self.get_output_resolution(tmp_input_res)
            self.stride = tmp_input_res // tmp_output_res
        assert self.stride in [1, 2]

        # Depth-wise convolution
        self.proj = None
        if self.stride > 1:
            self.proj = nn.Sequential(
                    nn.Conv2d(self.mid_channels, self.mid_channels, self.kernel_size, stride=self.stride,
                        padding=(self.kernel_size-1)//2, groups=self.mid_channels, bias=False),
                    nn.BatchNorm2d(self.mid_channels),
                    GhostConv(self.mid_channels, self.out_channels // 2),
                    nn.BatchNorm2d(self.out_channels // 2),
                    nn.ReLU(),
            )
        else:
            self.proj = nn.Sequential(
                    GhostConv(self.mid_channels, self.out_channels // 2),
                    nn.BatchNorm2d(self.out_channels // 2),
                    nn.ReLU(),
            )            
        # self.afterconcat = nn.Sequential(
        #     GhostConv(self.out_channels, self.out_channels),
        #     nn.BatchNorm2d(self.out_channels),
        #     nn.ReLU(),
        # )

    def forward(self, x):
        if len(self.block_list) == 0:
            return x
        
        if self.stride == 1:
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            output = x2
            for inner_block in self.block_list:
                output = inner_block(output)
            x_proj = self.proj(x1)
            x = torch.cat((x_proj, output), 1)
            x = self.channel_shuffle(x)
        else:
            x_proj = x
            output = x
            for inner_block in self.block_list:
                output = inner_block(output)
            x = torch.cat((self.proj(x_proj), output), 1)

        return x

    # # shufflenetv2
    # def channel_shuffle(self, x):
    #     batchsize, num_channels, height, width = x.data.size()
    #     assert (num_channels % 4 == 0)
    #     x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    #     x = x.permute(1, 0, 2)
    #     x = x.reshape(2, -1, num_channels // 2, height, width)
    #     return x[0], x[1]

    # shufflenetv1
    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group

        x = x.reshape(batchsize, group_channels, self.group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)

        return x

    def __str__(self):
        block_str = f'GhostShuffleBlock({self.in_channels},{self.stride},'
        for inner_block in self.block_list:
            block_str += str(inner_block)

        block_str += ')'
        return block_str

    def __repr__(self):
        block_str = f'GhostShuffleBlock({self.block_name}|{self.in_channels},{self.stride},'
        for inner_block in self.block_list:
            block_str += str(inner_block)

        block_str += ')'
        return block_str

    def get_output_resolution(self, input_resolution):
        the_res = input_resolution
        for the_block in self.block_list:
            the_res = the_block.get_output_resolution(the_res)

        return the_res

    def get_FLOPs(self, input_resolution):
        the_res = input_resolution
        the_flops = 0
        for the_block in self.block_list:
            the_flops += the_block.get_FLOPs(the_res)
            the_res = the_block.get_output_resolution(the_res)

        if self.stride > 1:
            the_flops += self.mid_channels * self.mid_channels * self.kernel_size ** 2 * (the_res / self.stride) ** 2 + \
                self.mid_channels * (self.out_channels // 4) * the_res ** 2  + \
                (self.out_channels // 4)  * 3 ** 2 * the_res ** 2 
        else:
            the_flops += self.mid_channels * (self.out_channels // 4) * the_res ** 2  + \
                (self.out_channels // 4) * 3 ** 2 * the_res ** 2

        # the_flops += self.out_channels / 2 * self.out_channels / 2 * the_res ** 2  + \
        #        self.out_channels / 2 * self.out_channels / 2 * 3 ** 2 * \
        #         the_res ** 2 // self.out_channels

        return the_flops
        
    def get_model_size(self):
        the_size = 0
        for the_block in self.block_list:
            the_size += the_block.get_model_size()

        if self.stride > 1:
            the_size += self.mid_channels * self.mid_channels * self.kernel_size ** 2 + \
                self.mid_channels * (self.out_channels // 4) + (self.out_channels // 4) * 3 ** 2
        else:
            the_size += self.mid_channels * (self.out_channels // 4) + (self.out_channels // 4) * 3 ** 2
        return the_size

    def set_in_channels(self, channels):
        self.in_channels = channels
        self.mid_channels = channels // 2
        if not self.no_create:
            if self.stride > 1:
                self.proj = nn.Sequential(
                        nn.Conv2d(self.mid_channels, self.mid_channels, self.kernel_size, stride=self.stride,
                            padding=(self.kernel_size-1)//2, groups=self.mid_channels, bias=False),
                        nn.BatchNorm2d(self.mid_channels),
                        GhostConv(self.mid_channels, self.out_channels // 2),
                        nn.BatchNorm2d(self.out_channels // 2),
                        HSwish(),
                )
            else:
                self.proj = nn.Sequential(
                        GhostConv(self.mid_channels, self.out_channels // 2),
                        nn.BatchNorm2d(self.out_channels // 2),
                        HSwish(),
                )   
            self.proj.train()
            self.proj.requires_grad_(True)

    @classmethod
    def create_from_str(cls, struct_str, no_create=False, **kwargs):
        assert cls.is_instance_from_str(struct_str)
        idx = _get_right_parentheses_index_(struct_str)
        assert idx is not None
        the_stride = None
        param_str = struct_str[len(cls.__name__ + '('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = f'uuid{uuid.uuid4().hex}'
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        first_comma_index = param_str.find(',')
        # cannot parse in_channels, missing, use default
        if first_comma_index < 0 or not param_str[0:first_comma_index].isdigit():
            in_channels = None
            the_block_list, remaining_s = _create_netblock_list_from_str_(param_str, no_create=no_create)
        else:
            in_channels = int(param_str[0:first_comma_index])
            param_str = param_str[first_comma_index + 1:]
            second_comma_index = param_str.find(',')
            if second_comma_index < 0 or not param_str[0:second_comma_index].isdigit():
                the_block_list, remaining_s = _create_netblock_list_from_str_(param_str, no_create=no_create)
            else:
                the_stride = int(param_str[0:second_comma_index])
                param_str = param_str[second_comma_index + 1:]
                the_block_list, remaining_s = _create_netblock_list_from_str_(param_str, no_create=no_create)

        assert len(remaining_s) == 0
        if the_block_list is None or len(the_block_list) == 0:
            return None, struct_str[idx + 1:]
        for block in the_block_list:
            if isinstance(block, ConvDW):
                kernel_size = block.kernel_size
        return cls(block_list=the_block_list, in_channels=in_channels,
                   kernel_size=kernel_size, stride=the_stride, no_create=no_create,
                   block_name=tmp_block_name), struct_str[idx + 1:]

class HSwish(nn.Module):

	def __init__(self):
		super(HSwish, self).__init__()

	def forward(self, inputs):
		clip = torch.clamp(inputs + 3, 0, 6) / 6
		return inputs * clip

class HS(PlainNetBasicBlockClass):

    def __init__(self, out_channels, no_create=False, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = out_channels
        self.out_channels = out_channels
        self.no_create = no_create

    def forward(self, inputs):
        clip = torch.clamp(inputs + 3, 0, 6) / 6
        return inputs * clip

    def __str__(self):
        return f'HS({self.out_channels})'

    def __repr__(self):
        return f'HS({self.block_name}|{self.out_channels})'

    def get_output_resolution(self, input_resolution):
        return input_resolution

    def get_FLOPs(self, input_resolution):
        return 0

    def get_model_size(self):
        return 0

    def set_in_channels(self, channels):
        self.in_channels = channels
        self.out_channels = channels

    @classmethod
    def create_from_str(cls, struct_str, no_create=False, **kwargs):
        assert HS.is_instance_from_str(struct_str)
        idx = _get_right_parentheses_index_(struct_str)
        assert idx is not None
        param_str = struct_str[len('HS('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = f'uuid{uuid.uuid4().hex}'
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        out_channels = int(param_str)
        return HS(out_channels=out_channels, no_create=no_create, block_name=tmp_block_name), struct_str[idx + 1:]

class AdaptiveAvgPool(PlainNetBasicBlockClass):
    """Adaptive average pool layer"""

    def __init__(self, out_channels, output_size, no_create=False, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = out_channels
        self.out_channels = out_channels
        self.output_size = output_size
        self.no_create = no_create
        if not no_create:
            self.netblock = nn.AdaptiveAvgPool2d(output_size=(self.output_size, self.output_size))

    def forward(self, input_):
        return self.netblock(input_)

    def __str__(self):
        return type(self).__name__ + f'({self.out_channels // self.output_size**2},{self.output_size})'

    def __repr__(self):
        return type(self).__name__ + f'({self.block_name}|{self.out_channels // self.output_size ** 2},'\
                                        f'{self.output_size})'

    def get_output_resolution(self, input_resolution):
        return self.output_size

    def get_FLOPs(self, input_resolution):
        return 0

    def get_model_size(self):
        return 0

    def set_in_channels(self, channels):
        self.in_channels = channels
        self.out_channels = channels

    @classmethod
    def create_from_str(cls, struct_str, no_create=False, **kwargs):
        assert AdaptiveAvgPool.is_instance_from_str(struct_str)
        idx = _get_right_parentheses_index_(struct_str)
        assert idx is not None
        param_str = struct_str[len('AdaptiveAvgPool('):idx]

        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = f'uuid{uuid.uuid4().hex}'
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        param_str_split = param_str.split(',')
        out_channels = int(param_str_split[0])
        output_size = int(param_str_split[1])
        return AdaptiveAvgPool(out_channels=out_channels, output_size=output_size,
                               block_name=tmp_block_name, no_create=no_create), struct_str[idx + 1:]


class BN(PlainNetBasicBlockClass):

    def __init__(self, out_channels=None, copy_from=None, no_create=False, **kwargs):
        super().__init__(**kwargs)
        self.no_create = no_create

        if copy_from is not None:
            assert isinstance(copy_from, nn.BatchNorm2d)
            self.in_channels = copy_from.weight.shape[0]
            self.out_channels = copy_from.weight.shape[0]
            assert out_channels is None or out_channels == self.out_channels
            self.netblock = copy_from

        else:
            self.in_channels = out_channels
            self.out_channels = out_channels
            if no_create:
                return
            self.netblock = nn.BatchNorm2d(num_features=self.out_channels)

    def forward(self, input_):
        return self.netblock(input_)

    def __str__(self):
        return f'BN({self.out_channels})'

    def __repr__(self):
        return f'BN({self.block_name}|{self.out_channels})'

    def get_output_resolution(self, input_resolution):
        return input_resolution

    def get_FLOPs(self, input_resolution):
        return input_resolution ** 2 * self.out_channels

    def get_model_size(self):
        return self.out_channels

    def set_in_channels(self, channels):
        self.in_channels = channels
        self.out_channels = channels
        if not self.no_create:
            self.netblock = nn.BatchNorm2d(num_features=self.out_channels)
            self.netblock.train()
            self.netblock.requires_grad_(True)

    @classmethod
    def create_from_str(cls, struct_str, no_create=False, **kwargs):
        assert BN.is_instance_from_str(struct_str)
        idx = _get_right_parentheses_index_(struct_str)
        assert idx is not None
        param_str = struct_str[len('BN('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = f'uuid{uuid.uuid4().hex}'
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]
        out_channels = int(param_str)
        return BN(out_channels=out_channels, block_name=tmp_block_name, no_create=no_create), struct_str[idx + 1:]


# pylint: disable=too-many-instance-attributes
class ConvKX(PlainNetBasicBlockClass):
    """convolutional layer"""

    def __init__(self, in_channels=None, out_channels=None, kernel_size=None, stride=None, groups=1, copy_from=None,
                 no_create=False, **kwargs):
        super().__init__(**kwargs)
        self.no_create = no_create

        if copy_from is not None:
            assert isinstance(copy_from, nn.Conv2d)
            self.in_channels = copy_from.in_channels
            self.out_channels = copy_from.out_channels
            self.kernel_size = copy_from.kernel_size[0]
            self.stride = copy_from.stride[0]
            self.groups = copy_from.groups
            assert in_channels is None or in_channels == self.in_channels
            assert out_channels is None or out_channels == self.out_channels
            assert kernel_size is None or kernel_size == self.kernel_size
            assert stride is None or stride == self.stride
            self.netblock = copy_from
        else:
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.stride = stride
            self.groups = groups
            self.kernel_size = kernel_size
            self.padding = (self.kernel_size - 1) // 2
            if no_create or self.in_channels == 0 or self.out_channels == 0 or \
                    self.kernel_size == 0 or self.stride == 0:
                return
            self.netblock = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                      kernel_size=self.kernel_size, stride=self.stride,
                                      padding=self.padding, bias=False, groups=self.groups)

    def forward(self, input_):
        return self.netblock(input_)

    def __str__(self):
        return type(self).__name__ + f'({self.in_channels},{self.out_channels},{self.kernel_size},'\
                                        f'{self.stride})'

    def __repr__(self):
        return type(self).__name__ + f'({self.block_name}|{self.in_channels},{self.out_channels},'\
                                        f'{self.kernel_size},{self.stride})'

    def get_output_resolution(self, input_resolution):
        return input_resolution // self.stride

    def get_FLOPs(self, input_resolution):
        return self.in_channels * self.out_channels * self.kernel_size ** 2 * \
            input_resolution ** 2 // self.stride ** 2 // self.groups

    def get_model_size(self):
        return self.in_channels * self.out_channels * self.kernel_size ** 2 // self.groups

    def set_in_channels(self, channels):
        self.in_channels = channels
        if not self.no_create:
            self.netblock = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                      kernel_size=self.kernel_size, stride=self.stride,
                                      padding=self.padding, bias=False)
            self.netblock.train()
            self.netblock.requires_grad_(True)

    @classmethod
    def create_from_str(cls, struct_str, no_create=False, **kwargs):
        assert cls.is_instance_from_str(struct_str)
        idx = _get_right_parentheses_index_(struct_str)
        assert idx is not None
        param_str = struct_str[len(cls.__name__ + '('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = f'uuid{uuid.uuid4().hex}'
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        split_str = param_str.split(',')
        in_channels = int(split_str[0])
        out_channels = int(split_str[1])
        kernel_size = int(split_str[2])
        stride = int(split_str[3])
        return cls(in_channels=in_channels, out_channels=out_channels,
                   kernel_size=kernel_size, stride=stride, no_create=no_create,
                   block_name=tmp_block_name), struct_str[idx + 1:]


class ConvDW(PlainNetBasicBlockClass):
    """depthwise convolutional layer"""

    def __init__(self, out_channels=None, kernel_size=None, stride=None, copy_from=None,
                 no_create=False, **kwargs):
        super().__init__(**kwargs)
        self.no_create = no_create

        if copy_from is not None:
            assert isinstance(copy_from, nn.Conv2d)
            self.in_channels = copy_from.in_channels
            self.out_channels = copy_from.out_channels
            self.kernel_size = copy_from.kernel_size[0]
            self.stride = copy_from.stride[0]
            assert self.in_channels == self.out_channels
            assert out_channels is None or out_channels == self.out_channels
            assert kernel_size is None or kernel_size == self.kernel_size
            assert stride is None or stride == self.stride

            self.netblock = copy_from
        else:

            self.in_channels = out_channels
            self.out_channels = out_channels
            self.stride = stride
            self.kernel_size = kernel_size

            self.padding = (self.kernel_size - 1) // 2
            if no_create or self.in_channels == 0 or self.out_channels == 0 or self.kernel_size == 0 \
                    or self.stride == 0:
                return
            self.netblock = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                      kernel_size=self.kernel_size, stride=self.stride,
                                      padding=self.padding, bias=False, groups=self.in_channels)

    def forward(self, input_):
        return self.netblock(input_)

    def __str__(self):
        return f'ConvDW({self.out_channels},{self.kernel_size},{self.stride})'

    def __repr__(self):
        return f'ConvDW({self.block_name}|{self.out_channels},{self.kernel_size},{self.stride})'

    def get_output_resolution(self, input_resolution):
        return input_resolution // self.stride

    def get_FLOPs(self, input_resolution):
        return self.out_channels * self.kernel_size ** 2 * input_resolution ** 2 // self.stride ** 2

    def get_model_size(self):
        return self.out_channels * self.kernel_size ** 2

    def set_in_channels(self, channels):
        self.in_channels = channels
        self.out_channels = self.in_channels
        if not self.no_create:
            self.netblock = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                      kernel_size=self.kernel_size, stride=self.stride,
                                      padding=self.padding, bias=False, groups=self.in_channels)
            self.netblock.train()
            self.netblock.requires_grad_(True)

    @classmethod
    def create_from_str(cls, struct_str, no_create=False, **kwargs):
        assert ConvDW.is_instance_from_str(struct_str)
        idx = _get_right_parentheses_index_(struct_str)
        assert idx is not None
        param_str = struct_str[len('ConvDW('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = f'uuid{uuid.uuid4().hex}'
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        split_str = param_str.split(',')
        out_channels = int(split_str[0])
        kernel_size = int(split_str[1])
        stride = int(split_str[2])
        return ConvDW(out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, no_create=no_create,
                      block_name=tmp_block_name), struct_str[idx + 1:]


class ConvKXG2(ConvKX):
    """convolution group 2"""

    def __init__(self, in_channels=None, out_channels=None, kernel_size=None, stride=None, copy_from=None,
                 no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, copy_from=copy_from, no_create=no_create,
                         groups=2, **kwargs)


class ConvKXG4(ConvKX):
    """convolution group 4"""

    def __init__(self, in_channels=None, out_channels=None, kernel_size=None, stride=None, copy_from=None,
                 no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, copy_from=copy_from, no_create=no_create,
                         groups=4, **kwargs)


class ConvKXG8(ConvKX):
    """convolution group 8"""

    def __init__(self, in_channels=None, out_channels=None, kernel_size=None, stride=None, copy_from=None,
                 no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, copy_from=copy_from, no_create=no_create,
                         groups=8, **kwargs)


class ConvKXG16(ConvKX):
    """convolution group 16"""

    def __init__(self, in_channels=None, out_channels=None, kernel_size=None, stride=None, copy_from=None,
                 no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, copy_from=copy_from, no_create=no_create,
                         groups=16, **kwargs)


class ConvKXG32(ConvKX):
    """convolution group 32"""

    def __init__(self, in_channels=None, out_channels=None, kernel_size=None, stride=None, copy_from=None,
                 no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, copy_from=copy_from, no_create=no_create,
                         groups=32, **kwargs)


class Flatten(PlainNetBasicBlockClass):
    """flatten layer"""

    def __init__(self, out_channels, no_create=False, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = out_channels
        self.out_channels = out_channels
        self.no_create = no_create

    def forward(self, input_):
        return torch.flatten(input_, 1)

    def __str__(self):
        return f'Flatten({self.out_channels})'

    def __repr__(self):
        return f'Flatten({self.block_name}|{self.out_channels})'

    def get_output_resolution(self, input_resolution):
        return 1

    def get_FLOPs(self, input_resolution):
        return 0

    def get_model_size(self):
        return 0

    def set_in_channels(self, channels):
        self.in_channels = channels
        self.out_channels = channels

    @classmethod
    def create_from_str(cls, struct_str, no_create=False, **kwargs):
        assert Flatten.is_instance_from_str(struct_str)
        idx = _get_right_parentheses_index_(struct_str)
        assert idx is not None
        param_str = struct_str[len('Flatten('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = f'uuid{uuid.uuid4().hex}'
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        out_channels = int(param_str)
        return Flatten(out_channels=out_channels, no_create=no_create, block_name=tmp_block_name), struct_str[idx + 1:]


class Linear(PlainNetBasicBlockClass):
    """Linear layer"""

    def __init__(self, in_channels=None, out_channels=None, bias=True, copy_from=None,
                 no_create=False, **kwargs):
        super().__init__(**kwargs)
        self.no_create = no_create

        if copy_from is not None:
            assert isinstance(copy_from, nn.Linear)
            self.in_channels = copy_from.weight.shape[1]
            self.out_channels = copy_from.weight.shape[0]
            self.use_bias = copy_from.bias is not None
            assert in_channels is None or in_channels == self.in_channels
            assert out_channels is None or out_channels == self.out_channels

            self.netblock = copy_from
        else:
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.use_bias = bias
            if not no_create:
                self.netblock = nn.Linear(self.in_channels, self.out_channels,
                                          bias=self.use_bias)

    def forward(self, input_):
        return self.netblock(input_)

    def __str__(self):
        return f'Linear({self.in_channels},{self.out_channels},{int(self.use_bias)})'

    def __repr__(self):
        return f'Linear({self.block_name}|{self.in_channels},{self.out_channels},{int(self.use_bias)})'

    def get_output_resolution(self, input_resolution):
        assert input_resolution == 1
        return 1

    def get_FLOPs(self, input_resolution):
        return self.in_channels * self.out_channels

    def get_model_size(self):
        return self.in_channels * self.out_channels + int(self.use_bias)

    def set_in_channels(self, channels):
        self.in_channels = channels
        if not self.no_create:
            self.netblock = nn.Linear(self.in_channels, self.out_channels,
                                      bias=self.use_bias)
            self.netblock.train()
            self.netblock.requires_grad_(True)

    @classmethod
    def create_from_str(cls, struct_str, no_create=False, **kwargs):
        assert Linear.is_instance_from_str(struct_str)
        idx = _get_right_parentheses_index_(struct_str)
        assert idx is not None
        param_str = struct_str[len('Linear('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = f'uuid{uuid.uuid4().hex}'
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        split_str = param_str.split(',')
        in_channels = int(split_str[0])
        out_channels = int(split_str[1])
        use_bias = int(split_str[2])

        return Linear(in_channels=in_channels, out_channels=out_channels, bias=use_bias == 1,
                      block_name=tmp_block_name, no_create=no_create), struct_str[idx + 1:]


class MaxPool(PlainNetBasicBlockClass):
    """maxpool layer"""

    def __init__(self, out_channels, kernel_size, stride, no_create=False, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = out_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        self.no_create = no_create
        if not no_create:
            self.netblock = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    def forward(self, input_):
        return self.netblock(input_)

    def __str__(self):
        return f'MaxPool({self.out_channels},{self.kernel_size},{self.stride})'

    def __repr__(self):
        return f'MaxPool({self.block_name}|{self.out_channels},{self.kernel_size},{self.stride})'

    def get_output_resolution(self, input_resolution):
        return input_resolution // self.stride

    def get_FLOPs(self, input_resolution):
        return 0

    def get_model_size(self):
        return 0

    def set_in_channels(self, channels):
        self.in_channels = channels
        self.out_channels = channels
        if not self.no_create:
            self.netblock = nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    @classmethod
    def create_from_str(cls, struct_str, no_create=False, **kwargs):
        assert MaxPool.is_instance_from_str(struct_str)
        idx = _get_right_parentheses_index_(struct_str)
        assert idx is not None
        param_str = struct_str[len('MaxPool('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = f'uuid{uuid.uuid4().hex}'
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        param_str_split = param_str.split(',')
        out_channels = int(param_str_split[0])
        kernel_size = int(param_str_split[1])
        stride = int(param_str_split[2])
        return MaxPool(out_channels=out_channels, kernel_size=kernel_size, stride=stride, no_create=no_create,
                       block_name=tmp_block_name), struct_str[idx + 1:]


class Sequential(PlainNetBasicBlockClass):
    """sequential module"""

    def __init__(self, block_list, no_create=False, **kwargs):
        super().__init__(**kwargs)
        self.block_list = block_list
        if not no_create:
            self.module_list = nn.ModuleList(block_list)
        self.in_channels = block_list[0].in_channels
        self.out_channels = block_list[-1].out_channels
        self.no_create = no_create
        res = 1024
        for block in self.block_list:
            res = block.get_output_resolution(res)
        self.stride = 1024 // res

    def forward(self, input_):
        output = input_
        for inner_block in self.block_list:
            output = inner_block(output)
        return output

    def __str__(self):
        block_str = 'Sequential('
        for inner_block in self.block_list:
            block_str += str(inner_block)
        block_str += ')'
        return block_str

    def __repr__(self):
        return str(self)

    def get_output_resolution(self, input_resolution):
        the_res = input_resolution
        for the_block in self.block_list:
            the_res = the_block.get_output_resolution(the_res)
        return the_res

    def get_FLOPs(self, input_resolution):
        the_res = input_resolution
        the_flops = 0
        for the_block in self.block_list:
            the_flops += the_block.get_FLOPs(the_res)
            the_res = the_block.get_output_resolution(the_res)
        return the_flops

    def get_model_size(self):
        the_size = 0
        for the_block in self.block_list:
            the_size += the_block.get_model_size()

        return the_size

    def set_in_channels(self, channels):
        self.in_channels = channels
        if len(self.block_list) == 0:
            self.out_channels = channels
            return

        self.block_list[0].set_in_channels(channels)
        last_channels = self.block_list[0].out_channels
        if len(self.block_list) >= 2 and isinstance(self.block_list[1], BN):
            self.block_list[1].set_in_channels(last_channels)

    @classmethod
    def create_from_str(cls, struct_str, no_create=False, **kwargs):
        assert Sequential.is_instance_from_str(struct_str)
        the_right_paraen_idx = _get_right_parentheses_index_(struct_str)
        param_str = struct_str[len('Sequential(') + 1:the_right_paraen_idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = f'uuid{uuid.uuid4().hex}'
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        the_block_list, remaining_s = _create_netblock_list_from_str_(param_str, no_create=no_create)
        assert len(remaining_s) == 0
        if the_block_list is None or len(the_block_list) == 0:
            return None, ''
        return Sequential(block_list=the_block_list, no_create=no_create, block_name=tmp_block_name), ''


class MultiSumBlock(PlainNetBasicBlockClass):
    """multiple sum block"""

    def __init__(self, block_list, no_create=False, **kwargs):
        super().__init__(**kwargs)
        self.block_list = block_list
        if not no_create:
            self.module_list = nn.ModuleList(block_list)
        self.in_channels = np.max([x.in_channels for x in block_list])
        self.out_channels = np.max([x.out_channels for x in block_list])
        self.no_create = no_create

        res = 1024
        res = self.block_list[0].get_output_resolution(res)
        self.stride = 1024 // res

    def forward(self, input_):
        output = self.block_list[0](input_)
        for inner_block in self.block_list[1:]:
            output2 = inner_block(input_)
            output = output + output2
        return output

    def __str__(self):
        block_str = f'MultiSumBlock({self.block_name}|'
        for inner_block in self.block_list:
            block_str += str(inner_block) + ';'
        block_str = block_str[:-1]
        block_str += ')'
        return block_str

    def __repr__(self):
        return str(self)

    def get_output_resolution(self, input_resolution):
        the_res = self.block_list[0].get_output_resolution(input_resolution)
        for the_block in self.block_list:
            assert the_res == the_block.get_output_resolution(input_resolution)

        return the_res

    def get_FLOPs(self, input_resolution):
        the_flops = 0
        for the_block in self.block_list:
            the_flops += the_block.get_FLOPs(input_resolution)

        return the_flops

    def get_model_size(self):
        the_size = 0
        for the_block in self.block_list:
            the_size += the_block.get_model_size()

        return the_size

    def set_in_channels(self, channels):
        self.in_channels = channels
        for the_block in self.block_list:
            the_block.set_in_channels(channels)

    @classmethod
    def create_from_str(cls, struct_str, no_create=False, **kwargs):
        assert MultiSumBlock.is_instance_from_str(struct_str)
        idx = _get_right_parentheses_index_(struct_str)
        assert idx is not None
        param_str = struct_str[len('MultiSumBlock('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = f'uuid{uuid.uuid4().hex}'
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        the_s = param_str

        the_block_list = []
        while len(the_s) > 0:
            tmp_block_list, remaining_s = _create_netblock_list_from_str_(the_s, no_create=no_create)
            the_s = remaining_s
            if tmp_block_list is None:
                pass
            elif len(tmp_block_list) == 1:
                the_block_list.append(tmp_block_list[0])
            else:
                the_block_list.append(Sequential(block_list=tmp_block_list, no_create=no_create))

        if len(the_block_list) == 0:
            return None, struct_str[idx + 1:]

        return MultiSumBlock(block_list=the_block_list, block_name=tmp_block_name,
                             no_create=no_create), struct_str[idx + 1:]


class MultiCatBlock(PlainNetBasicBlockClass):
    """multiple block concatenation"""

    def __init__(self, block_list, no_create=False, **kwargs):
        super().__init__(**kwargs)
        self.block_list = block_list
        if not no_create:
            self.module_list = nn.ModuleList(block_list)
        self.in_channels = np.max([x.in_channels for x in block_list])
        self.out_channels = np.sum([x.out_channels for x in block_list])
        self.no_create = no_create

        res = 1024
        res = self.block_list[0].get_output_resolution(res)
        self.stride = 1024 // res

    def forward(self, input_):
        output_list = []
        for inner_block in self.block_list:
            output = inner_block(input_)
            output_list.append(output)

        return torch.cat(output_list, dim=1)

    def __str__(self):
        block_str = f'MultiCatBlock({self.block_name}|'
        for inner_block in self.block_list:
            block_str += str(inner_block) + ';'

        block_str = block_str[:-1]
        block_str += ')'
        return block_str

    def __repr__(self):
        return str(self)

    def get_output_resolution(self, input_resolution):
        """return single block's output size"""
        the_res = self.block_list[0].get_output_resolution(input_resolution)
        for the_block in self.block_list:
            assert the_res == the_block.get_output_resolution(input_resolution)

        return the_res

    def get_FLOPs(self, input_resolution):
        the_flops = 0
        for the_block in self.block_list:
            the_flops += the_block.get_FLOPs(input_resolution)

        return the_flops

    def get_model_size(self):
        the_size = 0
        for the_block in self.block_list:
            the_size += the_block.get_model_size()

        return the_size

    def set_in_channels(self, channels):
        self.in_channels = channels
        for the_block in self.block_list:
            the_block.set_in_channels(channels)
        self.out_channels = np.sum([x.out_channels for x in self.block_list])

    @classmethod
    def create_from_str(cls, struct_str, no_create=False, **kwargs):
        assert MultiCatBlock.is_instance_from_str(struct_str)
        idx = _get_right_parentheses_index_(struct_str)
        assert idx is not None
        param_str = struct_str[len('MultiCatBlock('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = f'uuid{uuid.uuid4().hex}'
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        the_s = param_str

        the_block_list = []
        while len(the_s) > 0:
            tmp_block_list, remaining_s = _create_netblock_list_from_str_(the_s, no_create=no_create)
            the_s = remaining_s
            if tmp_block_list is None:
                pass
            elif len(tmp_block_list) == 1:
                the_block_list.append(tmp_block_list[0])
            else:
                the_block_list.append(Sequential(block_list=tmp_block_list, no_create=no_create))

        if len(the_block_list) == 0:
            return None, struct_str[idx + 1:]

        return MultiCatBlock(block_list=the_block_list, block_name=tmp_block_name,
                             no_create=no_create), struct_str[idx + 1:]


class RELU(PlainNetBasicBlockClass):
    """RELU layer"""

    def __init__(self, out_channels, no_create=False, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = out_channels
        self.out_channels = out_channels
        self.no_create = no_create

    def forward(self, input_):
        return F.relu(input_)

    def __str__(self):
        return f'RELU({self.out_channels})'

    def __repr__(self):
        return f'RELU({self.block_name}|{self.out_channels})'

    def get_output_resolution(self, input_resolution):
        return input_resolution

    def get_FLOPs(self, input_resolution):
        return 0

    def get_model_size(self):
        return 0

    def set_in_channels(self, channels):
        self.in_channels = channels
        self.out_channels = channels

    @classmethod
    def create_from_str(cls, struct_str, no_create=False, **kwargs):
        assert RELU.is_instance_from_str(struct_str)
        idx = _get_right_parentheses_index_(struct_str)
        assert idx is not None
        param_str = struct_str[len('RELU('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = f'uuid{uuid.uuid4().hex}'
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        out_channels = int(param_str)
        return RELU(out_channels=out_channels, no_create=no_create, block_name=tmp_block_name), struct_str[idx + 1:]


class ResBlock(PlainNetBasicBlockClass):
    """ResBlock(in_channles, inner_blocks_str). If in_channels is missing, use block_list[0].in_channels as in_channels
    """

    def __init__(self, block_list, in_channels=None, stride=None, no_create=False, **kwargs):
        super().__init__(**kwargs)
        self.block_list = block_list
        self.stride = stride
        self.no_create = no_create
        if not no_create:
            self.module_list = nn.ModuleList(block_list)

        if in_channels is None:
            self.in_channels = block_list[0].in_channels
        else:
            self.in_channels = in_channels
        self.out_channels = block_list[-1].out_channels

        if self.stride is None:
            tmp_input_res = 1024
            tmp_output_res = self.get_output_resolution(tmp_input_res)
            self.stride = tmp_input_res // tmp_output_res

        self.proj = None
        if self.stride > 1 or self.in_channels != self.out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, 1, self.stride),
                nn.BatchNorm2d(self.out_channels),
            )

    def forward(self, input_):
        if len(self.block_list) == 0:
            return input_

        output = input_
        for inner_block in self.block_list:
            output = inner_block(output)

        if self.proj is not None:
            output = output + self.proj(input_)
        else:
            output = output + input_

        return output

    def __str__(self):
        block_str = f'ResBlock({self.in_channels},{self.stride},'
        for inner_block in self.block_list:
            block_str += str(inner_block)

        block_str += ')'
        return block_str

    def __repr__(self):
        block_str = f'ResBlock({self.block_name}|{self.in_channels},{self.stride},'
        for inner_block in self.block_list:
            block_str += str(inner_block)

        block_str += ')'
        return block_str

    def get_output_resolution(self, input_resolution):
        the_res = input_resolution
        for the_block in self.block_list:
            the_res = the_block.get_output_resolution(the_res)

        return the_res

    def get_FLOPs(self, input_resolution):
        the_res = input_resolution
        the_flops = 0
        for the_block in self.block_list:
            the_flops += the_block.get_FLOPs(the_res)
            the_res = the_block.get_output_resolution(the_res)

        if self.proj is not None:
            the_flops += self.in_channels * self.out_channels * (the_res / self.stride) ** 2 + \
                (the_res / self.stride) ** 2 * self.out_channels

        return the_flops

    def get_model_size(self):
        the_size = 0
        for the_block in self.block_list:
            the_size += the_block.get_model_size()

        if self.proj is not None:
            the_size += self.in_channels * self.out_channels + self.out_channels

        return the_size

    def set_in_channels(self, channels):
        self.in_channels = channels
        if len(self.block_list) == 0:
            self.out_channels = channels
            return

        self.block_list[0].set_in_channels(channels)
        last_channels = self.block_list[0].out_channels
        if len(self.block_list) >= 2 and \
                (isinstance(self.block_list[0], (ConvKX, ConvDW))) and isinstance(self.block_list[1], BN):
            self.block_list[1].set_in_channels(last_channels)

        self.proj = None
        if not self.no_create:
            if self.stride > 1 or self.in_channels != self.out_channels:
                self.proj = nn.Sequential(
                    nn.Conv2d(self.in_channels, self.out_channels, 1, self.stride),
                    nn.BatchNorm2d(self.out_channels),
                )
                self.proj.train()
                self.proj.requires_grad_(True)

    @classmethod
    def create_from_str(cls, struct_str, no_create=False, **kwargs):
        assert ResBlock.is_instance_from_str(struct_str)
        idx = _get_right_parentheses_index_(struct_str)
        assert idx is not None
        the_stride = None
        param_str = struct_str[len('ResBlock('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = f'uuid{uuid.uuid4().hex}'
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        first_comma_index = param_str.find(',')
        # cannot parse in_channels, missing, use default
        if first_comma_index < 0 or not param_str[0:first_comma_index].isdigit():
            in_channels = None
            the_block_list, remaining_s = _create_netblock_list_from_str_(param_str, no_create=no_create)
        else:
            in_channels = int(param_str[0:first_comma_index])
            param_str = param_str[first_comma_index + 1:]
            second_comma_index = param_str.find(',')
            if second_comma_index < 0 or not param_str[0:second_comma_index].isdigit():
                the_block_list, remaining_s = _create_netblock_list_from_str_(param_str, no_create=no_create)
            else:
                the_stride = int(param_str[0:second_comma_index])
                param_str = param_str[second_comma_index + 1:]
                the_block_list, remaining_s = _create_netblock_list_from_str_(param_str, no_create=no_create)

        assert len(remaining_s) == 0
        if the_block_list is None or len(the_block_list) == 0:
            return None, struct_str[idx + 1:]
        return ResBlock(block_list=the_block_list, in_channels=in_channels,
                        stride=the_stride, no_create=no_create, block_name=tmp_block_name), struct_str[idx + 1:]


class ResBlockProj(PlainNetBasicBlockClass):
    """ResBlockProj(in_channles, inner_blocks_str). If in_channels is missing,
       use block_list[0].in_channels as in_channels
    """
    def __init__(self, block_list, in_channels=None, stride=None, no_create=False, **kwargs):
        super().__init__(**kwargs)
        self.block_list = block_list
        self.stride = stride
        self.no_create = no_create
        if not no_create:
            self.module_list = nn.ModuleList(block_list)

        if in_channels is None:
            self.in_channels = block_list[0].in_channels
        else:
            self.in_channels = in_channels
        self.out_channels = block_list[-1].out_channels

        if self.stride is None:
            tmp_input_res = 1024
            tmp_output_res = self.get_output_resolution(tmp_input_res)
            self.stride = tmp_input_res // tmp_output_res

        self.proj = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1, self.stride),
            nn.BatchNorm2d(self.out_channels),)

    def forward(self, input_):
        if len(self.block_list) == 0:
            return input_

        output = input_
        for inner_block in self.block_list:
            output = inner_block(output)
        output = output + self.proj(input_)
        return output

    def __str__(self):
        block_str = f'ResBlockProj({self.in_channels},{self.stride},'
        for inner_block in self.block_list:
            block_str += str(inner_block)

        block_str += ')'
        return block_str

    def __repr__(self):
        block_str = f'ResBlockProj({self.block_name}|{self.in_channels},{self.stride},'
        for inner_block in self.block_list:
            block_str += str(inner_block)

        block_str += ')'
        return block_str

    def get_output_resolution(self, input_resolution):
        the_res = input_resolution
        for the_block in self.block_list:
            the_res = the_block.get_output_resolution(the_res)

        return the_res

    def get_FLOPs(self, input_resolution):
        the_res = input_resolution
        the_flops = 0
        for the_block in self.block_list:
            the_flops += the_block.get_FLOPs(the_res)
            the_res = the_block.get_output_resolution(the_res)

        if self.proj is not None:
            the_flops += self.in_channels * self.out_channels * (the_res / self.stride) ** 2 + \
                (the_res / self.stride) ** 2 * self.out_channels

        return the_flops

    def get_model_size(self):
        the_size = 0
        for the_block in self.block_list:
            the_size += the_block.get_model_size()

        if self.proj is not None:
            the_size += self.in_channels * self.out_channels + self.out_channels

        return the_size

    def set_in_channels(self, channels):
        self.in_channels = channels
        if len(self.block_list) == 0:
            self.out_channels = channels
            return

        self.block_list[0].set_in_channels(channels)
        last_channels = self.block_list[0].out_channels
        if len(self.block_list) >= 2 and \
                (isinstance(self.block_list[0], (ConvKX, ConvDW))) and isinstance(self.block_list[1], BN):
            self.block_list[1].set_in_channels(last_channels)

        self.proj = None
        if not self.no_create:
            if self.stride > 1 or self.in_channels != self.out_channels:
                self.proj = nn.Sequential(
                    nn.Conv2d(self.in_channels, self.out_channels, 1, self.stride),
                    nn.BatchNorm2d(self.out_channels),
                )
                self.proj.train()
                self.proj.requires_grad_(True)

    @classmethod
    def create_from_str(cls, struct_str, no_create=False, **kwargs):
        assert ResBlockProj.is_instance_from_str(struct_str)
        idx = _get_right_parentheses_index_(struct_str)
        assert idx is not None
        the_stride = None
        param_str = struct_str[len('ResBlockProj('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = f'uuid{uuid.uuid4().hex}'
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        first_comma_index = param_str.find(',')
        # cannot parse in_channels, missing, use default
        if first_comma_index < 0 or not param_str[0:first_comma_index].isdigit():
            in_channels = None
            the_block_list, remaining_s = _create_netblock_list_from_str_(param_str, no_create=no_create)
        else:
            in_channels = int(param_str[0:first_comma_index])
            param_str = param_str[first_comma_index + 1:]
            second_comma_index = param_str.find(',')
            if second_comma_index < 0 or not param_str[0:second_comma_index].isdigit():
                the_block_list, remaining_s = _create_netblock_list_from_str_(param_str, no_create=no_create)
            else:
                the_stride = int(param_str[0:second_comma_index])
                param_str = param_str[second_comma_index + 1:]
                the_block_list, remaining_s = _create_netblock_list_from_str_(param_str, no_create=no_create)

        assert len(remaining_s) == 0
        if the_block_list is None or len(the_block_list) == 0:
            return None, struct_str[idx + 1:]
        return ResBlockProj(block_list=the_block_list, in_channels=in_channels,
                            stride=the_stride, no_create=no_create, block_name=tmp_block_name), struct_str[idx + 1:]


class SE(PlainNetBasicBlockClass):
    """Squeeze and Excitation"""

    def __init__(self, out_channels=None, copy_from=None,
                 no_create=False, **kwargs):
        super().__init__(**kwargs)
        self.no_create = no_create

        if copy_from is not None:
            raise RuntimeError('Not implemented')
        self.in_channels = out_channels
        self.out_channels = out_channels
        self.se_ratio = 0.25
        self.se_channels = max(1, int(round(self.out_channels * self.se_ratio)))
        if no_create or self.out_channels == 0:
            return
        self.netblock = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.se_channels, kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(self.se_channels),
            HSwish(),
            nn.Conv2d(in_channels=self.se_channels, out_channels=self.out_channels, kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid()
        )

    def forward(self, input_):
        se_x = self.netblock(input_)
        #se_x = torch.clamp(se_x + 3, 0, 6) / 6
        return se_x * input_

    def __str__(self):
        return 'SE({self.out_channels})'

    def __repr__(self):
        return 'SE({self.block_name}|{self.out_channels})'

    def get_output_resolution(self, input_resolution):
        return input_resolution

    def get_FLOPs(self, input_resolution):
        return self.in_channels * self.se_channels + self.se_channels * self.out_channels + self.out_channels + \
            self.out_channels * input_resolution ** 2

    def get_model_size(self):
        return self.in_channels * self.se_channels + 2 * self.se_channels + self.se_channels * self.out_channels + \
            2 * self.out_channels

    def set_in_channels(self, channels):
        self.in_channels = channels
        if not self.no_create:
            self.netblock = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channels=self.out_channels, out_channels=self.se_channels, kernel_size=1, stride=1,
                          padding=0, bias=False),
                nn.BatchNorm2d(self.se_channels),
                HSwish(),
                nn.Conv2d(in_channels=self.se_channels, out_channels=self.out_channels, kernel_size=1, stride=1,
                          padding=0, bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.Sigmoid()
            )
            self.netblock.train()
            self.netblock.requires_grad_(True)

    @classmethod
    def create_from_str(cls, struct_str, no_create=False, **kwargs):
        assert SE.is_instance_from_str(struct_str)
        idx = _get_right_parentheses_index_(struct_str)
        assert idx is not None
        param_str = struct_str[len('SE('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = f'uuid{uuid.uuid4().hex}'
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        out_channels = int(param_str)
        return SE(out_channels=out_channels, no_create=no_create, block_name=tmp_block_name), struct_str[idx + 1:]


# pylint: disable=arguments-differ,abstract-method
class SwishImplementation(torch.autograd.Function):
    """swish implementation"""

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(PlainNetBasicBlockClass):
    """swish activation"""

    def __init__(self, out_channels=None, copy_from=None,
                 no_create=False, **kwargs):
        super().__init__(**kwargs)
        self.no_create = no_create

        if copy_from is not None:
            raise RuntimeError('Not implemented')
        self.in_channels = out_channels
        self.out_channels = out_channels

    def forward(self, input_):
        return SwishImplementation.apply(input_)

    def __str__(self):
        return f'Swish({self.out_channels})'

    def __repr__(self):
        return f'Swish({self.block_name}|{self.out_channels})'

    def get_output_resolution(self, input_resolution):
        return input_resolution

    def get_FLOPs(self, input_resolution):
        return self.out_channels * input_resolution ** 2

    def get_model_size(self):
        return 0

    def set_in_channels(self, channels):
        self.in_channels = channels
        self.out_channels = channels

    @classmethod
    def create_from_str(cls, struct_str, no_create=False, **kwargs):
        assert Swish.is_instance_from_str(struct_str)
        idx = _get_right_parentheses_index_(struct_str)
        assert idx is not None
        param_str = struct_str[len('Swish('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = f'uuid{uuid.uuid4().hex}'
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        out_channels = int(param_str)
        return Swish(out_channels=out_channels, no_create=no_create, block_name=tmp_block_name), struct_str[idx + 1:]


def _add_bn_layer_(block_list):
    """add bn layer to all blocks in block_list"""
    new_block_list = []
    for the_block in block_list:
        if isinstance(the_block, (ConvKX, ConvDW)):
            out_channels = the_block.out_channels
            new_bn_block = BN(out_channels=out_channels, no_create=True)
            new_seq_with_bn = Sequential(block_list=[the_block, new_bn_block], no_create=True)
            new_block_list.append(new_seq_with_bn)
        elif hasattr(the_block, 'block_list'):
            new_block_list = _add_bn_layer_(the_block.block_list)
            the_block.module_list = nn.ModuleList(new_block_list)
            the_block.block_list = new_block_list
            new_block_list.append(the_block)
        else:
            new_block_list.append(the_block)

    return new_block_list


def _remove_bn_layer_(block_list):
    """remove bn layer from all blocks in block_list"""
    new_block_list = []
    for the_block in block_list:
        if isinstance(the_block, BN):
            continue
        if hasattr(the_block, 'block_list'):
            new_block_list = _remove_bn_layer_(the_block.block_list)
            the_block.module_list = nn.ModuleList(new_block_list)
            the_block.block_list = new_block_list
            new_block_list.append(the_block)
        else:
            new_block_list.append(the_block)

    return new_block_list


def _add_se_layer_(block_list):
    """add se layer to all blocks in block_list"""
    new_block_list = []
    for the_block in block_list:
        if isinstance(the_block, RELU):
            out_channels = the_block.out_channels
            new_se_block = SE(out_channels=out_channels, no_create=True)
            new_seq_with_bn = Sequential(block_list=[the_block, new_se_block], no_create=True)
            new_block_list.append(new_seq_with_bn)
        elif hasattr(the_block, 'block_list'):
            new_block_list = _add_se_layer_(the_block.block_list)
            the_block.module_list = nn.ModuleList(new_block_list)
            the_block.block_list = new_block_list
            new_block_list.append(the_block)
        else:
            new_block_list.append(the_block)

    return new_block_list


def _replace_relu_with_swish_layer_(block_list):
    """replace all relu with swish in all blocks"""
    new_block_list = []
    for the_block in block_list:
        if isinstance(the_block, RELU):
            out_channels = the_block.out_channels
            new_swish_block = Swish(out_channels=out_channels, no_create=True)
            new_block_list.append(new_swish_block)
        elif hasattr(the_block, 'block_list'):
            new_block_list = _replace_relu_with_swish_layer_(the_block.block_list)
            the_block.module_list = nn.ModuleList(new_block_list)
            the_block.block_list = new_block_list
            new_block_list.append(the_block)
        else:
            new_block_list.append(the_block)

    return new_block_list


def _fuse_convkx_and_bn_(convkx, batch_norm):
    """fuse conv and bn layer"""
    the_weight_scale = batch_norm.weight / torch.sqrt(batch_norm.running_var + batch_norm.eps)
    convkx.weight[:] = convkx.weight * the_weight_scale.view((-1, 1, 1, 1))
    the_bias_shift = (batch_norm.weight * batch_norm.running_mean) / \
        torch.sqrt(batch_norm.running_var + batch_norm.eps)
    batch_norm.weight[:] = 1
    batch_norm.bias[:] = batch_norm.bias - the_bias_shift
    batch_norm.running_var[:] = 1.0 - batch_norm.eps
    batch_norm.running_mean[:] = 0.0


def _fuse_bn_layer_for_blocks_list_(block_list):
    """apply fuse operation to all blocks"""
    last_block = None  # type: ConvKX
    with torch.no_grad():
        for the_block in block_list:
            if isinstance(the_block, BN):
                # assert isinstance(last_block, ConvKX) or isinstance(last_block, ConvDW)
                if isinstance(last_block, (ConvKX, ConvDW)):
                    _fuse_convkx_and_bn_(last_block.netblock, the_block.netblock)
                else:
                    print(f'--- warning! Cannot fuse BN={the_block} because last_block={last_block}')

                last_block = None
            elif isinstance(the_block, (ConvKX, ConvDW)):
                last_block = the_block
            elif hasattr(the_block, 'block_list') and the_block.block_list is not None and \
                    len(the_block.block_list) > 0:
                _fuse_bn_layer_for_blocks_list_(the_block.block_list)
            else:
                pass


def register_netblocks_dict(netblocks_dict: dict):
    """add all basic layer classes to dict"""
    this_py_file_netblocks_dict = {
        'GhostConv': GhostConv,
        'GhostShuffleBlock': GhostShuffleBlock,
        'HS': HS,
        'AdaptiveAvgPool': AdaptiveAvgPool,
        'BN': BN,
        'ConvDW': ConvDW,
        'ConvKX': ConvKX,
        'ConvKXG2': ConvKXG2,
        'ConvKXG4': ConvKXG4,
        'ConvKXG8': ConvKXG8,
        'ConvKXG16': ConvKXG16,
        'ConvKXG32': ConvKXG32,
        'Flatten': Flatten,
        'Linear': Linear,
        'MaxPool': MaxPool,
        'MultiSumBlock': MultiSumBlock,
        'MultiCatBlock': MultiCatBlock,
        'PlainNetBasicBlockClass': PlainNetBasicBlockClass,
        'RELU': RELU,
        'ResBlock': ResBlock,
        'ResBlockProj': ResBlockProj,
        'Sequential': Sequential,
        'SE': SE,
        'Swish': Swish,
    }
    netblocks_dict.update(this_py_file_netblocks_dict)
    return netblocks_dict
