import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

class SE(nn.Module):
    """Squeeze and Excitation"""

    def __init__(self, out_channels=None, **kwargs):
        super().__init__(**kwargs)

        self.in_channels = out_channels
        self.out_channels = out_channels
        self.se_ratio = 0.25
        self.se_channels = max(1, int(round(self.out_channels * self.se_ratio)))

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

class HSwish(nn.Module):

	def __init__(self):
		super(HSwish, self).__init__()

	def forward(self, inputs):
		clip = torch.clamp(inputs + 3, 0, 6) / 6
		return inputs * clip

class GhostConv(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, kernel_size=1, ratio=2, dw_size=3, 
                stride=1, relu=True, **kwargs):
        super(GhostConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dw_size = dw_size
        self.relu = relu
        self.init_channels = math.ceil(out_channels / ratio)
        self.new_channels = self.init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.init_channels, self.kernel_size, self.stride, self.kernel_size//2, bias=False),
            #nn.BatchNorm2d(self.init_channels),
            #HSwish(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(self.init_channels, self.new_channels, self.dw_size, 1, self.dw_size//2, groups=self.init_channels, bias=False),
            #nn.BatchNorm2d(self.new_channels),
            #HSwish(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]

class GhostShuffleBlock(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, bottleneck_channels=None, stride=None,
                 kernel_size=None, repeate_times=None):
        super(GhostShuffleBlock, self).__init__()

        self.stride = stride

        self.group = 2 # group = 2

        self.in_channels = in_channels
        self.bottleneck_channels = bottleneck_channels

        if self.stride == 1:
            self.mid_channels = self.in_channels // 2
        else:
            self.mid_channels = self.in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size

        assert self.stride in [1, 2]

        # main branch
        branch_main = [
            # pw
            #nn.Conv2d(self.mid_channels,self.bottleneck_channels,kernel_size=1,stride=1,padding=0,bias=False),
            GhostConv(self.mid_channels, self.bottleneck_channels),
            nn.BatchNorm2d(self.bottleneck_channels),
            HSwish(),
            # dw
            nn.Conv2d(self.bottleneck_channels, self.bottleneck_channels, self.kernel_size, 
            self.stride, padding=(self.kernel_size-1)//2, groups=self.bottleneck_channels, bias=False),
            nn.BatchNorm2d(self.bottleneck_channels),
            SE(self.bottleneck_channels), #yfq
            # pw-linear
            #nn.Conv2d(self.bottleneck_channels, self.out_channels // 2,kernel_size=1,stride=1,padding=0,bias=False),
            GhostConv(self.bottleneck_channels, self.out_channels // 2),
            nn.BatchNorm2d(self.out_channels // 2),
            HSwish(),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        # Depth-wise convolution
        self.proj = None
        if self.stride > 1:
            self.proj = nn.Sequential(
                    nn.Conv2d(self.mid_channels, self.mid_channels, self.kernel_size, stride=self.stride,
                        padding=(self.kernel_size-1)//2, groups=self.mid_channels, bias=False),
                    nn.BatchNorm2d(self.mid_channels),
                    GhostConv(self.mid_channels, self.out_channels // 2),
                    #nn.Conv2d(self.mid_channels,self.out_channels // 2,kernel_size=1,stride=1,padding=0,bias=False),
                    nn.BatchNorm2d(self.out_channels // 2),
                    HSwish(),
            )
        else:
            self.proj = nn.Sequential(
                    GhostConv(self.mid_channels, self.out_channels // 2),
                    #nn.Conv2d(self.mid_channels,self.out_channels // 2,kernel_size=1,stride=1,padding=0,bias=False),
                    nn.BatchNorm2d(self.out_channels // 2),
                    HSwish(),
            ) 

    @property
    def stride_num(self):
        return self.stride

    def forward(self, x):
        
        if self.stride == 1:
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            output =  self.branch_main(x2)
            x_proj = self.proj(x1)
            x = torch.cat((x_proj, output), 1)
        else:
            x_proj = x
            output = x
            output = self.branch_main(output)
            x = torch.cat((self.proj(x_proj), output), 1)
        x = self.channel_shuffle(x)
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

class GhostShuffleNet(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, bottleneck_channels=None, stride=None,
                 kernel_size=None, repeate_times=None, **args):
        super(GhostShuffleNet, self).__init__()
        
# SuperConvK3BNRELU(3,40,2,1)SuperGhostShuffleK7(40,144,2,80,1)SuperGhostShuffleK5(144,192,2,176,1)SuperGhostShuffleK3(192,2048,2,384,5)SuperGhostShuffleK3(2048,576,2,608,1)SuperConvK1BNRELU(576,128,1,1)
        branch=[
            # SuperConvK3BNRELU(3,40,2,1)
            nn.Conv2d(3, 40, 3, stride=2, padding=(3-1)//2, bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU(inplace=True),
            # SuperGhostShuffleK7(40,144,2,80,1)
            GhostShuffleBlock(40, 144, 80, 2, 7),
            # SuperGhostShuffleK5(144,192,2,176,1)
            GhostShuffleBlock(144, 192, 176, 2, 5),
            # SuperGhostShuffleK3(192,2048,2,384,5)
            GhostShuffleBlock(192, 2048, 384, 2, 3),
            GhostShuffleBlock(2048, 2048, 384, 1, 3),
            GhostShuffleBlock(2048, 2048, 384, 1, 3),
            GhostShuffleBlock(2048, 2048, 384, 1, 3),
            GhostShuffleBlock(2048, 2048, 384, 1, 3),
            # SuperGhostShuffleK3(2048,576,2,608,1)
            GhostShuffleBlock(2048, 576, 608, 2, 3),
        ]
        self.branch_main = nn.Sequential(*branch)
        self._initialize_weights()

    def forward(self, x):
        #start_timer = time.time()
        outputs = []
        output = x
        for stage in self.branch_main:
            output = stage(output)
            if isinstance(stage, GhostShuffleBlock) and stage.stride_num == 2:
                outputs.append(output)
        # p1 = self.stage1(outputs[-3])
        # p2 = self.stage2(outputs[-2])
        # p3 = self.stage3(outputs[-1])
        # torch.cuda.synchronize()
        # end_timer = time.time()
        # the_latency = (end_timer - start_timer)
        # print("ghostshuffle latency: %s" % (the_latency))
        #return tuple([p1, p2, p3])
        return tuple(outputs[-3:])

    def _initialize_weights(self):
        print("init weights...")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
