import paddle
import paddle.nn as nn
import math
# TODO: attention学习
class se_block(nn.Layer):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias_attr=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, channel, bias_attr=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size
        y = self.avg_pool(x).reshape(b, c)
        y = self.fc(y).reshape(b, c, 1, 1)
        return x * y

class ChannelAttention(nn.Layer):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveMaxPool2D(1)

        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv2D(in_planes, in_planes // ratio, 1, bias_attr=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2D(in_planes // ratio, in_planes, 1, bias_attr=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2D(2, 1, kernel_size, padding=padding, bias_attr=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = paddle.mean(x, axis=1, keepdim=True)
        max_out = paddle.max(x, axis=1, keepdim=True)
        x = paddle.concat([avg_out, max_out], axis=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class cbam_block(nn.Layer):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x*self.channelattention(x)
        x = x*self.spatialattention(x)
        return x

class eca_block(nn.Layer):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv = nn.Conv1D(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias_attr=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        # transpose 无法交换处理轴为1的张量，reshape可以等效处理
        y = y.squeeze(-1).reshape(y.shape[0:1] + y.shape[-2:-4:-1])
        y = self.conv(y).reshape(y.shape[0:1] + y.shape[-1:-3:-1]).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

if __name__ == '__main__':
    # net = ChannelAttention(10)
    net = eca_block(512)
    # import numpy as np
    # x = paddle.to_tensor(np.random.random([10,512,13,13]),)
    # y = net(x)
    paddle.summary(net, (10, 512, 13, 13))