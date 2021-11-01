import paddle
import paddle.nn as nn


#-------------------------------------------------#
#   卷积块 out_shape = in_shape
#   Conv2d + BatchNorm2d + LeakyReLU
#-------------------------------------------------#
class BasicConv(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias_attr=False)
        self.bn = nn.BatchNorm2D(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

         
'''
                    input
                      |
                  BasicConv
                      -----------------------
                      |                     |
                 route_group              route
                      |                     |
                  BasicConv                 |
                      |                     |
    -------------------                     |
    |                 |                     |
 route_1          BasicConv                 |
    |                 |                     |
    -----------------cat                    |
                      |                     |
        ----      BasicConv                 |
        |             |                     |
      feat           cat---------------------
                      |
                 MaxPooling2D
'''
#---------------------------------------------------#
#   CSPdarknet53-tiny的结构块
#   存在一个大残差边
#   这个大残差边绕过了很多的残差结构
#---------------------------------------------------#
class Resblock_body(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(Resblock_body, self).__init__()
        self.out_channels = out_channels

        self.conv1 = BasicConv(in_channels, out_channels, 3)

        self.conv2 = BasicConv(out_channels//2, out_channels//2, 3)
        self.conv3 = BasicConv(out_channels//2, out_channels//2, 3)

        self.conv4 = BasicConv(out_channels, out_channels, 1)
        self.maxpool = nn.MaxPool2D(2,2)

    def forward(self, x):
        # 利用一个3x3卷积进行特征整合
        x = self.conv1(x)
        # 引出一个大的残差边route
        route = x
        # 对特征层的通道进行分割，取第二部分作为主干部分。
        x = paddle.split(x, 2, axis = 1)[1]
        # 对主干部分进行3x3卷积
        x = self.conv2(x)
        # 引出一个小的残差边route_1
        route1 = x
        # 对第主干部分进行3x3卷积
        x = self.conv3(x)
        # 主干部分与残差部分进行相接
        x = paddle.concat([x,route1], axis = 1) 

        # 对相接后的结果进行1x1卷积
        x = self.conv4(x)
        feat = x
        x = paddle.concat([route, x], axis = 1)
        
        # 利用最大池化进行高和宽的压缩
        x = self.maxpool(x)
        return x,feat

class CSPDarkNet(nn.Layer):
    def __init__(self):
        super(CSPDarkNet, self).__init__()
        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        self.conv1 = BasicConv(3, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv(32, 64, kernel_size=3, stride=2)

        # 104,104,64 -> 52,52,128
        self.resblock_body1 =  Resblock_body(64, 64)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 =  Resblock_body(128, 128)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 =  Resblock_body(256, 256)
        # 13,13,512 -> 13,13,512
        self.conv3 = BasicConv(512, 512, kernel_size=3)

        self.num_features = 1

    def forward(self, x):
        # 416,416,3 -> 208,208,32 -> 104,104,64
        x = self.conv1(x)
        x = self.conv2(x)

        # 104,104,64 -> 52,52,128
        x, _    = self.resblock_body1(x)
        # 52,52,128 -> 26,26,256
        x, _    = self.resblock_body2(x)
        # 26,26,256 -> x为13,13,512
        #           -> feat1为26,26,256
        x, feat1    = self.resblock_body3(x)
        

        # 13,13,512 -> 13,13,512
        x = self.conv3(x)
        feat2 = x
        return feat1,feat2


if __name__ == '__main__':
    net = CSPDarkNet()
    import numpy as np
    np.random.seed(0)
    input = paddle.to_tensor(np.random.random([10,3,416,416]),dtype=paddle.float32)
    x,y=net(input)
    print(f"mean:{x.mean().item()},var:{x.var().item()}")
    print(f"mean:{y.mean().item()},var:{y.var().item()}")
    # paddle.summary(net, (1,3,416,416))