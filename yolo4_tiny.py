import paddle
import paddle.nn as nn
from CSPdarknet53_tiny import CSPDarkNet
from attention import cbam_block, eca_block, se_block

### test function ###
def F(tensor):
    print(f"sum:{tensor.sum().item()}, mean:{tensor.mean().item()}, min-max:[{tensor.min().item()}-{tensor.max().item()}]")
### test function ###
#-------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   Conv2d + BatchNormalization + LeakyReLU
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

#---------------------------------------------------#
#   卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        BasicConv(in_filters, filters_list[0], 3),
        nn.Conv2D(filters_list[0], filters_list[1], 1),
    )
    return m

#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
attention_block = [se_block, cbam_block, eca_block]
class YoloBody(nn.Layer):
    def __init__(self, num_anchors, num_classes, phi=0):
        super(YoloBody, self).__init__()
        if phi >= 4:
            raise AssertionError("Phi must be less than or equal to 3 (0, 1, 2, 3).")

        self.phi            = phi
        self.backbone       = CSPDarkNet()

        self.conv_for_P5    = BasicConv(512,256,1)
        self.yolo_headP5    = yolo_head([512, num_anchors * (5 + num_classes)],256)

        self.upsample       = Upsample(256,128)
        self.yolo_headP4    = yolo_head([256, num_anchors * (5 + num_classes)],384)

        if 1 <= self.phi and self.phi <= 3:
            self.feat1_att      = attention_block[self.phi - 1](256)
            self.feat2_att      = attention_block[self.phi - 1](512)
            self.upsample_att   = attention_block[self.phi - 1](128)

    def forward(self, x):
        #---------------------------------------------------#
        #   生成CSPdarknet53_tiny的主干模型
        #   feat1的shape为26,26,256
        #   feat2的shape为13,13,512
        #---------------------------------------------------#
        feat1, feat2 = self.backbone(x)
        if 1 <= self.phi and self.phi <= 3:
            feat1 = self.feat1_att(feat1)
            feat2 = self.feat2_att(feat2)
        # 13,13,512 -> 13,13,256
        P5 = self.conv_for_P5(feat2)
        # 13,13,256 -> 13,13,512 -> 13,13,255
        out0 = self.yolo_headP5(P5) 

        # 13,13,256 -> 13,13,128 -> 26,26,128
        P5_Upsample = self.upsample(P5)
        # 26,26,256 + 26,26,128 -> 26,26,384
        if 1 <= self.phi and self.phi <= 3:
            P5_Upsample = self.upsample_att(P5_Upsample)
        P4 = paddle.concat([P5_Upsample,feat1],axis=1)

        # 26,26,384 -> 26,26,256 -> 26,26,255
        out1 = self.yolo_headP4(P4)
        return out0, out1

if __name__ == '__main__':
    import numpy as np
    cls_num = 80

    # np.random.seed(0)
    # input = np.random.random([10, 3, 416, 416])
    # input = paddle.to_tensor(input, dtype=paddle.float32)
    import cv2
    input = cv2.imread("C:/Users/311/Desktop/sheep.png")
    image = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    crop_img = cv2.resize(image, [416]*2, cv2.INTER_CUBIC) / 255.0 # cv2.resize 不在原图动作
    # h,w,c -> c,h,w
    input = [ crop_img.transpose([2, 0, 1]) ]

    input = paddle.to_tensor(input, dtype=paddle.float32)

    model = YoloBody(3, cls_num)

    model.load_dict(paddle.load("D:/File/seaDrive/97052426/我的资料库/云文件/project/python/pytorch/yolov4-tiny-paddle/yolo_weight.pd"))
    out0, out1 = model(input)
    F(out0)
    F(out1)
    # with open("D:/File/seaDrive/97052426/我的资料库/云文件/project/python/pytorch/cmp/p_model.txt",'w') as f:
    #     for key in model.state_dict().keys():
    #         f.write(key.replace('running_mean','_mean').replace('running_var','_variance')+'\n')
    
    # output0, output1 = model(input)

    ### yolo_head 网络初始化不一致
    # paddle.summary(model, (1,3,416,416))

    ### diffirents
    # backbone.conv1.bn.running_mean 
    # backbone.conv1.bn._mean

    # backbone.conv1.bn.running_var 
    # backbone.conv1.bn._variance

    # backbone.conv1.bn.num_batches_tracked
    # None

