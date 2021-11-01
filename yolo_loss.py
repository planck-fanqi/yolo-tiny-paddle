import math
import paddle
import paddle.nn as nn

### test function ###
def random_location(max, seed=0):
    np.random.seed(seed)
    max = np.array(max)
    return tuple((np.random.random(max.shape) * max).astype(np.int64).tolist())

def random_sample(v, max):
    l = []
    for i in range(10):
        l.append(v[random_location(max, i)].item())
    print(l)

def feature_num(tensor):
    print(f"sum:{tensor.sum().item()}, mean:{tensor.mean().item()}, min-max:[{tensor.min().item()}-{tensor.max().item()}]")
### test function ###

FloatTensor = lambda x: paddle.to_tensor(x, dtype=paddle.float32)
LongTensor = lambda x: paddle.to_tensor(x, dtype=paddle.int64)
neg_bool = lambda x: (1 - x.astype(paddle.int64)).astype(paddle.bool)

def mask_select(v, m):
    m = m.clone().astype(paddle.bool)
    if len(v.shape) != len(m.shape):
        if v.shape[:-1] == m.shape:
            m = m.unsqueeze(-1).expand(v.shape)
            return v.masked_select(m).reshape([-1, v.shape[-1]])
        else: raise ValueError("mask 与 input 形状不一致") # paddle masked_select 必须维度一致
    else:
        if v.shape == m.shape:
            return v.masked_select(m)
        else: raise ValueError("mask 与 input 形状不一致")

def jaccard(_box_a, _box_b):
    b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
    b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
    b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
    b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2

    box_a = paddle.zeros_like(_box_a)
    box_b = paddle.zeros_like(_box_b)
    box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
    box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2
    A = box_a.shape[0]
    B = box_b.shape[0]
    max_xy = paddle.minimum(box_a[:, 2:].unsqueeze(1).expand((A, B, 2)),
                       box_b[:, 2:].unsqueeze(0).expand((A, B, 2)))
    min_xy = paddle.maximum(box_a[:, :2].unsqueeze(1).expand((A, B, 2)),
                       box_b[:, :2].unsqueeze(0).expand((A, B, 2)))
    inter = paddle.clip((max_xy - min_xy), min=0)

    inter = inter[:, :, 0] * inter[:, :, 1]
    # 计算先验框和真实框各自的面积
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    # 求IOU
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def BCELoss(pred,target):
    epsilon = 1e-7
    pred = pred.clone().clip(epsilon, 1-epsilon)
    output = -target * paddle.log(pred) - (1.0 - target) * paddle.log(1.0 - pred)
    return output
    
#---------------------------------------------------#
#   平滑标签
#---------------------------------------------------#
def smooth_labels(y_true, label_smoothing,num_classes):
    return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

def box_ciou(b1, b2):
    """
    输入为：b1: num_mask ,4   b2: num_mask ,4
    返回为： ciou: num_mask
    """
    # 求出预测框左上角右下角
    b1_xy = b1[:, :2]
    b1_wh = b1[:, 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    # 求出真实框左上角右下角
    b2_xy = b2[:, :2]
    b2_wh = b2[:, 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # 求真实框和预测框所有的iou
    intersect_mins = paddle.maximum(b1_mins, b2_mins)
    intersect_maxes = paddle.minimum(b1_maxes, b2_maxes)
    intersect_wh = paddle.maximum(intersect_maxes - intersect_mins, paddle.zeros_like(intersect_maxes))
    intersect_area = intersect_wh[:, 0] * intersect_wh[:, 1]
    b1_area = b1_wh[:, 0] * b1_wh[:, 1]
    b2_area = b2_wh[:, 0] * b2_wh[:, 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / paddle.clip(union_area,min = 1e-6)

    # 计算中心的差距
    center_distance = paddle.sum(paddle.pow((b1_xy - b2_xy), 2), axis=-1)
    
    # 找到包裹两个框的最小框的左上角和右下角
    enclose_mins = paddle.minimum(b1_mins, b2_mins)
    enclose_maxes = paddle.maximum(b1_maxes, b2_maxes)
    enclose_wh = paddle.maximum(enclose_maxes - enclose_mins, paddle.zeros_like(intersect_maxes))
    # 计算对角线距离
    enclose_diagonal = paddle.sum(paddle.pow(enclose_wh,2), axis=-1)

    v = (4 / (math.pi ** 2)) * paddle.pow((paddle.atan(b1_wh[:, 0]/paddle.clip(b1_wh[:, 1],min = 1e-6)) - paddle.atan(b2_wh[:, 0]/paddle.clip(b2_wh[:, 1],min = 1e-6))), 2)
    alpha = v / paddle.clip((1.0 - iou + v),min=1e-6)
    ciou = 1 - iou + center_distance / paddle.clip(enclose_diagonal,min = 1e-6) + alpha * v
    return ciou
  

class YOLOLoss(nn.Layer):
    def __init__(self, anchors,num_classes, label_smooth=0, normalize=True):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_anchors = 3
        self.num_classes = num_classes
        self.img_size = 416
        self.label_smooth = label_smooth

        self.ignore_threshold = 0.5
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0
        self.lambda_loc = 1.0
        self.normalize = normalize

    def forward(self, input, targets=None): # targets [n, 5(x,y,w,h,cls)]
        #   input的shape为  bs, num_anchor*(5+num_classes), 13, 13 或 bs, num_anchor*(5+num_classes), 26, 26

        bs = input.shape[0]
        feature_size = input.shape[2] #   特征层的大小
        # 转换成 [bs, 3, 13, 13, 5 + num_classes] 或 [bs, 3, 26, 26, 5 + num_classes]
        prediction = input.reshape([bs, self.num_anchors, self.num_classes + 5, feature_size, feature_size]).transpose([0, 1, 3, 4, 2])
        #-----------------------------------------------------------------------#
        #   每一个特征点对应原来的图片上多少个像素点
        #   如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点
        #   如果特征层为26x26的话，一个特征点就对应原来的图片上的16个像素点
        #-----------------------------------------------------------------------#
        feature_stride = self.img_size / feature_size
        scaled_anchors = self.anchors /feature_stride
        
        # 获得置信度，是否有物体
        conf = nn.Sigmoid()(prediction[:,:,:,:, 4])
        # 种类置信度
        pred_cls = nn.Sigmoid()(prediction[:,:,:,:, 5:])
        #---------------------------------------------------------------#
        #   找到哪些先验框内部包含物体
        #   利用真实框和先验框计算交并比
        #----------------------------------------------------------------#
        mask, noobj_mask, t_box, tcls, box_loss_scale_x, box_loss_scale_y = self.get_target(targets, scaled_anchors, feature_size)
        #---------------------------------------------------------------#
        #   将预测结果进行解码，判断预测结果和真实值的重合程度
        #   如果重合程度过大则忽略，因为这些特征点属于预测比较准确的特征点
        #   作为负样本不合适
        #----------------------------------------------------------------#
        noobj_mask, pred_boxes_for_ciou = self.get_ignore(prediction, targets, scaled_anchors, feature_size, noobj_mask)
        
        box_loss_scale = 2 - box_loss_scale_x * box_loss_scale_y
        #---------------------------------------------------------------#
        #   计算预测结果和真实结果的CIOU
        #----------------------------------------------------------------#
        ciou = box_ciou( mask_select(pred_boxes_for_ciou,mask), mask_select(t_box,mask)) * mask_select(box_loss_scale, mask)
        loss_loc = paddle.sum(ciou)


        # 计算置信度的loss
        loss_conf = paddle.mean(BCELoss(conf, mask) * mask) + \
                    paddle.mean(BCELoss(conf, mask) * noobj_mask)
                    
        loss_cls = paddle.mean(BCELoss(mask_select(pred_cls, mask), smooth_labels(mask_select(tcls, mask),self.label_smooth,self.num_classes)))
        
        loss = loss_conf * self.lambda_conf + loss_cls * self.lambda_cls + loss_loc * self.lambda_loc

        if self.normalize:
            num_pos = paddle.sum(mask)
        else:
            num_pos = bs/2
        
        return loss, num_pos

    def get_target(self, target, anchors, feature_size):
        '''
        mask        [bs, num_anchor, feature_size, feature_size]                  有目标的特征点 \n
        noobj_mask  [bs, num_anchor, feature_size, feature_size]                  无目标的特征点 \n
        box_scale   [bs, num_anchor, feature_size, feature_size]                  缩放比例 \n
        t_box       [bs, num_anchor, feature_size, feature_size, 4]               中心宽高的真实值 \n
        tcls        [bs, num_anchor, feature_size, feature_size, num_classes]     种类真实值
        '''
        # 小特征层使用大anchor
        anchor_group = [3,4,5] if feature_size == 13 else [1,2,3]

        bs = len(target)
        anchor_map_size = (bs, self.num_anchors, feature_size, feature_size)
        mask = paddle.zeros(anchor_map_size)
        noobj_mask = paddle.ones(anchor_map_size)

        t_box = paddle.zeros(anchor_map_size + (4,))
        tcls = paddle.zeros(anchor_map_size + (self.num_classes,))
        box_loss_scale_x = paddle.zeros(anchor_map_size)
        box_loss_scale_y = paddle.zeros(anchor_map_size)

        for b in range(bs):
            if len(target[b])==0: continue
            feature_target = target[b][:4] * feature_size # [num_target, 4]
            gt_box_shape = FloatTensor(paddle.concat([paddle.zeros((len(feature_target), 2)), feature_target[:,2:4]], 1)) #   [num_target, 4] 
            anchor_shapes = FloatTensor(paddle.concat([paddle.zeros((len(anchors), 2)), FloatTensor(anchors)], 1)) #   [6, 4]
            #-------------------------------------------------------#
            #   计算anchor x,y为0时，交并比
            #   num_target, num_anchor=3
            #-------------------------------------------------------#
            anch_ious = jaccard(gt_box_shape, anchor_shapes)

            #-------------------------------------------------------#
            #   计算形状最匹配的先验框是哪个
            #   num_target, num_anchor=3
            #-------------------------------------------------------#
            best_ns = paddle.argmax(anch_ious,axis=-1)
            for i, best_n in enumerate(best_ns):
                #-------------------------------------------------------#
                #   计算出正样本属于特征层的哪个特征点
                #-------------------------------------------------------#
                if best_n not in anchor_group: continue
                anchor_index = anchor_group.index(best_n)
                gi, gj = paddle.floor(feature_target[i][:2]).astype(paddle.int64)

                noobj_mask[b, anchor_index, gj, gi] = 0   #   noobj_mask代表无目标的特征点
                mask[b, anchor_index, gj, gi] = 1         #   mask代表有目标的特征点

                t_box[b, anchor_index, gj, gi] = feature_target[i][:4]
                #----------------------------------------#
                #   用于获得wh的比例
                #   大目标loss权重小，小目标loss权重大
                #----------------------------------------#
                box_loss_scale_x[b, anchor_index, gj, gi] = target[b][i, 2]
                box_loss_scale_y[b, anchor_index, gj, gi] = target[b][i, 3]
                tcls[b, anchor_index, gj, gi, target[b][i, 4].astype(paddle.int64)] = 1 #   tcls代表种类置信度
        return mask, noobj_mask, t_box, tcls, box_loss_scale_x, box_loss_scale_y

    def get_ignore(self,prediction,target,scaled_anchors,feature_size,noobj_mask):
        anchor_group = [3,4,5] if feature_size == 13 else [1,2,3]
        scaled_anchors = scaled_anchors[LongTensor(anchor_group)]
        bs = len(target)
        #-------------------------------------------------------#
        #   获得当前特征层先验框所属的编号，方便后面对先验框筛选
        #-------------------------------------------------------#
        feature_map_size = (bs, self.num_anchors, feature_size, feature_size)
        x = nn.Sigmoid()(prediction[:,:,:,:, 0])  
        y = nn.Sigmoid()(prediction[:,:,:,:, 1])
        w = prediction[:,:,:,:, 2]
        h = prediction[:,:,:,:, 3]

        # 生成网格，先验框中心，网格左上角
        grid_x = paddle.meshgrid([paddle.linspace(0, feature_size - 1, feature_size)]*2)[1].expand(feature_map_size)
        grid_y = paddle.meshgrid([paddle.linspace(0, feature_size - 1, feature_size)]*2)[0].expand(feature_map_size)
        # 生成先验框的宽高
        anchor_w = scaled_anchors[:,0:1]
        anchor_h = scaled_anchors[:,1:]
        
        anchor_w = anchor_w.expand((bs, self.num_anchors,feature_size*feature_size)).reshape(feature_map_size)
        anchor_h = anchor_h.expand((bs, self.num_anchors,feature_size*feature_size)).reshape(feature_map_size)
        
        #-------------------------------------------------------#
        #   计算调整后的先验框中心与宽高
        #-------------------------------------------------------#
        pred_boxes = paddle.zeros(LongTensor(prediction.shape[:-1] + [4])) # [bs, num_anchor, size, size, 4]
        pred_boxes[:,:,:,:, 0] = x + grid_x
        pred_boxes[:,:,:,:, 1] = y + grid_y
        pred_boxes[:,:,:,:, 2] = paddle.exp(w) * anchor_w
        pred_boxes[:,:,:,:, 3] = paddle.exp(h) * anchor_h
        for i in range(bs):
            pred_boxes_for_ignore = pred_boxes[i].reshape([-1, 4]) # [num_anchor*feature^2, 4]
            if len(target[i]) > 0:
                #   计算真实框，并把真实框转换成相对于特征层的大小 [num_target, 4]
                gt_box = target[i][:,:4] * feature_size
                #   [num_target, num_predict_anchors] = jaccard([num_target, 4], [num_anchor*feature^2, 4])
                anch_ious = jaccard(gt_box, pred_boxes_for_ignore) 
                #-------------------------------------------------------#
                #   每个先验框最匹配的真实框
                #   [num_predict_anchors]
                #-------------------------------------------------------#
                anch_ious_max = paddle.max(anch_ious,axis=0)
                anch_ious_max = anch_ious_max.reshape((self.num_anchors, feature_size, feature_size))
                noobj_mask[i][anch_ious_max>self.ignore_threshold] = 0
        return noobj_mask, pred_boxes

if __name__ == '__main__':
    bs = 10
    num_cls = 80
    feature_size = 13

    anchors = paddle.to_tensor([[10,14],  [23,27],  [37,58],  [81,82],  [135,169],  [344,319]], dtype=paddle.float32)
    yolo_loss    = YOLOLoss(anchors, 80, 0, True)

    import numpy as np
    np.random.seed(0)
    features = np.random.random([bs, 3*(5+num_cls),feature_size,feature_size])
    targets = np.random.random([bs, 3, 5])

    features = paddle.to_tensor(features, dtype=paddle.float32)
    targets = paddle.to_tensor(targets, dtype=paddle.float32)
    print(features.mean().item())

    loss_item, num_pos = yolo_loss(features, targets)
    print(loss_item, num_pos)
