import cv2
import numpy as np
from PIL import Image
import paddle
import paddle.nn as nn
from yolo4_tiny import YoloBody

### test function ###
def F(tensor):
    print(f"sum:{tensor.sum().item()}, mean:{tensor.mean().item()}, min-max:[{tensor.min().item()}-{tensor.max().item()}]")
### test function ###

FloatTensor = lambda x: paddle.to_tensor(x, dtype=paddle.float32)
LongTensor = lambda x: paddle.to_tensor(x, dtype=paddle.int64)

def mask_select(v, m):
    m = m.clone().astype(paddle.bool)
    if len(v.shape) != len(m.shape):
        if v.shape[:-1] == m.shape:
            m = m.unsqueeze(-1).expand(v.shape) # 最后一维升维再扩展
            return v.masked_select(m).reshape([-1, v.shape[-1]]) # [N, ndim]
        else: raise ValueError(f"mask({m.shape}) 与 input({v.shape}) 形状不一致") # paddle masked_select 必须维度一致
    else:
        if v.shape == m.shape:
            return v.masked_select(m)
        elif v.shape[:-1] == m.shape[:-1]:
            return v.masked_select(m.expand(v.shape)).reshape([-1, v.shape[-1]])
        else: raise ValueError(f"mask({m.shape}) 与 input({v.shape}) 形状不一致")

def nms(bboxes, scores, threshold=0.5):
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1)*(y2-y1)   # [N,] 每个bbox的面积
    order = scores.argsort(0,descending=True)

    keep = []
    while order.numel() > 0:       # torch.numel()返回张量元素个数
        if order.numel() == 1:     # 保留框只剩一个
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()    # 保留scores最大的那个框box[i]
            keep.append(i)

        xx1 = x1[order[1:]].clip(min=x1[i])
        yy1 = y1[order[1:]].clip(min=y1[i])
        xx2 = x2[order[1:]].clip(max=x2[i])
        
        yy2 = y2[order[1:]].clip(max=y2[i])
        inter = (xx2-xx1).clip(min=0) * (yy2-yy1).clip(min=0)   # [N-1,]

        iou = inter / (areas[i]+areas[order[1:]]-inter)  # [N-1,]
        idx = (iou <= threshold).nonzero().squeeze() # 注意此时idx为[N-1,] 而order为[N,]
        if idx.numel() == 0: break
        order = order[idx+1]  # 修补索引之间的差值
    return LongTensor(keep)   # Pytorch的索引值为LongTensor

def non_max_suppression(prediction, conf_thre = 0.5, nms_thre=0.4):
    '''
    return [N, 4(x1,y1,x2,y2) + conf + class_idx]
    '''
    box_corner = prediction.clone()
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    # [bs, grids, num_cls + 5]
    output = []
    for batch_idx, pred in enumerate(prediction):
        # [grids, 1], [girds, 1] = max([grids, num_cls]), max([grids, num_cls])
        class_conf, class_idx = paddle.max(pred[:, 5:], 1, keepdim=True), paddle.argmax(pred[:, 5:], 1, keepdim=True)
        # 置信度筛选
        conf_mask = (pred[:, 4] * class_conf[:, 0] >= conf_thre).astype(paddle.bool)
        
        pred        = mask_select(pred, conf_mask)
        class_conf  = mask_select(class_conf, conf_mask)
        class_idx   = mask_select(class_idx, conf_mask)

        if not pred.shape[0]: continue
        # x1, y1, x2, y2, obj_conf*class_conf, class_idx
        detections = paddle.concat([pred[:, :4], pred[:, 4:5]*class_conf, class_idx.astype(paddle.float32)], 1)
        # n_class_filter_grid, 6
        max_detections = []
        for c in class_idx.unique():
            # class_grid, 6
            detections_class = mask_select(detections, class_idx == c)
            keep = nms(
                detections_class[:, :4],
                detections_class[:, 4],
                nms_thre
            )
            # paddle.to_tensor([[1],[2],[3]])[paddle.to_tensor([0,2]) -> [[1],[3]] 
            # paddle.to_tensor([[1],[2],[3]])[paddle.to_tensor([1])] ->  [2]
            detections_class = detections_class[keep] if len(keep) > 1 else detections_class[keep:keep+1]
            max_detections.append(detections_class)
        max_detections = paddle.concat(max_detections)

        if max_detections is not []: output.append(max_detections)
    return output

        
class YOLO(object):
    def __init__(self) -> None:
        self.model_path     =   "yolo_weight.pd"
        self.anchor_path    =   "conf/yolo_anchors.txt"
        self.cls_path       =   "conf/coco_classes.txt"
        self.input_size     =   416
        self.iou            =   0.3
        self.confidence     =   0.5

        with open(self.cls_path, 'r') as f:
            self.clses = list(map(lambda x: x.strip(), f.readlines()))
            self.num_cls = len(self.clses)

        with open(self.anchor_path, 'r') as f:
            self.anchors = FloatTensor([float(x) for x in f.readline().split(',')]).reshape([-1,2])
            self.num_anchors = len(self.anchors) // 2

        self.net = YoloBody(self.num_anchors, self.num_cls)
        self.net.eval()
        self.net.load_dict(paddle.load(self.model_path))

    def decodeBox(self, input):
        bs = input.shape[0]
        feature_size = input.shape[2] #   特征层的大小
        feature_stride = self.input_size / feature_size
        # 转换成 [bs, 3, 13, 13, 5 + num_classes] 或 [bs, 3, 26, 26, 5 + num_classes]
        prediction = input.reshape([bs, self.num_anchors, self.num_cls + 5, feature_size, feature_size]).transpose([0, 1, 3, 4, 2])
        anchors_mask = LongTensor([[3,4,5],[1,2,3]])
        scaled_anchors = self.anchors / feature_stride
        scaled_anchors = scaled_anchors[anchors_mask[0]] if feature_size == 13 else scaled_anchors[anchors_mask[1]]

        feature_map_size = (bs, self.num_anchors, feature_size, feature_size)
        conf = nn.Sigmoid()(prediction[:,:,:,:, 4:5])
        pred_cls = nn.Sigmoid()(prediction[:,:,:,:, 5:])

        grid_x = paddle.meshgrid([paddle.linspace(0, feature_size - 1, feature_size)]*2)[1].expand(feature_map_size)
        grid_y = paddle.meshgrid([paddle.linspace(0, feature_size - 1, feature_size)]*2)[0].expand(feature_map_size)

        anchor_w = scaled_anchors[:,0:1].expand((bs, self.num_anchors,feature_size*feature_size)).reshape(feature_map_size)
        anchor_h = scaled_anchors[:,1:].expand((bs, self.num_anchors,feature_size*feature_size)).reshape(feature_map_size)

        pred_boxes = paddle.zeros(prediction.shape[:-1] + [4]) # [bs, num_anchor, size, size, 4]
        pred_boxes[:,:,:,:, 0] = nn.Sigmoid()(prediction[:,:,:,:, 0])   + grid_x
        pred_boxes[:,:,:,:, 1] = nn.Sigmoid()(prediction[:,:,:,:, 1])   + grid_y
        pred_boxes[:,:,:,:, 2] = paddle.exp(prediction[:,:,:,:, 2])     * anchor_w
        pred_boxes[:,:,:,:, 3] = paddle.exp(prediction[:,:,:,:, 3])     * anchor_h

        output = paddle.concat([pred_boxes*feature_stride, conf, pred_cls], -1).reshape([bs, -1, self.num_cls + 5])
        return output

    def detect(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crop_img = cv2.resize(image, [self.input_size]*2, cv2.INTER_CUBIC) / 255.0 # cv2.resize 不在原图动作
        # h,w,c -> c,h,w
        images = [ crop_img.transpose([2, 0, 1]) ]
        
        with paddle.no_grad():
            images = paddle.to_tensor(images)
            outputs = self.net(images)
            pred_grids = paddle.concat([self.decodeBox(outputs[0]), self.decodeBox(outputs[1])], 1)
            # x1, y1, x2, y2, obj_conf, class_conf, class_idx
            batch_detections = non_max_suppression( pred_grids,
                                                    conf_thre=self.confidence,
                                                    nms_thre=self.iou)

            #   如果没有检测出物体，返回原图
            if not len(batch_detections):
                return image
            single_detection = batch_detections[0].cpu().numpy()
            # h,w,c
            boxes = (single_detection[:, :4] / self.input_size * np.array(image.shape[-2::-1]*2)).astype(np.int64)
            conf = single_detection[:, 4]
            labels = single_detection[:, 5].astype(np.int64)

        thickness = max(sum(image.shape[:2])//self.input_size, 1)
        # font_scale = 1 一个字符占 20px
        font_scale = np.sqrt(np.prod(image.shape[:2]) / 1000**2) * 2
        font_size = int(font_scale*20)

        def draw_annotation(img, left_bottom, ano):
            left_top = np.array(left_bottom); left_top[1] -= font_size
            cv2.rectangle(img, left_top, left_top + font_size*np.array([len(ano), 1]), (0,255,0),-1)
            cv2.putText(img, ano, left_bottom,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        for i, c in enumerate(labels):
            left    = max(0,           boxes[i, 0])
            top     = max(0,           boxes[i, 1])
            right   = min(boxes[i, 2], image.shape[1])
            bottom  = min(boxes[i, 3], image.shape[0])

            draw_annotation(image, [left, top] ,'{} {:.2f}'.format(self.clses[c], conf[i]))
            cv2.rectangle(image, [left, top], [right, bottom], (0, 255, 0), thickness)
            
        return image


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    os.chdir("D:/File/seaDrive/97052426/我的资料库/云文件/project/python/pytorch/yolov4-tiny-paddle")
    yolo = YOLO()
    r_image = yolo.detect(cv2.imread("C:/Users/311/Desktop/sheep.png"))
    plt.imshow(r_image)
    plt.show()

