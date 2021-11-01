import numpy as np
import paddle, os
from PIL import Image
from paddle.io import Dataset, DataLoader

rand = lambda a=0, b=1: np.random.rand() * (b-a) + a
FloatTensor = lambda x: paddle.to_tensor(x, dtype=paddle.float32)

class YoloDataset(Dataset):
    def __init__(self, annotation_file, img_root, is_train=True):
        super(YoloDataset, self).__init__()
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
        self.lines = lines[int(0.1*len(lines)):] if is_train else lines[:int(0.1*len(lines))] # 数据集划分
        self.img_root = img_root

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        lines = self.lines
        n = len(self.lines)
        index = index % n

        input_size = 416
        annotation_line = lines[index] # format: imagepath yolobox1(x,y,w,h,cls) yolobox2 ...
        path = annotation_line.split()[0]
        boxes = annotation_line.split()[1:]
        image = Image.open(os.path.join(self.img_root, path))
        iw, ih = image.size
        boxes = FloatTensor([ list(map(float, box.split(','))) for box in boxes])

        scale = min(input_size/iw, input_size/ih)
        scaled_by_width = (input_size/iw) < (input_size/ih)
        scaled_image_size = (input_size, int(ih*scale)) if scaled_by_width else (int(iw*scale), input_size)
        scaled_image = image.resize(scaled_image_size, Image.BICUBIC)

        dx = (input_size-scaled_image.size[0])//2
        dy = (input_size-scaled_image.size[1])//2
        
        padding_image = Image.new('RGB', (input_size,input_size), (128,128,128))
        padding_image.paste(scaled_image, (dx, dy))

        if scaled_by_width: # 按宽度缩放，x轴比例不变
            boxes[:, 1] = (boxes[:, 1]*scaled_image.size[1] + dy) / input_size
            boxes[:, 3] = boxes[:, 3]*scaled_image.size[1] / input_size
        else:
            boxes[:, 0] = (boxes[:, 0]*scaled_image.size[0] + dx) / input_size
            boxes[:, 2] = boxes[:, 2]*scaled_image.size[0] / input_size

        return (np.array(padding_image, dtype=np.float32)/255.0).transpose((2, 0, 1)), boxes


# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    return paddle.to_tensor(images), bboxes

if __name__ == '__main__':
    ### test function ###
    def F(tensor):
        print(f"sum:{tensor.sum().item()}, mean:{tensor.mean().item()}, min-max:[{tensor.min().item()}-{tensor.max().item()}]")
    ### test function ###

    train_dataset   = YoloDataset('conf/annotation.txt',"D:/File/Dataset/VOC2007/JPEGImages", is_train=True)
    # img, box = train_dataset[100]
    # print(box)
    # import matplotlib.pyplot as plt
    # img=img.transpose([1,2,0])
    # bbox = np.zeros([4])
    # bbox[:2] = box[0, :2] - box[0, 2:4]/2
    # bbox[2:] = box[0, :2] + box[0, 2:4]/2
    # bbox = (416*bbox).astype(np.int64)
    # import cv2
    # cv2.rectangle(img, bbox[:2], bbox[2:], (255,0,0),2)
    # plt.imshow(img)
    # plt.show()

    gen             = DataLoader(train_dataset, shuffle=True, batch_size=10,
                            drop_last=True, collate_fn=yolo_dataset_collate)
    
    for idx, batch in enumerate(gen()):
        img, target = batch
        print(img.shape)
        print(target)
        break
    # print(train_dataset[10][0].shape)
    # print(train_dataset[10][1])