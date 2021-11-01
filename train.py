#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import paddle
import paddle.optimizer as optim
from paddle.io import DataLoader
from tqdm import tqdm

from yolo4_tiny import YoloBody
from yolo_loss import YOLOLoss
from dataloader import YoloDataset, yolo_dataset_collate


def fit_one_epoch(net, loss_fn, optimizer, epoch, epochs, batch_len, gen, train=True):
    if train: net.train()
    else: net.eval()

    total_loss = 0

    with tqdm(total=batch_len,desc=f'Epoch {epoch}/{epochs}',mininterval=0.3) as pbar:
        for batch_idx, batch in enumerate(gen):
            if batch_idx >= batch_len: break

            losses ,num_pos_all= [], 0
            images, targets = batch[0], batch[1]
            optimizer.clear_grad()
            
            if train:
                outputs = net(images)
                for i in range(2):
                    loss_item, num_pos = loss_fn(outputs[i], targets)
                    losses.append(loss_item)
                    num_pos_all += num_pos

                loss = sum(losses) / num_pos_all
                loss.backward()
                optimizer.step()
            
            else:
                with paddle.no_grad():
                    outputs = net(images)
                    for i in range(2):
                        loss_item, num_pos = loss_fn(outputs[i], targets)
                        losses.append(loss_item)
                        num_pos_all += num_pos

                    loss = sum(losses) / num_pos_all

            total_loss += loss.item()
            
            pbar.set_postfix({'total_loss': total_loss / (batch_idx + 1), 'lr':optimizer.get_lr(), 'train':train})
            pbar.update(1)
    if train:
        paddle.save(net.state_dict(), 'logs/Epoch%d-Loss%.4f.pd'%((epoch+1),total_loss/(batch_len+1)))

#----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
if __name__ == "__main__":
    import os
    os.chdir("D:/File/seaDrive/97052426/我的资料库/云文件/project/python/pytorch/yolov4-tiny-paddle")

    anchor_path = 'conf/yolo_anchors.txt'
    model_path = None
    class_path = 'conf/coco_classes.txt'
    annotation_path = 'conf/annotation.txt'
    img_root = "D:/File/Dataset/VOC2007/JPEGImages"
    #-------------------------------#
    #   所使用的注意力机制的类型
    #   {0: None, 1:SE, 2:CBAM, 3:ECA}
    #-------------------------------#
    phi = 0
    #------------------------------------------------------#
    #   是否对损失进行归一化，用于改变loss的大小
    #   用于决定计算最终loss是除上batch_size还是除上正样本数量
    #------------------------------------------------------#
    normalize = False
    #------------------------------------------------------#
    #   Yolov4的tricks应用
    #   Cosine_scheduler 余弦退火学习率 True or False
    #   label_smoothing 标签平滑 0.01以下一般 如0.01、0.005
    #------------------------------------------------------#
    mosaic = False
    Cosine_lr = False
    smoooth_label = 0

    #---------------------------------------------------#
    #   获得类和先验框
    #---------------------------------------------------#
    with open(class_path, 'r') as f:
        num_classes = len(f.readlines())

    with open(anchor_path, 'r') as f:
        anchors = f.readline()
    anchors = paddle.to_tensor([float(x) for x in anchors.split(',')]).reshape([-1,2])

    net = YoloBody(len(anchors)//2, num_classes)
    if model_path:
        net.load_dict(paddle.load(model_path))
        print('Load Finished!')

    yolo_loss = YOLOLoss(anchors, num_classes, smoooth_label, normalize)

    #------------------------------------------------------#
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    lr              = 1e-3
    Batch_size      = 32
    Freeze_Epoch    = 50
    Freeze_Stage = [1e-3, 32, True] # [lr, batch_size, Freeze
    unFreeze_Stage = [1e-4, 16, False] # [lr, batch_size, Freeze
    #------------------------------------#
    #   冻结一定部分训练，待训练误差小解冻
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #------------------------------------#
    for lr, Batch_size, Freeze in [Freeze_Stage, unFreeze_Stage]:
        #----------------------------------------------------------------------------#
        #   我在实际测试时，发现optimizer的weight_decay起到了反作用，
        #   所以去除掉了weight_decay，大家也可以开起来试试，一般是weight_decay=5e-4
        #----------------------------------------------------------------------------#
        scheduler       = optim.lr.StepDecay(lr, step_size=1, gamma=0.92)
        optimizer       = optim.Adam(parameters=net.parameters(),learning_rate=scheduler )

        train_dataset   = YoloDataset(annotation_path, img_root, is_train=True)
        val_dataset     = YoloDataset(annotation_path, img_root, is_train=False)
        gen             = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size,
                                drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val         = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size,
                                drop_last=True, collate_fn=yolo_dataset_collate)

        batch_len      = len(train_dataset) // Batch_size
        batch_len_val  = len(val_dataset) // Batch_size
        
        if batch_len == 0 or batch_len_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        for param in net.backbone.parameters():
            param.stop_gradient = Freeze

        epoch_start = 0 if Freeze else 50
        epoch_end   = 50 if Freeze else 100
        for epoch in range(epoch_start, epoch_end):
            fit_one_epoch(net, yolo_loss, optimizer, epoch, Freeze_Epoch, batch_len, gen)
            fit_one_epoch(net, yolo_loss, optimizer, epoch, Freeze_Epoch, batch_len_val, gen_val)
            # train_one_epoch(net,yolo_loss,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch)
            scheduler.step()

