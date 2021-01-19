import os
import time
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils, models
from torchvision.transforms import *
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix # 混淆矩阵

def plot_confusion_matrix(num_classes, true_labels, pred_labels, 
                          path="./confusion_matrix.png"):
    """
    画混淆矩阵。
    Args:
        num_classes: 类别数量。
        true_labels: 真实标签。
        pred_labels: 预测出的结果。
        path: 混淆矩阵热力图保存路径。
    """
    labels = range(num_classes)
    cm = confusion_matrix(true_labels, pred_labels, labels)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion Matrix', fontsize = 18)
    plt.savefig(path)
    plt.close()

def main():
    # 计算设备
    DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载设置
    with open('./config.json') as data_file:
        config = json.load(data_file)
    # 输出文件夹
    ts = time.asctime(time.localtime(time.time()))
    RESULT_FOLDER = str(ts)+'/'
    if not os.path.exists(os.path.join(config['output_dir'], RESULT_FOLDER)):
        if not (os.path.exists(os.path.join(config['output_dir']))):
            os.mkdir(os.path.join(config['output_dir']))
        os.mkdir(os.path.join(config['output_dir'], RESULT_FOLDER))
    RESULT_FOLDER = os.path.join(config['output_dir'], RESULT_FOLDER)
    # Tensorboard
    writer = SummaryWriter()
    # 设置/参数
    EPOCH_NUM = config['num_epochs']
    BATCH_SIZE = config['batch_size']
    LR = config['lr']
    # 预处理操作
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
        ]),
        'val': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
        ]),
    }
    # 加载训练集
    data_dir = config['train_data_folder']
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    # 构建网络
    net = models.vgg16(pretrained=True)
    # 修改vgg16的最后几层全连接层
    net.classifier = Sequential(nn.Linear(in_features=25088, out_features=4096, bias=True),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(in_features=4096, out_features=1024, bias=True),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(in_features=1024, out_features=len(class_names), bias=True)
                                )

    # 将网络放在对应的计算设备
    net = net.to(DEVICE)
    # 创建优化器
    optimizer = torch.optim.SGD([
                                 {'params': net.features.parameters(), 'lr':1e-5},
                                 {'params': net.classifier.parameters()},
                                ], lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # 定义Loss函数（交叉熵）
    loss_F=torch.nn.CrossEntropyLoss()
    for epoch in range(EPOCH_NUM):
        # 训练与验证交替进行
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                net.train()
            else:
                net.eval()
                # 保存验证集标签与预测结果，用于画混淆矩阵
                labels_array=np.array([])
                preds_array=np.array([])

            running_loss = 0.0
            running_corrects = 0

            for i, (instances, labels) in enumerate(dataloaders[phase]):
                print(i)
                instances = instances.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    labels = labels.view(-1)
                    outputs = net(instances)
                    _, preds = torch.max(outputs, 1)
                    loss=loss_F(outputs, labels) 
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        # 画loss曲线
                        writer.add_scalar('scalar/loss', loss.data, epoch*(dataset_sizes['train']/BATCH_SIZE)+i)
                    else:
                        labels_array = np.concatenate((labels_array, labels.to('cpu').numpy()), axis=0)
                        preds_array = np.concatenate((preds_array, preds.to('cpu').numpy()), axis=0)
                # 统计loss与正确数量
                running_loss += loss.item() * instances.size(0)
                running_corrects += torch.sum(preds == labels.data)
            #　计算整个eopch的loss与精度
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            # 保存统计的loss与精度
            if phase == 'train':
                train_loss = epoch_loss
                train_acc = epoch_acc
            else:
                val_loss = epoch_loss
                val_acc = epoch_acc
        # 打印训练结果
        print('Epoch: {}\t|　train -　Loss: {:.4f}, Acc: {:.2f}%\t|　val -　Loss: {:.4f}, Acc: {:.2f}%'.format(epoch, train_loss, train_acc*100.0, val_loss, val_acc*100.0))
        # 画精度曲线
        writer.add_scalars('scalar/accuracy', {'train': train_acc, 'val':val_acc}, epoch)
        plot_confusion_matrix(len(class_names), labels_array, preds_array)
    # 保存模型
    torch.save(net.state_dict(), "./{}/model.pth".format(RESULT_FOLDER))
    writer.close()

if __name__ == "__main__":
    main()
