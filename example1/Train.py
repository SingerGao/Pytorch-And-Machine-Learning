import os
import time
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import *
from Model import Net
from Dataloader import OXFlowerDataset


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

    # 设置/参数
    EPOCH_NUM = config['num_epochs']
    BATCH_SIZE = config['batch_size']
    LR = config['lr']
    # 预处理操作
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), #将图片转换为Tensor
        transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
        ])
    # 加载训练集
    ox_flower_dataset = OXFlowerDataset(root_dir=config['train_data_folder'], 
                                    transform=transform)

    train_loader = DataLoader(ox_flower_dataset, 
                            batch_size=BATCH_SIZE,
                            shuffle=True, 
                            num_workers=config['num_workers'])

    # 创建网络
    net = Net().to(DEVICE)
    print(net)
    # 创建优化器
    optimizer=torch.optim.Adam(net.parameters(),lr=LR)
    # 定义Loss函数（交叉熵）
    loss_F=torch.nn.CrossEntropyLoss()
    # Training loop
    for epoch in range(EPOCH_NUM):
        avg_loss = 0
        running_acc = 0
        total = 0
        for i, (instances, labels) in enumerate(train_loader):
            # 更新网络参数
            labels = labels.view(-1)
            outputs = net(instances.to(DEVICE))
            loss=loss_F(outputs, labels.to(DEVICE)) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #　记录loss与精确度
            total += labels.size(0)
            avg_loss = (avg_loss*i+ loss)/(i+1)
            _, pred = torch.max(outputs, 1)
            correct_num = (pred.to("cpu") == labels).sum()
            running_acc += correct_num.data
            # 间隔输出训练信息
        running_acc = 100.0 * running_acc.float() / float(total)
        print("Epoch: {}/{}\t|loss:{:.4f}\t|acc:{:.2f}%".format(epoch, EPOCH_NUM, avg_loss, running_acc))
    # 保存模型
    torch.save(net.state_dict(), "./{}/model.pth".format(RESULT_FOLDER))

if __name__ == "__main__":
    main()
