import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import matplotlib.pyplot as plt
import os
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
import pandas as pd
import numpy as np,interp
from sklearn.metrics import *
from sklearn.preprocessing import label_binarize

from tqdm import tqdm
from model import resnet34

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

best_acc = 0.0

# 数据预处理
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 随机裁剪为224x224大小
                                 transforms.RandomHorizontalFlip(),  # 随机水平翻转
                                 transforms.ToTensor(),  # 转为Tensor
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),  # 归一化
    "val": transforms.Compose([transforms.Resize(256),  # 调整大小为256x256
                               transforms.CenterCrop(224),  # 中心裁剪为224x224大小
                               transforms.ToTensor(),  # 转为Tensor
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # 归一化
}

# 数据集路径
data_root = os.path.abspath(os.path.join(os.getcwd(), "D:/PyTorch"))
image_path = data_root + "/resnet34-ship/salt-0.01-0.03/"

# 创建训练集和验证集的数据集实例
train_dataset1 = datasets.ImageFolder(root=image_path+"subset1", transform=data_transform["train"])
train_dataset2 = datasets.ImageFolder(root=image_path+"subset2", transform=data_transform["train"])

# 创建类别索引字典并保存为JSON文件
ship_list = train_dataset1.class_to_idx
cla_dict = dict((val, key) for key, val in ship_list.items())

json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

# 设置批处理大小和数据加载器
batch_size = 64
train_loader1 = torch.utils.data.DataLoader(train_dataset1, batch_size=batch_size, shuffle=True, num_workers=0)
train_loader2 = torch.utils.data.DataLoader(train_dataset2, batch_size=batch_size, shuffle=True, num_workers=0)

# 创建验证集的数据加载器
valid_dataset = datasets.ImageFolder(root=image_path + "val1", transform=data_transform["val"])
val_num = len(valid_dataset)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# 创建模型实例并将其移动到设备上
net = resnet34(num_class=5)
net.to(device)

# 定义损失函数和优化器
loss_function = nn.CrossEntropyLoss()
loss_function.cuda()

optimizer1 = optim.Adam(net.parameters(), lr=0.000005, weight_decay=1e-2)
optimizer2 = optim.Adam(net.parameters(), lr=0.00001, weight_decay=1e-2)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=20, gamma=0.5)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=20, gamma=0.1)

# 创建保存模型的路径和相关参数列表
save_paths = ['./resNet34-lr1.pth', './resNet34-lr2.pth']
optimizers = [optimizer1, optimizer2]
schedulers = [scheduler1, scheduler2]
train_loaders = [train_loader1, train_loader2]

# 创建保存训练结果的DataFrame并保存为CSV文件
df = pd.DataFrame(columns=['epoch','train_loss','train_acc','val_loss', 'val_acc'])
df.to_csv("D:/PyTorch/resnet34-ship/csv/lr/3/1.csv")

# 循环训练两个不同的优化器参数配置
for i in range(2):
    save_path = save_paths[i]
    optimizer = optimizers[i]
    scheduler = schedulers[i]
    train_loader = train_loaders[i]

    best_acc = 0.0
    for epoch in range(1, 11):
        # 训练
        net.train()
        train_total = 0
        train_correct = 0
        train_loss = 0
        for batch in tqdm(train_loader):
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(imgs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            predicted = torch.argmax(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            del imgs, labels
            torch.cuda.empty_cache()
        scheduler.step()
        train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        # 验证
        net.eval()
        valid_correct, valid_total, valid_loss = 0, 0, 0
        for batch in tqdm(valid_loader):
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = net(imgs)
            loss = loss_function(outputs, labels)
            predicted = torch.max(outputs.data, 1)[1]
            valid_correct += (predicted == labels).sum().item()
            valid_loss += loss.item()
            valid_total += labels.size(0)
            del imgs, labels
            torch.cuda.empty_cache()
        valid_accuracy = valid_correct / valid_total
        valid_loss = valid_loss / len(valid_loader)

        if valid_accuracy > best_acc:
            best_acc = valid_accuracy
            torch.save(net.state_dict(), save_path)

        print(f'| Epoch: {epoch:02}')
        print(f'| Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy * 100:.3f}%')
        print(f'| Val. Loss: {(valid_loss):.4f} | Val. Acc: {valid_accuracy * 100:.3f}%')
        print(f"last, best acc [{valid_accuracy:.5f}]")
        print('=' * 100)

        jc = "%d" % (epoch)
        train_loss = '%.4f' % (train_loss)
        train_acc = '%.4f' % (train_accuracy * 100)
        val_loss = '%.4f' % (valid_loss)
        val_acc = '%.4f' % (valid_accuracy * 100)
