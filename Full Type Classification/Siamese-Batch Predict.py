import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
import os
from torchvision.models import resnet34

# 定义Siamese网络
class Siamese(nn.Module):
    def __init__(self, embedding_dim):
        super(Siamese, self).__init__()
        self.embedding_dim = embedding_dim

        # 加载预训练的ResNet34模型
        self.resnet = resnet34(pretrained=False)

        # 冻结conv5_block1_out之前的所有网络层
        trainable = False
        for name, param in self.resnet.named_parameters():
            if name == "layer4.0.conv1.weight":
                trainable = True
            param.requires_grad = trainable

        # 替换最后一层全连接层
        self.resnet.fc = nn.Linear(512, embedding_dim)

        # 创建共享权重的模块
        self.shared_fc = nn.Linear(embedding_dim, embedding_dim)

    def forward_once(self, x):
        output = self.resnet(x)
        output = self.shared_fc(output)
        return output

    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        return output1, output2, output3

def normalize(embedding):
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm

def euclidean_similarity(query_embedding, support_embeddings):
    distances = np.linalg.norm(support_embeddings - query_embedding, axis=1)
    similarities = 1 / (1 + distances)
    similarities_percentage = similarities * 100
    return similarities_percentage

model = Siamese(embedding_dim=128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
model.load_state_dict(torch.load('Ships dataset-pre-80.5692.pth', map_location=torch.device('cuda')))

df = pd.DataFrame(columns=['similarity'])
df.to_csv("D:/PyTorch/OSL-pt/emb-2.csv")

# 加载数据集
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model.eval()
best_acc = 0.45
query_dir = 'D:/PyTorch/OSL-pt/dataset_total/query'
support_dir = 'D:/PyTorch/OSL-pt/dataset_total/support'
support_dataset = []
for class_name in os.listdir(support_dir):
    class_dir = os.path.join(support_dir, class_name)
    if os.path.isdir(class_dir):
        support_images = []
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            support_image = Image.open(image_path).convert('RGB')
            support_tensor = transform(support_image).unsqueeze(0)
            support_tensor = support_tensor.to('cuda')
            support_images.append(support_tensor)
        # 计算支持集类别平均特征向量
        support_embeddings = [model.forward_once(support_image).detach().cpu().numpy() for support_image in
                                  support_images]
        support_embedding = np.mean(support_embeddings, axis=0)
        support_dataset.append((class_name, support_embedding))

    # 遍历 query 目录下的所有图像
correct_num = 0  # 记录预测正确的数量
total_num = 0  # 记录查询集下所有图片的数量
for class_name in os.listdir(query_dir):
    class_dir = os.path.join(query_dir, class_name)
    if os.path.isdir(class_dir):
        for image_name in os.listdir(class_dir):
            total_num += 1  # 查询集下所有图片数量加一
            image_path = os.path.join(class_dir, image_name)
            query_image = Image.open(image_path).convert('RGB')
            query_tensor = transform(query_image).unsqueeze(0)
            query_tensor = query_tensor.to('cuda')

                # 提取特征向量
            query_embedding = model.forward_once(query_tensor).detach().cpu().numpy()

                # 预测标签
            similarities = []

            for support_label, support_embedding in support_dataset:
                    # 计算余弦相似度
                similarity_cosine = np.dot(query_embedding, support_embedding.T) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(support_embedding))
                similarities.append(similarity_cosine)

            similarities = np.array(similarities)
            # 综合计算相似度
            labels = np.argmax(similarities)
            similarity = np.round(similarities[labels], decimals=9)

                # 判断预测是否正确
            if support_dataset[labels][0] == class_name:
                correct_num += 1

            print(f'Correct Predictions: {correct_num}')

accuracy = correct_num / total_num
print(f'Accuracy: {accuracy:.2%}')

print(f'Correct Predictions: {correct_num}')
print(f'Total Predictions: {total_num}')
