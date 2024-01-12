import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import resnet34,vgg16,alexnet
#from VGG_model import vgg
from utils import LoadData, write_result
import os
from torchvision import transforms
import pandas as pd
from PIL import Image
from tqdm import tqdm

def eval(dataloader, model):
    label_list = []
    likelihood_list = []
    pred_list = []
    model.eval()
    with torch.no_grad():
        # 加载数据加载器，得到里面的X（图片数据）和y(真实标签）
        for X, y in tqdm(dataloader, desc="Model is predicting, please wait"):
            # 将数据转到GPU
            X = X.cuda()
            # 将图片传入到模型当中就，得到预测的值pred
            pred = model(X)

            pred_softmax = torch.softmax(pred,1).cpu().numpy()
            # 获取可能性最大的标签
            label = torch.softmax(pred,1).cpu().numpy().argmax()
            label_list.append(label)
            # 获取可能性最大的值（即概率）
            likelihood = torch.softmax(pred,1).cpu().numpy().max()
            likelihood_list.append(likelihood)
            pred_list.append(pred_softmax.tolist()[0])

        return label_list,likelihood_list, pred_list


if __name__ == "__main__":

    model_name = resnet34()
    pth_loc="./resNet34-6-salt.pth"
    image_tags_corresponding = "test.txt"
    label = ["Cargo", "Carrier","Cruise","Military","Tankers"]

    '''
        加载预训练模型
    '''
    # 1. 导入模型结构
    model = model_name
    #num_ftrs = model.fc.to()   # 获取全连接层的输入
    #model.fc = nn.Linear(num_ftrs, 5)  # 全连接层改为不同的输出
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. 加载模型参数

    model_loc = pth_loc
    #model.load_state_dict(torch.load(weights_path, map_location=device))
    model_dict = torch.load(model_loc)
    model.load_state_dict(model_dict)
    model = model.to(device)

    '''
       加载需要预测的图片
    '''
    valid_data = LoadData(image_tags_corresponding, train_flag=False)
    test_dataloader = DataLoader(dataset=valid_data, num_workers=4, pin_memory=True, batch_size=1)


    '''
      获取结果
    '''
    # 获取模型输出
    label_list, likelihood_list, pred =  eval(test_dataloader, model)

    # 将输出保存到exel中，方便后续分析
    label_names = label     # 可以把标签写在这里
    df_pred = pd.DataFrame(data=pred, columns=label_names)

    df_pred.to_csv('pred_result.csv', encoding='gbk', index=False)
    print("Done!")

    print(f'| Epoch: {epoch + 1}')
    print(f'| Train Loss: {train_loss:.5f} | Train Acc: {train_accuracy * 100:.5f}%')
    print(f'| Val. Loss: {(valid_loss):.5f} | Val. Acc: {valid_accuracy * 100:.5f}%')
    print('=' * 100)

    jc = "%d" % (epoch + 1)
    train_loss = '%.5f' % (train_loss)
    train_acc = '%.5f' % (train_accuracy * 100)
    val_loss = '%.5f' % (valid_loss)
    val_acc = '%.5f' % (valid_accuracy * 100)
    list = [jc, train_loss, train_acc, val_loss, val_acc]
    data = pd.DataFrame([list])
    data.to_csv('./last.csv', mode='a', header=False, index=False)

print(f"last,least loss [{least_loss:.5f}] best acc [{best_acc:.5f}]")
print('Finished Training')

net_loc = save_path
net_dict = torch.load(net_loc)
net.load_state_dict(net_dict)

label_list, likelihood_list, pred =  eval(test_dataloader, model)

label_names = label     # 可以把标签写在这里
df_pred = pd.DataFrame(data=pred, columns=label_names)
df_pred.to_csv('pred_result-0.csv', encoding='gbk', index=False)

print("Done!")