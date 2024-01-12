import csv
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib as mpl
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import csv

exampleFile = open('./csv/average_roc_Net.csv')  # 打开csv文件
exampleReader = csv.reader(exampleFile)  # 读取csv文件
exampleData = list(exampleReader)

length_lie = len(exampleData)
length_hang = len(exampleData[0])

print(length_lie)

x1 = list()
x2 = list()
x3 = list()
x4 = list()
x5 = list()
x6 = list()
x7 = list()
x8 = list()
x9 = list()

y1 = list()
y2 = list()
y3 = list()
y4 = list()
y5 = list()
y6 = list()
y7 = list()
y8 = list()
y9 = list()

for i in range(1, 260):
    x1.append(float(exampleData[i][0]))  # 将第1列数据从第1行读取到最后一行赋给列表x
    y1.append(float(exampleData[i][1]))

for i in range(1, 140):
    x2.append(float(exampleData[i][2]))
    y2.append(float(exampleData[i][3]))

for i in range(1, 163):
    x3.append(float(exampleData[i][4]))
    y3.append(float(exampleData[i][5]))

for i in range(1, 153):
    x4.append(float(exampleData[i][6]))
    y4.append(float(exampleData[i][7]))

for i in range(1, 193):
    x5.append(float(exampleData[i][8]))
    y5.append(float(exampleData[i][9]))

for i in range(1, 138):
    x6.append(float(exampleData[i][10]))
    y6.append(float(exampleData[i][11]))

for i in range(1, 162):
    x7.append(float(exampleData[i][12]))
    y7.append(float(exampleData[i][13]))

for i in range(1, 169):
    x8.append(float(exampleData[i][14]))
    y8.append(float(exampleData[i][15]))

for i in range(1, 72):
    x9.append(float(exampleData[i][16]))
    y9.append(float(exampleData[i][17]))


x_major_locator = MultipleLocator(0.1)
y1_major_locator = MultipleLocator(0.1)
#y2_major_locator = MultipleLocator(20)

ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y1_major_locator)
#ax.yaxis.set_major_locator(y2_major_locator)


fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

lw= 2.25
ms = 3

x = [0, 1]
y = [0, 1]

plt.plot(x, y,linestyle = "--", linewidth=lw,color='darkgray')
#plt.grid(True,color="darkgray",axis="both",ls="-.",lw=0.5)
plt.xlim(-0.02,1.02)
plt.ylim(-0.02,1.02)

models = ["LeNet",
     "AlexNet",
     "VGG11",
     "ResNet34",
     "ResNet101",
     "DenseNet121",
     "MobileNetV3",
     "GoogLeNet",
     "AMAM-Net"]

colors = [(r / 255, g / 255, b / 255) for r, g, b in
          [(2, 48, 71), (14, 91, 118), (26, 134, 163),
           (70, 172, 202), (155, 207, 232), (243, 249, 252),
           (255, 202, 95), (254, 168, 9), (251, 132, 2)]]

auc_values = ['0.66', '0.91', '0.87', '0.92', '0.81', '0.93', '0.86', '0.91', '0.96']

for i in range(len(models)):
    plt.plot(eval(f'x{i+1}'), eval(f'y{i+1}'), label=f'{models[i]}', lw=lw, color=colors[i])

plt.xlabel("False Positive Rate",fontsize=16,family='Times New Roman')
plt.ylabel("True Positive Rate",fontsize=16,family='Times New Roman')

plt.xticks(fontsize=12,family='Times New Roman')
plt.yticks(fontsize=12,family='Times New Roman')

handles, labels = plt.gca().get_legend_handles_labels()
labels = [f'{label} (AUC-area = {auc})' for label, auc in zip(labels, auc_values)]
num1 = 1.01
num2 = 0.05
num3 = 3
num4 = 0

plt.legend(handles, labels, fontsize=15, loc='best', prop={'family': 'Times New Roman', 'size': '11'},frameon=True)



plt.tight_layout()
plt.show()