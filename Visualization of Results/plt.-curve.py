import csv
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib as mpl
import numpy as np
import pandas as pd

exampleFile = open('D:/PyTorch/OSL-pt/csv/Loss&acc/embedding/Ships-embedding.csv')  # 打开csv文件
exampleReader = csv.reader(exampleFile)  # 读取csv文件
exampleData= list(exampleReader)  # csv数据转换为列表
length_hang = len(exampleData)  # 得到数据行数
length_lie = len(exampleData[0])  # 得到每行长度

x = list()


y1 = list()
y2 = list()
y3 = list()
y4 = list()
y5 = list()
y6 = list()
y7 = list()
y8 = list()


for i in range(1, 31):
    x.append(float(exampleData[i][0]))  # 将第1列数据从第1行读取到最后一行赋给列表x
    y1.append(float(exampleData[i][1]))
    y2.append(float(exampleData[i][2]))
    y3.append(float(exampleData[i][3]))
    y4.append(float(exampleData[i][4]))
    y5.append(float(exampleData[i][5]))
    y6.append(float(exampleData[i][6]))
    y7.append(float(exampleData[i][7]))
    y8.append(float(exampleData[i][8]))



x_major_locator = MultipleLocator(5)
y1_major_locator = MultipleLocator(10)
#y2_major_locator = MultipleLocator(20)

ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y1_major_locator)
#ax.yaxis.set_major_locator(y2_major_locator)



fig,ax = plt.subplots(1)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
a = 2
b = 0.25
lw= 2
ms = 8
plt.grid(True,color="darkgray",axis="both",ls=":",lw=1)
plt.xlim(0,30.1)
plt.ylim(68,98)

colors = [(r / 255, g / 255, b / 255) for r, g, b in [(2, 48, 71), (14, 91, 118), (26, 134, 163),
                                                      (70, 172, 202), (155, 207, 232), (243, 249, 252),
                                                      (255, 202, 95), (254, 168, 9), (251, 132, 2)]]

#(2/ 255, 48/ 255, 71/ 255), (14/ 255, 91/ 255, 118/ 255), (26/ 255, 134/ 255, 163/ 255),
#(70/ 255, 172/ 255, 202/ 255), (155/ 255, 207/ 255, 232/ 255), (243/ 255, 249/ 255, 252/ 255),
#(255/ 255, 202/ 255, 95/ 255), (254/ 255, 168/ 255, 9/ 255), (251/ 255, 132/ 255, 2/ 255)]


plt.plot(x,y1,label=a**6,marker='v',markersize=ms-2,lw=lw,color=(2/ 255, 48/ 255, 71/ 255))
plt.plot(x,y2,label=a**7,marker='p',markersize=ms-2,lw=lw,color=(14/ 255, 91/ 255, 118/ 255))
plt.plot(x,y3,label=a**8,marker='.',markersize=ms+2,lw=lw,color=(26/ 255, 134/ 255, 163/ 255))
plt.plot(x,y4,label=a**9,marker=6,markersize=ms,lw=lw,color=(155/ 255, 207/ 255, 232/ 255))
plt.plot(x,y5,label=a**10,marker='*',markersize=ms-2,lw=lw,color=(254/ 255, 168/ 255, 9/ 255))
plt.plot(x,y6,label=a**11,marker='v',markersize=ms-2,lw=lw,color=(251/ 255, 132/ 255, 2/ 255))
plt.plot(x,y7,label=a**12,marker='p',markersize=ms-2,lw=lw,color=(255/ 255, 202/ 255, 95/ 255))
plt.plot(x,y8,label=a**13,marker='.',markersize=ms-2,lw=lw,color=(70/ 255, 172/ 255, 202/ 255))

plt.xlabel("Epoch",fontsize=14,family='Times New Roman')#vector-dimension
plt.ylabel("Accuracy(%)",fontsize=14,family='Times New Roman')
#plt.title("Accuracy(%)",fontsize=16,family='Times New Roman')

plt.xticks(fontsize=12,family='Times New Roman')#,rotation= 45
plt.yticks(fontsize=12,family='Times New Roman')
plt.legend(loc="lower center",prop={'family' : 'Times New Roman','size':'11'},ncol=4)

plt.tight_layout()
plt.show()