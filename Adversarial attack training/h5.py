import math
import argparse
import h5py
import numpy as np
import h5py

def print_h5_dataset(dataset, name=''):
    if isinstance(dataset, h5py.Dataset):
        print(name, dataset.shape, dataset.dtype)
    elif isinstance(dataset, h5py.Group):
        print(name, "[Group]")
        for key in dataset.keys():
            print_h5_dataset(dataset[key], name + '/' + key)

# 读取.h5文件
h5_file = h5py.File('model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5', 'r')

# 打印文件内容
print_h5_dataset(h5_file)

# 关闭.h5文件
h5_file.close()
