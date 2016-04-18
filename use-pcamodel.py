# -*- coding:utf-8 -*-
from sklearn.externals import joblib

estimator = joblib.load('brain-pca.pkl') 

import logging
from time import time

from numpy.random import RandomState
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import MiniBatchKMeans
from sklearn import decomposition

import pandas as pd
data_con_path = 'G:\\Userworkspace\\brain-pca\\data\\ana8-11\\control\\'
data_pat_path = 'G:\\Userworkspace\\brain-pca\\data\\ana8-11\\patient\\'
#去掉一些size不是19*20的人，这边需要预处理
controllist = ['HUANGYAXIN','LIUZHIBING','SUNAIQUAN','Tanjinxin',
               'WANGCHUNXIANG','WANGKUNYING','WANGWEI','xiaowen',
               'XIAQIAORONG','XIEGUOLIANG','YAOYOUYUAN','YUANGUISHENG',
               'zhangruiwei','zhaoaiju','ZHOUGUIQUAN','zhouyan']

patientlist = ['CHENLIANXIANG','duxiaojing','jiangchangguo','jiangguozhi',
               'liangxiangxiu','liangxiangxiu2','limingcai','liuwuchou',
               'liuyingxiao','renqixia','wangdongfang','Wangfusheng',
               'wangjinguo','WangPingAn','wangsongyuan','wangwuji',
               'Wushaoxian','XIONGYIN','yangjianjun','zhouyongzhong','ZHUCHUANGUI']
filename = 'HUANGYAXIN'


##############################
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
#pca的主成分个数设置:前6个特征分数为0.28424068,  0.11663118,  0.09604093,  0.08072725,  0.07188693,
#        0.06497383,  0.05966482
n_row, n_col = 2, 2
n_components = n_row*n_col

###数据大小设置
data_row = 15
data_col = 22

data_num = 16
#图片大小设置
image_shape = (data_row, data_col)
rng = RandomState(0)

X = np.zeros([data_num,data_row*data_col])
classname = '_FA'

def plot_gallery(title, images, n_col=n_col, n_row=n_row,cmap=plt.cm.gray):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=cmap,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
### fit patient data ####
pat_num = 21

X_pat = np.zeros([pat_num, data_row*data_col])
classname = '_FA'
for idx,name in enumerate(patientlist):
    f = pd.read_excel(data_pat_path+name+classname+'.xls')
    f = np.array(f)
    a = np.zeros([data_row,data_col])
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            a[i][j] = f[i][j]
    a = a.reshape(-1)
    #print a.shape
    X_pat[idx] = a
    
plot_gallery("origial pat_data", X_pat[:pat_num],n_col=4, n_row=6)

out = np.zeros([pat_num, n_components])
for i in range(pat_num):
    out[i] = estimator.transform(X_pat[i])

cc = np.zeros([pat_num,data_row*data_col])
for i in range(pat_num):
    for j in range(n_components):
        cc[i] += (out[i][j]) * estimator.components_[j]

plot_gallery("out pat_data", cc[:pat_num],n_col=4, n_row=6,cmap=plt.cm.Reds)


plt.show()
