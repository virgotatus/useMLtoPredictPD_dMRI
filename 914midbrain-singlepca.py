# -*- encoding:utf-8 -*-
"""
============================
Faces dataset decompositions
============================

This example applies to :ref:`olivetti_faces` different unsupervised
matrix decomposition (dimension reduction) methods from the module
:py:mod:`sklearn.decomposition` (see the documentation chapter
:ref:`decompositions`) .

"""
print(__doc__)

# Authors: Vlad Niculae, Alexandre Gramfort
# License: BSD 3 clause

import logging
from time import time

from numpy.random import RandomState
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import MiniBatchKMeans
from sklearn import decomposition

import pandas as pd
data_con_path = 'G:\\Userworkspace\\brain-pca\\data\\ana9-14\\control\\'
data_pat_path = 'G:\\Userworkspace\\brain-pca\\data\\ana9-14\\patient\\'
#去掉一些size不是19*20的人，这边需要预处理
controllist = ['HUANGYAXIN','LIUZHIBING','SUNAIQUAN','Tanjinxin',
               'WANGCHUNXIANG','WANGKUNYING','WANGWEI','xiaowen',
               'XIAQIAORONG','XIEGUOLIANG','YAOYOUYUAN','YUANGUISHENG',
               'zhangruiwei','zhaoaiju','ZHOUGUIQUAN','zhouyan']

patientlist = ['CHENLIANXIANG','duxiaojing','jiangguozhi',
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
data_row = 25
data_col = 40

data_num = 16
#图片大小设置
image_shape = (data_row, data_col)
rng = RandomState(0)

X = np.zeros([data_num,data_row*data_col])
classname = '1_MSD'
#得到控制组的X
for idx,name in enumerate(controllist):
    f = pd.read_excel(data_con_path+name+classname+'.xls')
    f = np.array(f)    
    a = np.zeros([data_row,data_col])
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            a[i][j] = f[i][j]
    a = a.reshape(-1)
    print a.shape
    X[idx] = a


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
###############################################################################
    
plot_gallery("original control data", X[:data_num],n_col=4, n_row=4)


###############################################################################

# Load faces data
#dataset = fetch_olivetti_faces(shuffle=True, random_state=rng)
faces = X

n_samples, n_features = faces.shape

# global centering
faces_centered = faces - faces.mean(axis=0)

# local centering
faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

print("Dataset consists of %d faces" % n_samples)


###############################################################################
# List of the different estimators, whether to center and transpose the
# problem, and whether the transformer uses the clustering API.
estimatorname = 'Eigenfaces - RandomizedPCA'
estimator = decomposition.RandomizedPCA(n_components=n_components, whiten=True)
center = True


###############################################################################
# Plot a sample of the input data

#plot_gallery("First centered Olivetti faces", faces_centered[:n_components])

###############################################################################
# Do the estimation and plot it


print("Extracting the top %d %s..." % (n_components, estimatorname))
t0 = time()
data = faces
if center:
    data = faces_centered
estimator.fit(data)
train_time = (time() - t0)
print("done in %0.3fs" % train_time)
if hasattr(estimator, 'cluster_centers_'):
    components_ = estimator.cluster_centers_
else:
    components_ = estimator.components_
if hasattr(estimator, 'noise_variance_'):
    plot_gallery("Pixelwise variance",
                 estimator.noise_variance_.reshape(1, -1), n_col=1,
                 n_row=1)
plot_gallery('%s - Train time %.1fs' % (estimatorname, train_time),
             components_[:n_components])

con_num = 16

    
#plot_gallery("origial con_data", X[:con_num],n_col=4, n_row=6)

### fit patient data ####
pat_num = 21

X_pat = np.zeros([pat_num, data_row*data_col])

for idx,name in enumerate(patientlist):
    f = pd.read_excel(data_pat_path+name+classname+'.xls')
    f = np.array(f)
    a = np.zeros([data_row,data_col])
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            a[i][j] = f[i][j]
    a = a.reshape(-1)
    print a.shape
    X_pat[idx] = a
    
plot_gallery("original pat_data", X_pat[:pat_num],n_col=4, n_row=6)

out = np.zeros([pat_num, n_components])
for i in range(pat_num):
    out[i] = estimator.transform(X_pat[i])

cc = np.zeros([pat_num,data_row*data_col])
for i in range(pat_num):
    for j in range(n_components):
        cc[i] += (out[i][j]) * components_[j]

plot_gallery("out pat_data", cc[:pat_num],n_col=4, n_row=6,cmap=plt.cm.Reds)

### 这里是加上均值 ！！！ ###
dd = np.zeros([pat_num,data_row*data_col])
for i in range(pat_num):
    dd[i] = cc[i] + faces.mean(axis=0)
plot_gallery("out differ pat_data", dd[:pat_num],n_col=4, n_row=6,cmap=plt.cm.Reds)

wher = np.zeros([n_components,data_row*data_col])
for i in range(n_components):
    for j in range(data_row*data_col):
        if components_[i][j]>0.05:
            wher[i][j]=components_[i][j]
plot_gallery('%s - Train time %.1fs' % (estimatorname, train_time),
             wher[:n_components])

plt.show()

from sklearn.externals import joblib

joblib.dump(estimator, 'brain-pca_'+classname+'.pkl') 
clf = joblib.load('brain-pca_'+classname+'.pkl') 
plt.close()


### classification ###

conpca = np.zeros([data_num, n_components])
for i in range(data_num):
    conpca[i] = estimator.transform(X[i])
    
out

y_con = np.zeros(16,bool)
y_out = np.ones(pat_num,bool)

