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

from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import MiniBatchKMeans
from sklearn import decomposition

import pandas as pd
data_con_path = 'E:\\brain\\fa&gfa experi\\control_fa\\'
data_pat_path = 'E:\\brain\\fa&gfa experi\\patient_fa\\'
#去掉一些size不是19*20的人，这边需要预处理
controllist = ['HUANGYAXIN','LIUZHIBING','SUNAIQUAN','Tanjinxin',
                   'WANGCHUNXIANG','WANGKUNYING','WANGWEI','xiaowen',
                   'XIAQIAORONG','XIEGUOLIANG',
                   'zhangruiwei','zhaoaiju','ZHOUGUIQUAN','zhouyan']

patientlist = ['CHENLIANXIANG','duxiaojing','jiangchangguo','jiangguozhi',
               'liangxiangxiu','liangxiangxiu2','limingcai','liuwuchou',
               'LIUYAN','liuyingxiao','renqixia','wangdongfang','Wangfusheng',
               'wangjinguo','WangPingAn','wangsongyuan','wangwuji',
               'Wushaoxian','XIONGYIN','yangjianjun','zhouyongzhong','ZHUCHUANGUI']
filename = 'HUANGYAXIN'

X = np.zeros([14,380])
for idx,name in enumerate(controllist):
    f = pd.read_excel(data_con_path+name+'.xls')
    f = np.array(f)
    f = f.reshape(-1)
    X[idx] = f

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
#pca的主成分个数设置
n_row, n_col = 2, 3
n_components = n_row*n_col
#图片大小设置
image_shape = (19, 20)
rng = RandomState(0)

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
def plot_gallery(title, images, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)

###############################################################################
# List of the different estimators, whether to center and transpose the
# problem, and whether the transformer uses the clustering API.
estimators = [
    ('Eigenfaces - RandomizedPCA',
     decomposition.RandomizedPCA(n_components=n_components, whiten=True),
     True),

    ('Independent components - FastICA',
     decomposition.FastICA(n_components=n_components, whiten=True),
     True),


    ('Factor Analysis components - FA',
     decomposition.FactorAnalysis(n_components=n_components, max_iter=2),
     True),
]


###############################################################################
# Plot a sample of the input data

plot_gallery("First centered Olivetti faces", faces_centered[:n_components])

###############################################################################
# Do the estimation and plot it

for name, estimator, center in estimators:
    print("Extracting the top %d %s..." % (n_components, name))
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
    plot_gallery('%s - Train time %.1fs' % (name, train_time),
                 components_[:n_components])

plt.show()
