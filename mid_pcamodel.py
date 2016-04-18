# -*- encoding:utf-8 -*-
"""
============================
Faces dataset decompositions
============================

This example applies to :ref:`olivetti_faces` different unsupervised
matrix decomposition (dimension reduction) methods from the module
:py:mod:`sklearn.decomposition` (see the documentation chapter
:ref:`decompositions`) .

参数：

"""
print(__doc__)

# Authors: Vlad Niculae, Alexandre Gramfort
# License: BSD 3 clause
import os

import logging
from time import time

from numpy.random import RandomState
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import MiniBatchKMeans
from sklearn import decomposition
from sklearn.externals import joblib
import pandas as pd

import sys

outpath = os.getcwd()+'\\pcaout\\'

def getXdata(X,dnamelist,datapath,classname):  
    for idx,name in enumerate(dnamelist):
        f = pd.read_excel(datapath+name+classname+'.xls')
        f = np.array(f)    
        a = np.zeros([data_row,data_col])
        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                a[i][j] = f[i][j]
        a = a.reshape(-1)
        print a.shape
        X[idx] = a
    return X


def plot_gallery(title, images,image_shape, n_col=2, n_row=3,cmap=plt.cm.gray):
    plt.figure(figsize=(2.5 * n_col, 2.26 * n_row))
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
    plt.savefig(outpath+title+'.png',facecolor='grey')
    print 'savefig'


def centered(X):
    faces = X
    n_samples, n_features = faces.shape

    # global centering
    faces_centered = faces - faces.mean(axis=0)
    # local centering
    faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)
    
    return faces_centered
    
def pcamodel(X,n_row, n_col,pcaclass):
    # Load faces data
    #dataset = fetch_olivetti_faces(shuffle=True, random_state=rng)
    
    faces = X

    n_samples, n_features = faces.shape

    # global centering
    faces_centered = faces - faces.mean(axis=0)

    # local centering
    faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

    print("Dataset consists of %d faces" % n_samples)
    
    n_components = n_row*n_col
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
    plot_gallery(pcaclass+'eign',
                 components_[:n_components],image_shape)
    
    wher = np.zeros([n_components,data_row*data_col],'float')
    for i in range(n_components):
        for j in range(data_row*data_col):
            if components_[i][j]>0.05:
                wher[i][j]=components_[i][j]
    plot_gallery('%s - Train time' % (estimatorname),
                 wher[:n_components],image_shape)    
    
    return estimator

#def 

def generate_pca(data_con_path,data_pat_path,controllist,patientlist,n_row, n_col,
                 image_shape,data_num,classname):
    
    n_components = n_row*n_col
    

    #得到控制组的X
    X = np.zeros([data_num,data_row*data_col],'float')
    X = getXdata(X,controllist,data_con_path,classname)
    
    ###############################################################################
        
    plot_gallery("original control data", X[:data_num],image_shape,n_col=4, n_row=4)
    
    estimatorname = 'Eigenfaces - RandomizedPCA'
    con_pcamodel = pcamodel(X, n_row, n_col,'control')
    components = con_pcamodel.components_
    
    con_out = np.zeros([data_num, n_components-1],'float')   #去掉第一个特征
    
    con_out = con_pcamodel.transform(centered(X))[:,1:]  #centered        
    
    pd.DataFrame(con_out).to_csv(outpath+'con_outdata'+classname+'.csv')
    
    print 'save con_pcamodel'    
    joblib.dump(con_pcamodel, outpath+'mid-pca_control'+classname+'.pkl')
    #clf = joblib.load('brain-pca_'+classname+'.pkl') 
    
    
    ########### bingren　############    
    pat_num = 21    
    X_pat = np.zeros([pat_num, data_row*data_col])
    
    X_pat = getXdata(X_pat,patientlist,data_pat_path,classname)
    plot_gallery("original pat_data", X_pat[:pat_num],image_shape,n_col=4, n_row=6)
    
    out = np.zeros([pat_num, n_components],'float')
    
    for i in range(pat_num):
        out[i] = con_pcamodel.transform(centered(X_pat)[i])  #centered    
    
    cc = np.zeros([pat_num,data_row*data_col],'float')
    for i in range(pat_num):
        for j in range(n_components):
            cc[i] = out[i][j] * components[j]
    
    ## 不要重构了。。 ##
    plot_gallery("out pat_data", cc[:pat_num],image_shape,n_col=4, n_row=6)
    """
    ### 这里是加上均值  ###
    dd = np.zeros([pat_num,data_row*data_col])
    for i in range(pat_num):
        dd[i] = X_pat[i]- cc[i]
    plot_gallery("out differ pat_data", dd[:pat_num],n_col=4, n_row=6,cmap=plt.cm.Reds)
    """

    
    #plt.show()
    #plt.close()
    
    pat_pcamodel = pcamodel(X_pat,n_row, n_col,'patient')
    
    pat_out = np.zeros([pat_num, n_components-1],'float')
    
    pat_out = pat_pcamodel.transform(centered(X_pat))[:,1:]    
    pd.DataFrame(pat_out).to_csv(outpath+'pat_outdata'+classname+'.csv')    
    
    joblib.dump(pat_pcamodel, outpath+'mid-pca_patient'+classname+'.pkl')
    
    deffer_eign = np.zeros([n_components,data_row*data_col])
    for i in range(n_components):
        for j in range(data_row*data_col):
            deffer_eign[i][j]=components[i][j] - pat_pcamodel.components_[i][j]
    
    plot_gallery('deffer_eign',deffer_eign[:n_components],image_shape,cmap=plt.cm.Reds)
    
    
    ### classification ###
    
    y_con = np.zeros(16,bool)
    y_out = np.ones(pat_num,bool)
 
 
###数据大小设置
data_row = 18
data_col = 25

data_num = 16
#图片大小设置
image_shape = (data_row, data_col)
rng = RandomState(0)    

def dopcamodel(cla="_FA"):
    
    data_con_path = os.getcwd()+'\\pcadata\\ana8-11\\control\\'
    data_pat_path = os.getcwd()+'\\pcadata\\ana8-11\\patient\\'
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
    #0.06497383,  0.05966482
    n_row, n_col = 2, 3
    
    
    classname = '_FA'
    classlist = ['_FA','_MD','_MSD','_density','_csa_gfa']
    generate_pca(data_con_path,data_pat_path,controllist,patientlist,n_row, n_col,
                     image_shape,data_num,cla)
    """
    for cla in classlist:
        generate_pca(data_con_path,data_pat_path,controllist,patientlist,n_row, n_col,
                     image_shape,data_num,cla)
    
    """
