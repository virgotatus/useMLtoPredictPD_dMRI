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
import pylab as plt
from matplotlib.patches import Rectangle
import numpy as np
import nibabel as nib

from sklearn.cluster import MiniBatchKMeans
from sklearn import decomposition
from sklearn import feature_selection
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.learning_curve import learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

import os
import pandas as pd
from sklearn.externals import joblib
from os.path import join as pjoin
from nilearn import plotting
n_row, n_col = 2, 2
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
"""
def plot_haxby(activation, title):
    z = 25

    fig = plt.figure(figsize=(4, 5.4))
    fig.subplots_adjust(bottom=0., top=1., left=0., right=1.)
    plt.axis('off')
    # pl.title('SVM vectors')
    plt.imshow(mean_img[:, 4:58, z].T, cmap=plt.cm.gray,
              interpolation='nearest', origin='lower')
    plt.imshow(activation[:, 4:58, z].T, cmap=plt.cm.hot,
              interpolation='nearest', origin='lower')

    mask_house = nib.load(h.mask_house[0]).get_data()
    mask_face = nib.load(h.mask_face[0]).get_data()

    plt.contour(mask_house[:, 4:58, z].astype(np.bool).T, contours=1,
            antialiased=False, linewidths=4., levels=[0],
            interpolation='nearest', colors=['blue'], origin='lower')

    plt.contour(mask_face[:, 4:58, z].astype(np.bool).T, contours=1,
            antialiased=False, linewidths=4., levels=[0],
            interpolation='nearest', colors=['limegreen'], origin='lower')

    p_h = Rectangle((0, 0), 1, 1, fc="blue")
    p_f = Rectangle((0, 0), 1, 1, fc="limegreen")
    plt.legend([p_h, p_f], ["house", "face"])
    plt.title(title, x=.05, ha='left', y=.90, color='w', size=28)
"""
def plot_learning_curve(estimator, title, X, y, ylim=[0.2,1.1], cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    #plt.xticks(color='w')
    #plt.yticks(color='w')
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid(color = 'k')
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.3,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.3, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    
    #ax.patch.set_color('k')
    ax.patch.set_linewidth=2.5
    axis = plt.gca().xaxis


    for line in axis.get_ticklines():
        #line.set_color('w')
        line.set_markersize(10)
        line.set_markeredgewidth(1.5)

    #ax.set_edgecolor='w'
    #ax.patch.set_edgecolor='w'
    #fig.show()
    plt.legend(loc="best")
    #plt.savefig(title,facecolor='black')
    plt.savefig(title)
    
data_con_path = os.getcwd()+'//data//ana9-14//control//'
data_pat_path = os.getcwd()+'//data//ana9-14//patient//'
data_path = os.getcwd()+'/zhibiao/'
label_path = os.getcwd()+'/data/'
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

n_components = n_row*n_col

###数据大小设置
data_row = 50
data_col = 100
data_shape_pca = 5000
data_shape = 128*128*44
shape = (128,128,44)
data_num = 16
pat_num = 21
all_num = data_num+pat_num
#图片大小设置
image_shape = (data_row, data_col)
rng = RandomState(0)
#指标设置
classname = ['1_FA','1_MD','1_csa_gfa','1_MSD','1_den']
classname = ['_FA','_MD','_GFA','_MSD','_rtop_signal','_rtop_pdf']
classnum = 6

X = np.zeros([classnum,all_num,data_shape])
X_pca = np.zeros([classnum,all_num,data_shape_pca])
#得到控制组的X
for c,cla in enumerate(classname):
    print cla
    for idx,name in enumerate(controllist):
        print idx,name
        f_nii = nib.load(data_path+name+cla+'.nii.gz')
        d = f_nii.get_data()
        f_pca = nib.load(label_path+'/big_gong_block/'+'/control/'+name+'.nii-label.nii.gz')
        f_pca = f_pca.get_data() 
        f_pca = d[f_pca==1]
        f_pca = f_pca.reshape(-1)
        #f = np.array(f)    
        d = d.reshape(-1)
        print d.shape
        #d_std = StandardScaler().fit_transform(d)
        X[c,idx,:d.shape[0]] = d/d.max()
        #f_pca_std = StandardScaler().fit_transform(f_pca)
        X_pca[c,idx,:f_pca.shape[0]] = f_pca/f_pca.max()    
        
    for name in patientlist:
        idx+=1
        print idx,name
        #f = np.load(data_pat_path+name+cla+'.npy')
        f = nib.load(data_path+name+cla+'.nii.gz')
        d = f.get_data()
        f_pca = nib.load(label_path+'/big_gong_block/'+'/patient/'+name+'.nii-label.nii.gz')
        f_pca = f_pca.get_data() 
        f_pca = d[f_pca==1]
        f_pca = f_pca.reshape(-1)
        #f = np.array(f)
        d = d.reshape(-1)
        #d_std = StandardScaler().fit_transform(d)
        print d.shape
        X[c,idx,:d.shape[0]] = d/d.max()
        #f_pca_std = StandardScaler().fit_transform(f_pca)
        X_pca[c,idx,:f_pca.shape[0]] = f_pca/f_pca.max()

### fit patient data ####
#指标pca
print 'pca starting'
#存储pca模型
zhibiao_pca = decomposition.IncrementalPCA(n_components=3)
zhibiao_pca.fit(X_pca.reshape(classnum,-1).T)
#joblib.dump(zhibiao, os.getcwd()+'/modelsave/zhibiao_pca_5000.pkl')
#zhibiao_pca = joblib.load(os.getcwd()+'/modelsave/zhibiao_pca_5000.pkl')
X_outpca = zhibiao_pca.transform(X.reshape(classnum,-1).T)  #中脑或者全脑

"""另外分量"""
signal = X_outpca[:,2].reshape([all_num,128,128,44])
signal = signal.mean(axis=0)
ind = plt.imshow(signal[:,:,28].T)
plt.xticks(())
plt.yticks(())
plt.title('')
plt.colorbar(ind)

###############################################################################
#得出X，Y的数据

X_learning = np.zeros([all_num,data_shape*2])           #(n_samples, n_features) 
for i in range(all_num):
    X_learning[i,:data_shape]=X_outpca[i*data_shape:(i+1)*data_shape,0]
    X_learning[i,data_shape:]=X_outpca[i*data_shape:(i+1)*data_shape,1]

Y_learning = np.zeros(all_num,int)
Y_learning[data_num:] =  np.ones(pat_num,int)
Y_learning[Y_learning==1].shape

##############################################################################
#feature selection
"""
from sklearn.feature_selection import f_classif
from utils import datasets,masking
from dipy.segment.mask import median_otsu
maskdata, mask = median_otsu(data, 2, 1, False, vol_idx=range(10, 50), dilate=2)

X_img = Nifti1Image(X_learning, affine)
X = masking.apply_mask(X_img, mask, smoothing_fwhm=4)
X = signal.clean(X, standardize=True, detrend=False)

dataset_files = datasets.fetch_haxby_simple()
h = datasets.fetch_haxby()
mean_img = X_learning.reshape(128,128,44,-1).mean(axis=-1)

f_values, p_values = f_classif(X_learning, Y_learning)
p_values = -np.log10(p_values)
p_values[np.isnan(p_values)] = 0
p_values[p_values > 10] = 10
mask = '/home/gongyilong/dipy_data/control/HUANGYAXIN.nii-label.nii.gz'
p_unmasked = masking.unmask(p_values, mask)


plot_haxby(p_unmasked, 'F-score')
plt.savefig('haxby/haxby_fscore.pdf')
plt.savefig('haxby/haxby_fscore.eps')
"""

print 'feature selection'
feature_selec = feature_selection.SelectKBest(feature_selection.f_classif,k=5000)
X_reduced = feature_selec.fit_transform(X_learning,Y_learning)
where = feature_selec.get_support()
awhere = where.reshape(2,128,128,44)
awhere.astype('float')
#dipy_home = pjoin(os.path.expanduser('~'), 'dipy_data')
#folder = pjoin(dipy_home, 'control')
#fraw = pjoin(folder, filename+'.nii.gz')

#存储数据
#np.save('/home/gongyilong/brain-pca/mat_save/whole_brain_Xlearning',X_learning)
#np.save('/home/gongyilong/brain-pca/mat_save/whole_brain_Xreduced',X_reduced)

#joblib.dump(bayes_estimator, os.getcwd()+'/modelsave/whole_bayes_estimator.pkl')
#bayes_estimator = joblib.load(os.getcwd()+'/modelsave/whole_bayes_estimator.pkl')
#########################reduced
print 'random_forest'
random_forest = RandomForestClassifier(n_estimators=1000)
random_forest = random_forest.fit(X_reduced,Y_learning)
ranfortitle = 'Random Forest(1000) model CV(10) in fitting whole_brain PCA feature vector after feature selection(5000)'
plot_learning_curve(random_forest,ranfortitle,X_reduced,Y_learning, cv=10)

print 'bayes_estimator'
bayes_estimator = GaussianNB()
bayes_estimator.fit(X_reduced,Y_learning)
bayestitle = 'Naive Bayes model CV(10) in fitting whole_brain PCA feature vector after feature selection(5000)'
plot_learning_curve(bayes_estimator, bayestitle,X_reduced,Y_learning, cv=10)
#plt.savefig('figure_pdf/whole_bayes_reduced.pdf')
#plt.savefig('figure_pdf/whole_bayes_reduced.eps')

##########################learning
"""
print 'random_forest'
random_forest = RandomForestClassifier(n_estimators=1000)
random_forest = random_forest.fit(X_learning,Y_learning)
ranfortitle = 'Random Forest(1000) model CV(10) in fitting whole_brain PCA feature vector not selected'
plot_learning_curve(random_forest,ranfortitle,X_learning,Y_learning, cv=10)

print 'bayes_estimator'
bayes_estimator = GaussianNB()
bayes_estimator.fit(X_learning,Y_learning)
bayestitle = 'Naive Bayes model CV(10) in fitting whole_brain PCA feature vector not selected'
plot_learning_curve(bayes_estimator, bayestitle,X_learning,Y_learning, cv=10)
"""
"""
coef = bayes_estimator.theta_
# reverse feature selection
coef = feature_selec.inverse_transform(coef)
# reverse masking
"""
coef1 = np.zeros((awhere.shape[0],awhere.shape[1], awhere.shape[2], awhere.shape[3]),
            dtype=X.dtype, order="C")
coef1[awhere] = bayes_estimator.theta_[0]

# We use a masked array so that the voxels at '-1' are displayed
# transparently
act = np.ma.masked_array(coef1, coef1 == 0)

beijing = X_learning.reshape(37,2,128,128,44)
#工形区域!!23!!

#看看lutcolormap
for z in range(44):
    beijing_one = beijing[:,1,:,:,z].mean(axis=0)
    plt.figure()
    plt.imshow(beijing_one.T ,cmap=plt.cm.gray,
                     interpolation='nearest', origin='lower',vmin=-1,vmax=1)  #图片叠加，是要看她的值区间的，根据值区间分配颜色！
    
    ind = plt.imshow(act[0,:,:,z].T,cmap=plt.cm.hot,
                  interpolation='nearest', origin='lower') 
    
    labelname = '/home/gongyilong/brain-pca/data/big_gong_block/control/HUANGYAXIN.nii-label.nii.gz'
    mask_data = nib.load(labelname).get_data()
    plt.contour(mask_data[:, :, z].astype(np.bool).T, contours=1,
                antialiased=False, linewidths=4., levels=[0],
                interpolation='nearest', colors=['blue'], origin='image')
    plt.xticks(())
    plt.yticks(())
    if z == 22:
        bar = plt.colorbar(ind)
    #plt.savefig('figure_pdf/all_z/zhibiao1_label+feature_'+str(z)+'.pdf')
    #plt.savefig('figure_pdf/all_z/zhibiao1_label+feature_'+str(z)+'.eps')

#中脑
"""预测最后准确率"""   
pred = bayes_estimator.predict(X_reduced)
print ('Bayes Training Accuracy: %f\n', np.mean((pred == Y_learning)) * 100);
pred = random_forest.predict(X_reduced)
print ('Bayes Training Accuracy: %f\n', np.mean((pred == Y_learning)) * 100);
18
"""
beijing_zhong = beijing[:,0,:,:,20].mean(axis=0)
bei = plt.matshow(beijing_zhong+awhere[0,:,:,20])  #图片叠加，是要看她的值区间的，根据值区间分配颜色！
labelname = '/home/gongyilong/dipy_data/control/HUANGYAXIN.nii-label.nii.gz'
mask_data = nib.load(labelname).get_data()
plt.contour(mask_data[:, 4:58, 20].astype(np.bool).T, contours=1,
            antialiased=False, linewidths=4., levels=[0],
            interpolation='nearest', colors=['blue'], origin='lower')
"""

"""
bayes_estimator = GaussianNB()
bayes_estimator.fit(X_learning,Y_learning)

scores1=cross_validation.cross_val_score(bayes_estimator,X_learning,Y_learning, cv=10)
#scores = cross_validation.cross_val_score(clf, X_scaled_norm, y_le, cv=10)
print("Bayes: Accuracy: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))
plot_learning_curve(bayes_estimator, 'Learning Curves (Naive Bayes)', X_learning,Y_learning, cv=cv)
"""
# Load faces data
#dataset = fetch_olivetti_faces(shuffle=True, random_state=rng)
"""
faces = X

n_samples, n_features = faces.shape

# global centering
faces_centered = faces - faces.mean(axis=0)

# local centering
faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

print("Dataset consists of %d faces" % n_samples)
"""

###############################################################################
    
#plot_gallery("original control data", X[:data_num],n_col=4, n_row=4)


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

