# -*- coding: utf-8 -*-  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import pylab as pl
from sklearn.learning_curve import learning_curve
import sys
from array import array

data_con_path = 'E:\\brain\\fa&gfa experi\\control_fa\\'
data_pat_path = 'E:\\brain\\fa&gfa experi\\patient_fa\\'

controllist = ['HUANGYAXIN','LIUZHIBING','SUNAIQUAN','Tanjinxin',
               'WANGCHUNXIANG','WANGKUNYING','WANGWEI','xiaowen',
               'XIAQIAORONG','XIEGUOLIANG','YAOYOUYUAN','YUANGUISHENG',
               'zhangruiwei','zhaoaiju','ZHOUGUIQUAN','zhouyan']

patientlist = ['CHENLIANXIANG','duxiaojing','jiangchangguo','jiangguozhi',
               'liangxiangxiu','liangxiangxiu2','limingcai','liuwuchou',
               'LIUYAN','liuyingxiao','renqixia','wangdongfang','Wangfusheng',
               'wangjinguo','WangPingAn','wangsongyuan','wangwuji',
               'Wushaoxian','XIONGYIN','yangjianjun','zhouyongzhong','ZHUCHUANGUI']
filename = 'HUANGYAXIN'

targets = []
def pltsave(f,fname):
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_axis_off()
    
    ind = ax1.imshow(f, interpolation='nearest', origin='lower')
    #cbar.set_yticks(color='w')
    plt.savefig(fname+'fa.png',labelcolor='w',edgecolor='k')
    plt.close()

X = []

for idx,name in enumerate(controllist):
    f = pd.read_excel(data_con_path+name+'.xls')
    f = np.array(f)
    X.append(f)
    targets.append(0)
    
for name in patientlist:
    f = pd.read_excel(data_pat_path+name+'.xls')
    f = np.array(f)
    X.append(f)
    targets.append(1)

n_samples = len(X)
data = np.array(X)
targets = np.array(targets)
data = X.reshape((n_samples, -1))


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
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
    plt.xlabel("Training examples",color='w')
    plt.ylabel("Score",color='w')
    plt.xticks(color='w')
    plt.yticks(color='w')
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid(color = 'w')
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.3,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.3, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    
    ax.patch.set_color('k')
    ax.patch.set_linewidth=2.5
    axis = plt.gca().xaxis
    for label in axis.get_ticklabels():
        label.set_color('w')

    for line in axis.get_ticklines():
        line.set_color('w')
        line.set_markersize(10)
        line.set_markeredgewidth(1.5)

    ax.set_edgecolor='w'
    ax.patch.set_edgecolor='w'
    #fig.show()
    plt.legend(loc="best")
    return plt

#脚本
"""
origin = sys.stdout 
outfile = open('fa_sklearn_out.txt', 'w') 
sys.stdout = outfile
"""

#anova,value=ss.onewayvar('ispatient','csa_gfa_ave',data)
#ss.onewayvar('ispatient','csa_gfa_var',data)
#ss.onewayvar('ispatient','csa_gfapeak_ave',data)
#ss.onewayvar('ispatient','csa_gfapeak_var',data)
#ss.onewayvar('ispatient','csd_gfa_var',data)
print '\n'

#data_traning or sample
from sklearn import cross_validation
cv = cross_validation.ShuffleSplit(38, n_iter=10,
                                   test_size=0.2, random_state=0)

def my_custom_loss_func(ground_truth, predictions):
    diff = np.abs(ground_truth - predictions).max()
    return np.log(1 + diff)

from sklearn.metrics import fbeta_score, make_scorer
ftwo_scorer = make_scorer(fbeta_score, beta=2)
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]}, scoring=ftwo_scorer)

print 'fitting Dummy model'
loss  = make_scorer(my_custom_loss_func, greater_is_better=False)
score = make_scorer(my_custom_loss_func, greater_is_better=True)
from sklearn.dummy import DummyClassifier
dummy_estimator = DummyClassifier(strategy='most_frequent', random_state=0)

dummy_estimator.fit(X[n_samples / 10:n_samples*9 / 10],targets[n_samples / 10:n_samples*9 / 10])


#plot_learning_curve(dummy_estimator, 'Learning Curves(Dummy)', X, y, cv=cv)
#plt.savefig('Learning Curves(Dummy)',facecolor='black',labelcolor='k',edgecolor='k')
print '\n'
#accurity

from sklearn.lda import LDA
from sklearn import svm
from sklearn.pipeline import make_pipeline

from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
print 'fitting Bayes model'
title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
"""
bayes_estimator = GaussianNB()

for X in fcon:
    dummy_estimator = bayes_estimator.fit(X,'control')

for X in fpat:
    dummy_estimator = bayes_estimator.fit(X,'patient')

print 'Bayes-score:', bayes_estimator.score()
"""
#plot_learning_curve(bayes_estimator, title, X, y, cv=cv)
#plt.savefig('Learning Curves (Naive Bayes)',facecolor='black')
print '\n'
"""
svc = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
for X in fcon:
    svc.fit(X,'control')

for X in fpat:
    svc.fit(X,'patient')

scores1=cross_validation.cross_val_score(svc,X, 'patient', cv=5)
#scores = cross_validation.cross_val_score(clf, X_scaled_norm, y_le, cv=10)
print("SVM: Accuracy: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))
print '\n'
"""
"""
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
estimators = [('reduce_dim', PCA())]
clfall = Pipeline(estimators)

scoresone=cross_validation.cross_val_score(clfall,X, y, cv=10)
print("PCA: Accuracy: %0.2f (+/- %0.2f)" % (scoresone.mean(), scoresone.std() * 2))
"""

contestdata = 'HUANGYAXIN'
pattestdata = 'CHENLIANXIANG'

contest = pd.read_excel(data_con_path+contestdata+'.xls')
pattest = pd.read_excel(data_pat_path+pattestdata+'.xls')
print dummy_estimator.predict(pattest)
"""

print 'Dummy-score:',dummy_estimator.score()
if(dummy_estimator.predict(contest) == 'patient'):
    print 'dummy: is patient'
else:
    print 'dummy: is healthy'
"""
"""   
if(bayes_estimator.predict(contest)=='patient'):
    print 'bayes: is patient'
else:
    print 'bayes: is healthy'
"""
"""
sys.stdout = origin 
outfile.close() 
"""
