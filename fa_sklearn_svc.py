
# -*- coding: utf-8 -*-  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import pylab as pl
from sklearn.learning_curve import learning_curve
import sys
from sklearn import datasets, svm, metrics
from sklearn import cross_validation

def svc_judje(testname,testpath):
    
    data_con_path = 'E:\\brain\\fa&gfa experi\\control_fa\\'
    data_pat_path = 'E:\\brain\\fa&gfa experi\\patient_fa\\'
    
    controllist = ['HUANGYAXIN','LIUZHIBING','SUNAIQUAN','Tanjinxin',
                   'WANGCHUNXIANG','WANGKUNYING','WANGWEI','xiaowen',
                   'XIAQIAORONG','XIEGUOLIANG',
                   'zhangruiwei','zhaoaiju','ZHOUGUIQUAN','zhouyan']
    
    patientlist = ['CHENLIANXIANG','jiangguozhi',
                   'liangxiangxiu','liangxiangxiu2','limingcai','liuwuchou',
                   'LIUYAN','liuyingxiao','wangdongfang',
                   'wangjinguo','WangPingAn','wangwuji',
                   'Wushaoxian','XIONGYIN','yangjianjun','zhouyongzhong','ZHUCHUANGUI']
    filename = 'HUANGYAXIN'
    targets = np.ndarray([31],bool)
    X = np.ndarray([31,19,20])
    
    for idx,name in enumerate(controllist):
        f = pd.read_excel(data_con_path+name+'.xls')
        f = np.array(f)
        X[idx] = f
        targets[idx] = False
        
    for name in patientlist:
        f = pd.read_excel(data_pat_path+name+'.xls')
        f = np.array(f)
        X[idx] = f
        targets[idx] = True
    
    n_samples = len(X)
    
    data = X.reshape((n_samples, -1))
    
    # split into a training and testing set
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        data, targets, test_size=0.2)
    
    # Create a classifier: a support vector classifier
    classifier = svm.SVC(gamma=0.001)
    # We learn the digits on the first half of the digits
    classifier.fit(X_train,y_train)
    
    # Now predict the value of the digit on the second half:
    expected = y_test
    predicted = classifier.predict(X_test)
    
    
    print 'SVC模型，预测判别准确度报告：(Flase为患病，True为无病)\n',
    print   metrics.classification_report(expected, predicted)
    #print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
    
    f = pd.read_excel(testpath+testname+'.xls')
    print '被测：',
    print testname
    f = np.array(f)
    test = np.ndarray([1,19,20])
    test = test.reshape((1, -1))
    
    if(classifier.predict(test)):
        print '对定位区域预测其患病'
    
    else:
        print '对定位区域预测其没有患病'

origin = sys.stdout 
outfile = open('fa_svc_sklearn_out.txt', 'w') 
sys.stdout = outfile

       
testpath = 'D:\\dipy\\stand\\HARDI150\\'
testname = 'HARDI150_fa'       
svc_judje(testname,testpath)

