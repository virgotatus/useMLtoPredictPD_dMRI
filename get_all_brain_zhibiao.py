#-*- encoding: utf-8 -*-
'''
Created on 201５年１月２４日

@author: virgotatus
'''

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from os.path import join as pjoin
import os

from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.dsi import DiffusionSpectrumModel
from dipy.data import read_stanford_hardi, get_sphere

from dipy.reconst.shm import CsaOdfModel, normalize_data
from dipy.reconst.dti import fractional_anisotropy
from dipy.reconst.csdeconv import recursive_response
from dipy.reconst.peaks import peaks_from_model
from dipy.segment.mask import median_otsu

import through_label_sl

def get_csd_gfa(nii_data,gtab):    
    from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
    csd_model = ConstrainedSphericalDeconvModel(gtab, None, sh_order=6)

    GFA=csd_model.fit(data,mask).gfa    
    
    print ('csd_gfa ok')

def dodata(f_name,data_path):
    dipy_home = pjoin(os.path.expanduser('~'), 'dipy_data')
    folder = pjoin(dipy_home, data_path)
    fraw = pjoin(folder, f_name+'.nii.gz')
    fbval = pjoin(folder, f_name+'.bval')
    fbvec = pjoin(folder, f_name+'.bvec')
    flabels = pjoin(folder, f_name+'.nii-label.nii.gz')
    
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs)
    
    img = nib.load(fraw)
    data = img.get_data()
    affine = img.get_affine()
    
    label_img = nib.load(flabels)
    labels=label_img.get_data()
    lap=through_label_sl.label_position(labels, labelValue=1)    
    dataslice = data[40:80, 20:80, lap[2][2] / 2]
    #print lap[2][2]/2
    
    #get_csd_gfa(f_name,data,gtab,dataslice)
    
    maskdata, mask = median_otsu(data, 2, 1, False, vol_idx=range(10, 50), dilate=2) #不去背景
    
    """ get fa and tensor evecs and ODF"""
    from dipy.reconst.dti import TensorModel,mean_diffusivity
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask)
    
    sphere = get_sphere('symmetric724')
    
    FA = fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
      
    np.save(os.getcwd()+'\zhibiao'+f_name+'_FA.npy',FA)
    fa_img = nib.Nifti1Image(FA.astype(np.float32), affine)
    nib.save(fa_img,os.getcwd()+'\zhibiao'+f_name+'_FA.nii.gz')
    print('Saving "DTI_tensor_fa.nii.gz" sucessful.')
    evecs_img = nib.Nifti1Image(tenfit.evecs.astype(np.float32), affine)
    nib.save(evecs_img, os.getcwd()+'\zhibiao'+f_name+'_DTI_tensor_evecs.nii.gz')
    print('Saving "DTI_tensor_evecs.nii.gz" sucessful.')
    MD1 = mean_diffusivity(tenfit.evals)
    nib.save(nib.Nifti1Image(MD1.astype(np.float32), img.get_affine()), os.getcwd()+'\zhibiao'+f_name+'_MD.nii.gz')
    
    
    #tensor_odfs = tenmodel.fit(data[20:50, 55:85, 38:39]).odf(sphere)
    #from dipy.reconst.odf import gfa
    #dti_gfa=gfa(tensor_odfs)
    
    wm_mask = (np.logical_or(FA >= 0.4, (np.logical_and(FA >= 0.15, MD >= 0.0011))))

    response = recursive_response(gtab, data, mask=wm_mask, sh_order=8,
                                  peak_thr=0.01, init_fa=0.08,
                                  init_trace=0.0021, iter=8, convergence=0.001,
                                  parallel=False)
    from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
    csd_model = ConstrainedSphericalDeconvModel(gtab, response)
    
    #csd_fit = csd_model.fit(data)

    from dipy.direction import peaks_from_model

    csd_peaks = peaks_from_model(model=csd_model,
                                 data=data,
                                 sphere=sphere,
                                 relative_peak_threshold=.5,
                                 min_separation_angle=25,
                                 parallel=False)
    
    GFA = csd_peaks.gfa
    
    nib.save(GFA, os.getcwd()+'\zhibiao'+f_name+'_MSD.nii.gz')
    print('Saving "GFA.nii.gz" sucessful.')
    
    from dipy.reconst.shore import ShoreModel
    asm = ShoreModel(gtab)
    print('Calculating...SHORE msd')
    asmfit = asm.fit(data,mask)
    msd = asmfit.msd()
    msd[np.isnan(msd)] = 0
    
    #print GFA[:,:,slice].T
    print('Saving msd_img.png')
    nib.save(msd, os.getcwd()+'\zhibiao'+f_name+'_GFA.nii.gz')


patient3_name_list=['CHENLIANXIANG','duxiaojing','jiangchangguo','jiangguozhi','liangxiangxiu','liangxiangxiu2','limingcai','liuwuchou','LIUYAN'
             ,'liuyingxiao','renqixia','wangdongfang','Wangfusheng','wangjinguo','WangPingAn','wangsongyuan','wangwuji','Wushaoxian','XIONGYIN',
             'yangjianjun','zhouyongzhong','ZHUCHUANGUI']

control3_name_list = ['HUANGYAXIN','LIUZHIBING','SUNAIQUAN','Tanjinxin','WANGCHUNXIANG','WANGKUNYING','WANGWEI',
                     'xiaowen','XIAQIAORONG','XIEGUOLIANG','YAOYOUYUAN','YUANGUISHENG','zhangruiwei','zhaoaiju','ZHOUGUIQUAN','zhouyan']
data_path = 'patient'
f_name = 'duxiaojing'

#封装  还有控制病人要变，只for了一个
for f_name in control3_name_list:
    dipy_home = pjoin(os.path.expanduser('~'), 'dipy_data')
    folder = pjoin(dipy_home, 'control')
    fraw = pjoin(folder, f_name+'.nii.gz')
    fbval = pjoin(folder, f_name+'.bval')
    fbvec = pjoin(folder, f_name+'.bvec')
    flabels = pjoin(folder, f_name+'.nii-label.nii.gz')
    
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs)
    
    img = nib.load(fraw)
    data = img.get_data()
    affine = img.get_affine()
    
    label_img = nib.load(flabels)
    labels=label_img.get_data()
    lap=through_label_sl.label_position(labels, labelValue=1)    
    dataslice = data[40:80, 20:80, lap[2][2] / 2]
    #print lap[2][2]/2
    
    #get_csd_gfa(f_name,data,gtab,dataslice)
    
    maskdata, mask = median_otsu(data, 2, 1, False, vol_idx=range(10, 50), dilate=2) #不去背景
    
    """ get fa and tensor evecs and ODF"""
    from dipy.reconst.dti import TensorModel,mean_diffusivity
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask)
    
    sphere = get_sphere('symmetric724')
    
    FA = fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
      
    np.save(os.getcwd()+'\zhibiao'+f_name+'_FA.npy',FA)
    fa_img = nib.Nifti1Image(FA.astype(np.float32), affine)
    print FA.shape
    nib.save(fa_img,os.getcwd()+'/zhibiao/'+f_name+'_FA.nii.gz')
    print('Saving "DTI_tensor_fa.nii.gz" sucessful.')
    evecs_img = nib.Nifti1Image(tenfit.evecs.astype(np.float32), affine)
    nib.save(evecs_img, os.getcwd()+'/zhibiao/'+f_name+'_DTI_tensor_evecs.nii.gz')
    print('Saving "DTI_tensor_evecs.nii.gz" sucessful.')
    MD = mean_diffusivity(tenfit.evals)
    print MD.shape
    print('Saving "MD.nii.gz" sucessful.')
    nib.save(nib.Nifti1Image(MD.astype(np.float32), img.get_affine()), os.getcwd()+'/zhibiao/'+f_name+'_MD.nii.gz')
    
    
    tensor_odfs = tenmodel.fit(data[20:50, 55:85, 38:39]).odf(sphere)
    from dipy.reconst.odf import gfa
    dti_gfa=gfa(tensor_odfs)
    """
    wm_mask = (np.logical_or(FA >= 0.4, (np.logical_and(FA >= 0.15, MD >= 0.0011))))
    
    response = recursive_response(gtab, data, mask=wm_mask, sh_order=8,
                                  peak_thr=0.01, init_fa=0.08,
                                  init_trace=0.0021, iter=8, convergence=0.001,
                                  parallel=False)
    from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
    csd_model = ConstrainedSphericalDeconvModel(gtab, response)
    
    csd_fit = csd_model.fit(data)
    csd_odf = csd_fit.odf(sphere)
    
    
    csd_peaks = peaks_from_model(model=csd_model,
                                 data=data,
                                 sphere=sphere,
                                 relative_peak_threshold=.5,
                                 min_separation_angle=25,
                                 parallel=False)
    
    GFA = csd_peaks.gfa
    """
    sphere = get_sphere('symmetric724')
    csamodel = CsaOdfModel(gtab, 4)
    
    
    #csafit = csamodel.fit(data_small)
    
    csapeaks = peaks_from_model(model=csamodel,
                            data=maskdata,
                            sphere=sphere,
                            relative_peak_threshold=.5,
                            min_separation_angle=25,
                            mask=mask,
                            return_odf=False,
                            normalize_peaks=True)
    
    GFA = csapeaks.gfa
    GFA_img = nib.Nifti1Image(GFA.astype(np.float32), affine)
    print GFA.shape
    nib.save(GFA_img, os.getcwd()+'/zhibiao/'+f_name+'_GFA.nii.gz')
    print('Saving "GFA.nii.gz" sucessful.')
    
    from dipy.reconst.shore import ShoreModel
    asm = ShoreModel(gtab)
    print('Calculating...SHORE msd')
    asmfit = asm.fit(data,mask)
    msd = asmfit.msd()
    msd[np.isnan(msd)] = 0
    
    #print GFA[:,:,slice].T
    print('Saving msd_img.png')
    print msd.shape
    msd_img = nib.Nifti1Image(msd.astype(np.float32), affine)
    nib.save(msd_img, os.getcwd()+'/zhibiao/'+f_name+'_MSD.nii.gz')
    
    #密度？？
