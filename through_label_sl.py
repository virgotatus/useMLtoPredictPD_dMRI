#-*- encoding: utf-8 -*-
'''
Created on 2014��7��17��

@author: Yilong Gong 
'''
import os
import nibabel as nib
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from os.path import join as pjoin


from dipy.reconst.dti import fractional_anisotropy
from dipy.segment.mask import median_otsu
from dipy.tracking import utils
from dipy.viz import fvtk
from dipy.viz.colormap import line_colors

patient_name_list=['cen wang lan','chang jia guo','chen cui hua','chen ming zeng','chen nian xiang (D)','chen wan lan(D)','gong guo ming','gui yun shan','he ming you'
             ,'huang yi','jin he ping','ke dong xue','li tai qing','liu de xiang','liu ke ying','luo li ping','luo zhuo hua','peng hai cui','peng yu fang',
             'shun nian chen','song ming yong','su xia han','wang pin an (D)','wang yong xi','wang yu sheng',
             'wu hui wu','xiao guang yuan','xie guang hua','xue yang qing','yang li ji','yao guo li','ying guo hua','yu rong pu','yue feng ming',
             'zen xiu ying','zhai xian yao','zhang chuang sheng','zhang han ming','zhou bin yan','zhu chuan gui(D)','zhu chuang gui','zhu jing zhang',
             'zhu long ping']

control_name_list = ['guochunxiuNC','lihuilanNC','liu you jiao NC','liu zhi bing','liurenhuaNC','liuxiunanNC','lu gui zeng','lurenguoNC',
                     'wuchangyinNC','xuyuhuaNC','zeng hong NC','zhan xin tian','zhang rui wei(D)']
data_path = 'patient_data'


def label_position(label,labelValue):
    labelposition=[]
    x=0;y=0;z=0;
    for i in label:
        y=0
        for j in i:
            z=0
            for k in j:
                if(k==labelValue):
                    labelposition.append((x,y,z))
                z=z+1   
            y=y+1
        x=x+1
    return labelposition              
                    
    
def label_streamlines(streamlines,labels,labels_Value,affine,hdr,f_name,data_path):  
      
    cc_slice=labels==labels_Value
    cc_streamlines = utils.target(streamlines, labels, affine=affine)
    cc_streamlines = list(cc_streamlines)

    other_streamlines = utils.target(streamlines, cc_slice, affine=affine,
                                 include=False)
    other_streamlines = list(other_streamlines)
    assert len(other_streamlines) + len(cc_streamlines) == len(streamlines)
    

    print ("num of roi steamlines is %d",len(cc_streamlines))
    

    # Make display objects
    color = line_colors(cc_streamlines)
    cc_streamlines_actor = fvtk.line(cc_streamlines, line_colors(cc_streamlines))
    cc_ROI_actor = fvtk.contour(cc_slice, levels=[1], colors=[(1., 1., 0.)],
                            opacities=[1.])

    # Add display objects to canvas
    r = fvtk.ren()
    fvtk.add(r, cc_streamlines_actor)
    fvtk.add(r, cc_ROI_actor)

    # Save figures
    fvtk.record(r, n_frames=1, out_path=f_name+'_roi.png',
            size=(800, 800))
    fvtk.camera(r, [-1, 0, 0], [0, 0, 0], viewup=[0, 0, 1])
    fvtk.record(r, n_frames=1, out_path=f_name+'_roi.png',
            size=(800, 800))
    """"""

    csd_streamlines_trk = ((sl, None, None) for sl in cc_streamlines)
    csd_sl_fname = f_name+'_roi_streamline.trk'
    nib.trackvis.write(csd_sl_fname, csd_streamlines_trk, hdr, points_space='voxel')
    #nib.save(nib.Nifti1Image(FA, img.get_affine()), 'FA_map2.nii.gz')
    print('Saving "_roi_streamline.trk" sucessful.')

    import tractconverter as tc
    input_format=tc.detect_format(csd_sl_fname)
    input=input_format(csd_sl_fname)
    output=tc.FORMATS['vtk'].create(csd_sl_fname+".vtk",input.hdr)
    tc.convert(input,output)
    
    return cc_streamlines

def connective_label(streamlines,labels,affine,hdr,f_name,data_path):
    """
    Once we've targeted on the corpus callosum ROI, we might want to find out which
    regions of the brain are connected by these streamlines. To do this we can use
    the ``connectivity_matrix`` function. This function takes a set of streamlines
    and an array of labels as arguments. It returns the number of streamlines that
    start and end at each pair of labels and it can return the streamlines grouped
    by their endpoints. Notice that this function only considers the endpoints of
    each streamline.
    """
    
    M, grouping = utils.connectivity_matrix(streamlines, labels, affine=affine,
                                            return_mapping=True,
                                            mapping_as_streamlines=True)
#    M[:3, :] = 0
#    M[:, :3] = 0
    
    print M

    
    """
    We've set ``return_mapping`` and ``mapping_as_streamlines`` to ``True`` so that
    ``connectivity_matrix`` returns all the streamlines in ``cc_streamlines``
    grouped by their endpoint.
    
    Because we're typically only interested in connections between gray matter
    regions, and because the label 0 represents background and the labels 1 and 2
    represent white matter, we discard the first three rows and columns of the
    connectivity matrix.
    
    We can now display this matrix using matplotlib, we display it using a log
    scale to make small values in the matrix easier to see.
    """
    
    import matplotlib.pyplot as plt
    plt.imshow(np.log1p(M), interpolation='nearest')
    plt.savefig("connectivity.png")
    return M,grouping

def label_streamlines_density(streamlines,labels,affine,f_name,img,label_img):

    """
    .. figure:: connectivity.png
       :align: center
    
       **Connectivity of Corpus Callosum**
    
    .. include:: ../links_names.inc
    
    """

    
    shape = labels.shape
    dm = utils.density_map(streamlines, shape, affine=affine)
    sum=0 ;count=0
    for i in dm:
        for j in i:
            for k in j:
                if (k != 0):
                    sum=sum+k
                    count += 1
    density = sum*1.0/count
    print density
    
    """
    
    To do that, we will use tools available in [nibabel](http://nipy.org/nibabel)
    """
    

    
    # Save density map
    dm_img = nib.Nifti1Image(dm.astype("int16"), img.get_affine())
    dm_img.to_filename(f_name+"-dm.nii.gz")
    
    # Make a trackvis header so we can save streamlines
    voxel_size = label_img.get_header().get_zooms()
    trackvis_header = nib.trackvis.empty_header()
    trackvis_header['voxel_size'] = voxel_size
    trackvis_header['dim'] = shape
    trackvis_header['voxel_order'] = "RAS"
    
    # Move streamlines to "trackvis space"
    trackvis_point_space = utils.affine_for_trackvis(voxel_size)
    lr_sf_trk = utils.move_streamlines(streamlines,
                                       trackvis_point_space, input_space=affine)
    lr_sf_trk = list(lr_sf_trk)
    
    """
    # Save streamlines
    for_save = [(sl, None, None) for sl in lr_sf_trk]
    
    nib.trackvis.write(f_name+"_label1.trk", for_save, trackvis_header)
    """
    """
    import tractconverter as tc
    density_file = f_name+"_label1.trk"
    input_format=tc.detect_format(density_file)
    input=input_format(density_file)
    output=tc.FORMATS['vtk'].create(density_file+".vtk",input.hdr)
    tc.convert(input,output)
    """
    """
    Let's take a moment here to consider the representation of streamlines used in
    dipy. Streamlines are a path though the 3d space of an image represented by a
    set of points. For these points to have a meaningful interpretation, these
    points must be given in a known coordinate system. The ``affine`` attribute of
    the ``streamline_generator`` object specifies the coordinate system of the
    points with respect to the voxel indices of the input data.
    ``trackvis_point_space`` specifies the trackvis coordinate system with respect
    to the same indices. The ``move_streamlines`` function returns a new set of
    streamlines from an existing set of streamlines in the target space. The
    target space and the input space must be specified as affine transformations
    with respect to the same reference [#]_. If no input space is given, the input
    space will be the same as the current representation of the streamlines, in
    other words the input space is assumed to be ``np.eye(4)``, the 4-by-4 identity
    matrix.
    
    All of the functions above that allow streamlines to interact with volumes take
    an affine argument. This argument allows these functions to work with
    streamlines regardless of their coordinate system. For example even though we
    moved our streamlines to "trackvis space", we can still compute the density map
    as long as we specify the right coordinate system.
    """
    
    dm_trackvis = utils.density_map(lr_sf_trk, shape, affine=trackvis_point_space)
    assert np.all(dm == dm_trackvis)
    
    
    return dm,density
    """
    This means that streamlines can interact with any image volume, for example a
    high resolution structural image, as long as one can register that image to
    the diffusion images and calculate the coordinate system with respect to that
    image.
    """
    """
    .. rubric:: Footnotes
    
    .. [#] The image `aparc-reduced.nii.gz`, which we load as ``labels_img``, is a
        modified version of label map `aparc+aseg.mgz` created by freesurfer.  The
        corpus callosum region is a combination of the freesurfer labels 251-255.
        The remaining freesurfer labels were re-mapped and reduced so that they lie
        between 0 and 88. To see the freesurfer region, label and name, represented
        by each value see `label_info.txt` in `~/.dipy/stanford_hardi`.
    .. [#] An affine transformation is a mapping between two coordinate systems
        that can represent scaling, rotation, sheer, translation and reflection.
        Affine transformations are often represented using a 4x4 matrix where the
        last row of the matrix is ``[0, 0, 0, 1]``.
    """
    

def experiment1(f_name,data_path):
    "OUTPUT"
    f=open(f_name+'_out.txt','w')
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """Read Data"""
    dipy_home = pjoin(os.path.expanduser('~'), '.dipy')

    folder = pjoin(dipy_home, data_path)
    fraw = pjoin(folder, f_name+'.nii.gz')
    fbval = pjoin(folder, f_name+'.bval')
    fbvec = pjoin(folder, f_name+'.bvec')
    flabels = pjoin(folder, f_name+'.nii-label.nii.gz')

    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs)
    img = nib.load(fraw)
    data = img.get_data()
    print('data.shape (%d, %d, %d, %d)' % data.shape)
    print('Building DTI Model Data......')

    """Load label"""
    label_img = nib.load(flabels)
    labels=label_img.get_data()
    
    labelpo1=label_position(labels,1)
    print labelpo1
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    maskdata, mask = median_otsu(data, 3, 1, False, vol_idx=range(10, 50), dilate=2)

    from dipy.reconst.dti import TensorModel
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask)

    FA = fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
    np.save(f_name+'_FA',FA)
    fa_img = nib.Nifti1Image(FA.astype(np.float32), img.get_affine())
    #nib.save(fa_img,f_name+'_DTI_tensor_fa.nii.gz')
    #print('Saving "DTI_tensor_fa.nii.gz" sucessful.')
    evecs_img = nib.Nifti1Image(tenfit.evecs.astype(np.float32), img.get_affine())
    #nib.save(evecs_img, f_name+'_DTI_tensor_evecs.nii.gz')
    #print('Saving "DTI_tensor_evecs.nii.gz" sucessful.')
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    ""Fiber Tracking"""
    print('Fiber Tracking......')
    from dipy.tracking.eudx import EuDX
    from dipy.data import  get_sphere


    from dipy.tracking import utils
    seeds = utils.seeds_from_mask(labels, density=3)
    print('The number of seeds is %d.' % len(seeds))

    print >>f,('The number of seeds is %d.' % len(seeds))

    sphere = get_sphere('symmetric724')
    from dipy.reconst.dti import quantize_evecs
    evecs = evecs_img.get_data()
    peak_indices = quantize_evecs(evecs, sphere.vertices)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    streamline_generator = EuDX(FA.astype('f8'),
                            peak_indices,
                            seeds=10**5,
                            odf_vertices=sphere.vertices,
                            a_low=0.2,
                            step_sz=0.5,
                            ang_thr=60.0,
                            total_weight=.5,
                            max_points=10**5)
    streamlines_all = [streamlines_all for streamlines_all in streamline_generator]
    
    """"""""""""""""""""""""""""""
    """Select length bigger than 10"""
    from dipy.tracking.utils import length
    lengths = list(length(streamlines_all))
    select_length = 0
    length=len(streamlines_all)
    j=0
    for i in range(length):
        if ((lengths[i]) > select_length):   
            streamlines_all[j] = streamlines_all[i]
            j=j+1
    j=j-1
    streamlines = streamlines_all[0:j]
    
    
    hdr = nib.trackvis.empty_header()
    hdr['voxel_size'] = img.get_header().get_zooms()[:3]
    hdr['voxel_order'] = 'LAS'
    hdr['dim'] = FA.shape[:3]
    
    """划出roi streamlines"""
    affine = streamline_generator.affine
    cc_streamlines=label_streamlines(streamlines, labels,1, affine, hdr, f_name, data_path)
    #M,grouping = connective_label(cc_streamlines, labels, affine, hdr, f_name, data_path)
    label_streamlines_density(cc_streamlines, labels, affine,f_name,img, label_img)
    """两个label的问题"""
    flabels2 = pjoin(folder, f_name+'22.nii-label.nii.gz')
    label_img2 = nib.load(flabels2)
    labels2=label_img2.get_data()
    
    cc22_streamlines=label_streamlines(streamlines, labels2,3, affine, hdr, f_name, data_path)  
    labels3 = labels[:] 
    for i in range(len(labels)):
        for j in range(len(labels[i])):
                for k in range(len(labels[i][j])) :
                    if (labels[i][j][k]==0 and labels2[i][j][k]!=0):
                        labels3[i][j][k] = labels2[i][j][k]
    
    M,grouping = connective_label(streamlines, labels3, affine, hdr, f_name, data_path)
    print M
    print grouping[0,3]
    
    
#experiment1('zhu long ping','patient_data')