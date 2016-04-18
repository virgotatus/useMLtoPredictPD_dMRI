#-*- coding: utf-8 -*-

from nipy import save_image
import numpy as np
from nipy.core.api import Image, vox2mni

rawarray = np.ones((43,128,128), dtype=np.uint8)
arr_img = Image(rawarray, vox2mni(np.eye(4))) ##affine

print arr_img.shape
newimg = save_image(arr_img, 'G:\\Userworkspace\\brain-pca\\niwe.nii.gz')

#用affine，将image转 成3d的东西  nipy.pdf在brain里面
#print save_image.func_doc
#cmap = AffineTransform('kji', 'zxy', np.eye(4))
#img = Image(data, cmap)
