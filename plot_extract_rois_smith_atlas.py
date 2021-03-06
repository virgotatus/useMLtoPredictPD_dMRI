"""
Regions Extraction of Default Mode Networks using Smith Atlas
=============================================================

This simple example shows how to extract regions from Smith atlas
resting state networks.

In particular, we show how Default Mode Network regions are extracted
using :class:`nilearn.regions.RegionExtractor` from regions module
"""

################################################################################
# Fetching the smith ICA 10 RSN by importing datasets utilities
from nilearn import datasets

smith_atlas = datasets.fetch_atlas_smith_2009()
atlas_networks = smith_atlas.rsn10
from os.path import join as pjoin
import os
dipy_home = pjoin(os.path.expanduser('~'), 'dipy_data')
folder = pjoin(dipy_home, 'control')
fraw = pjoin(folder, 'HUANGYAXIN.nii.gz')
atlas_networks = fraw
################################################################################
# Import region extractor to extract atlas networks
from nilearn.regions import RegionExtractor

# min_region_size in voxel volume mm^3
extraction = RegionExtractor(atlas_networks, min_region_size=800,
                             threshold=10, thresholding_strategy='percentile')

# Just call fit() to execute region extraction procedure
extraction.fit()
regions_img = extraction.regions_img_

################################################################################
# Visualization
# Show region extraction results by importing image & plotting utilities
from nilearn import plotting
from nilearn.image import index_img
from nilearn.plotting import find_xyz_cut_coords

# Showing region extraction results using 4D maps visualization tool
plotting.plot_prob_atlas(regions_img, display_mode='z', cut_coords=2,
                         view_type='contours', title="Regions extracted.")

# To reduce the complexity, we choose to display all the regions
# extracted from network 3
import numpy as np

DMN_network = index_img(atlas_networks, 3)
plotting.plot_roi(DMN_network, display_mode='z', cut_coords=1,
                  title='Network 3')

regions_indices_network3 = np.where(np.array(extraction.index_) == 3)
for index in regions_indices_network3[0]:
    cur_img = index_img(extraction.regions_img_, index)
    coords = find_xyz_cut_coords(cur_img)
    plotting.plot_roi(cur_img, display_mode='z', cut_coords=coords[2:3],
                      title="Blob of network3")
            
plotting.show()
