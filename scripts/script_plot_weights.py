"""
script for getting group average weights
@ Ladan Shahshahani Jan 30 2023 12:57
"""
import os.path as op
import numpy as np
import deepdish as dd
import pandas as pd
import re
import sys
from collections import defaultdict
# sys.path.append('../cortico-cereb_connectivity')
# sys.path.append('../Functional_Fusion')
# sys.path.append('..')
import nibabel as nb
import Functional_Fusion.dataset as ds  # from functional fusion module
import ProbabilisticParcellation.util as ut
import data as cdata
import atlas_map as am
from pathlib import Path

# set base directory of the functional fusion
atlas_dir = ut.base_dir + '/Atlases'


def plot_weights(method="L2Regression",
                 cortex="Icosahedron-1002_Sym.32k",
                 cerebellum="NettekovenSym34",
                 log_alpha=8,
                 dataset_name="MDTB",
                 ses_id="ses-s1",
                 save_as=None):
    # get the file containing best weights
    filename = op.join(cdata.conn_dir, dataset_name, 'train',
                       f'{cortex}_{ses_id}_{method}_logalpha_{log_alpha}_best_weights.npy')
    weights = np.load(filename)

    # get atlases and create parcels/parcel labels
    atlas_cereb, _ = am.get_atlas('SUIT3', atlas_dir)
    atlas_cortex, _ = am.get_atlas('fs32k', atlas_dir)

    # get label files for cerebellum and cortex
    label_cereb = atlas_dir + '/tpl-SUIT' + \
        f'/atl-{cerebellum}_space-SUIT_dseg.nii'
    label_cortex = []
    for hemi in ['L', 'R']:
        label_cortex.append(atlas_dir + '/tpl-fs32k' +
                            f'/{cortex}.{hemi}.label.gii')

    # get parcel for both atlases
    atlas_cereb.get_parcel(label_cereb)
    atlas_cortex.get_parcel(label_cortex, unite_struct=False)

    # get the maps
    cifti_img = cdata.convert_cortex_to_cifti(
        weights, atlas_cereb, atlas_cortex)
    # save weight map
    if save_as is None:
        save_as = op.join(cdata.conn_dir, dataset_name, 'train',
                          f'{cortex}_{cerebellum}_{method}_{log_alpha}.dscalar.nii')
    nb.save(cifti_img, save_as)
    return


if __name__ == "__main__":
    # plot_weights(cerebellum='MDTB10')
    plot_weights(cerebellum="NettekovenSym34",
                 save_as=f'{ut.model_dir}/Atlases/NettekovenSym34_cortex_connmodel.dscalar.nii')
    # plot_weights(cerebellum="Verbal2Back")
    # plot_weights(cerebellum="Buckner7")
