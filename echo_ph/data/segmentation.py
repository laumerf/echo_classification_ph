import os
import numpy as np
import bz2
import json
import pickle
from collections import defaultdict


class SegmentationAnalyser:
    def __init__(self, sample_name, segm_res_dir, model_view='psax'):
        dir = os.path.join(segm_res_dir, sample_name, model_view)
        segm_mask_path = os.path.join(dir, sample_name + '-frames.segment_pbz2')
        label_path = os.path.join(dir, sample_name + '-segmentation_label.json')
        label_path = os.path.join(os.path.dirname(label_path), sample_name + '-segmentation_label.json')
        with open(label_path, 'r') as file:
            self.labels = json.load(file)
        data = bz2.BZ2File(segm_mask_path, 'rb')
        self.segm_mask = np.asarray(pickle.load(data))
        self.w, self.h = self.segm_mask.shape[1:3]

    def extract_max_percentile_frames(self, percentile=90, min_exp=False):
        volume_to_frame_nr = defaultdict(list)  # initialise the dict with an empty list (to add frame ids)
        for frame_nr, segm_mask_frame in enumerate(self.segm_mask):
            rv_vol = np.count_nonzero(segm_mask_frame == self.labels['rv'])
            lv_vol = np.count_nonzero(segm_mask_frame == self.labels['lv'])
            volume_to_frame_nr[lv_vol + rv_vol].append(frame_nr)

        volume_list = np.asarray(list(volume_to_frame_nr))
        p = (100-percentile) if min_exp else percentile  # if find minp, reverse
        percentile = np.percentile(volume_list, p, interpolation="nearest")
        if min_exp:
            top_percentile_volumes = volume_list[volume_list <= percentile]
        else:
            top_percentile_volumes = volume_list[volume_list >= percentile]
        max_expansion_frames = []
        for top_p in top_percentile_volumes:
            top_frame = volume_to_frame_nr[top_p]
            max_expansion_frames.extend(top_frame)
        return max_expansion_frames

    def get_segm_mask(self):
        return self.segm_mask


model_dict = {
            'psax': {
                'label_dim': 4,
                'restore_path': 'psax_45_20_all_model.ckpt-9300'
            },
            'plax': {
                'label_dim': 7,
                'restore_path': 'plax_45_20_all_model.ckpt-9600'
            },
            'a4c': {
                'label_dim': 6,
                'restore_path': 'a4c_45_20_all_model.ckpt-9000'
            }
}

segmentation_labels = {
            'plax': {
                'lv': 1,  # left ventricle
                'ivs': 2,  # interventricular septum
                'la': 3,  # left atrium
                'rv': 4,  # right ventricle
                'pm': 5,  # papilliary muscle
                'aor': 6  # aortic root (ascending)
            },
            'psax': {
                'lv': 2,  # left ventricle
                'rv': 3,  # right ventricle
                'lv-o': 1  # left ventricle outer tissue
            },
            'a4c': {
                'lv': 2, # left ventricle
                'rv': 3, # right ventricle
                'la': 4, # left atrium
                'ra': 5, # right atrium
                'lv-o': 1 # left ventricle outer tissue
            }
}

our_view_to_segm_view = {
    'CV': 'a4c',
    'KAPAP': 'psax'
}


