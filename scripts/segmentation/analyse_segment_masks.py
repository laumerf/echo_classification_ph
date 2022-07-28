import os
import numpy as np
import pickle
import bz2
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

"""
This script loads echo segmentation masks (saved as compressed pickle file), for analysis.
Assumes the following folder structure: 
/segmented_results
    /1KAPAP
        /psax
            /1KAPAP-frames.segment_pbz2
            /1KPAP-segmentation_label.json
        /plax
    /2KAPAP
        /psax
        /plax
Note, a similar but more visual processing can be found under notebooks/visualise_segmentation.ipynb.
"""


parser = ArgumentParser(
    description='Load and analyse of echo segmentation masks',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--segment_dir', default='segmented_results',
                    help='Path to the (root) directory storing segmentation results')
parser.add_argument('--echocv_view', default='psax', choices=['psax', 'plax', 'all'],
                    help='For what echo-view (according to echo-cv repo) should this analysis be performed on')
parser.add_argument('--samples', default=None, nargs='*',
                    help='Set this flag to only analyse segmentation masks of specific samples (videos) instead of all '
                         'the videos in the segment_dir. Input space seperated samples names (number and view, e.g.'
                         '30KAPAP) for which to analyse the segmentation masks.')


def main():
    args = parser.parse_args()
    if args.echocv_view == 'all':
        views = ['psax', 'plax']
    else:
        views = [args.echocv_view]
    for view in views:
        if args.samples is None:  # All samples found in segment dir
            segment_mask_paths = []
            sample_names = []
            for sample_sub_dir in os.listdir(args.segment_dir):
                sample_names.append(sample_sub_dir)
                segment_mask_file = sample_sub_dir + '-frames.segment_pbz2'
                path = os.path.join(args.segment_dir, sample_sub_dir, view, segment_mask_file)
                segment_mask_paths.append(path)
        else:
            segment_mask_paths = [os.path.join(args.segment_dir, sample, view, sample + '-frames.segment_pbz2')
                                  for sample in args.samples]
            sample_names = args.samples

        for path, sample_name in zip(segment_mask_paths, sample_names):
            print(f'==== analysing segmentation results for sample {sample_name } ====')
            label_path = os.path.join(os.path.dirname(path), sample_name + '-segmentation_label.json')
            with open(label_path, 'r') as file:
                labels = json.load(file)
            print('labels', labels)
            data = bz2.BZ2File(path, 'rb')
            segm_mask = np.asarray(pickle.load(data))
            print('mask shape', segm_mask.shape)
            w, h = segm_mask.shape[1:3]
            print('w', w, 'h', h)
            max_expansion_frame = -1
            max_expansion = -1
            lv_rv_valu_to_id = {}
            for i in range(0, len(segm_mask), 1):  # Just look at every 4th frame
                segm_mask_frame = segm_mask[i]
                print(f'--> Frame {i}:')
                total_cnt = 0
                for label in labels.keys():
                    cnt_label = np.count_nonzero(segm_mask_frame == labels[label])
                    if label == 'lv' or label == 'rv':  # Try this!
                        total_cnt += cnt_label
                    area_of_total = cnt_label / (w * h)
                    print(f'\t{label}: cnt={cnt_label}, area_of_total={area_of_total * 100:.2f}%')
                rv = np.count_nonzero(segm_mask_frame == labels['rv'])
                lv = np.count_nonzero(segm_mask_frame == labels['lv'])
                lv_rv = rv + rv
                if lv_rv not in lv_rv_valu_to_id:
                    lv_rv_valu_to_id[lv_rv] = [i]
                else:
                    lv_rv_valu_to_id[lv_rv].append(i)
                print(f'\tratio lv/rv: {lv/rv * 100}')
                if total_cnt > max_expansion:
                    max_expansion = total_cnt
                    max_expansion_frame = i
                print(f'\tcombined:  cnt={total_cnt}, area_of_total={total_cnt / (w * h) * 100:.2f}%')
            print(f'Max expansion frame is nr {max_expansion_frame}, with total cnt: {max_expansion} and area: '
                  f'{max_expansion / (w * h) * 100:.2f}%')
            print('lv_rv_map')
            print(sorted(lv_rv_valu_to_id))
            lv_rv_vals = np.asarray(list(lv_rv_valu_to_id))
            # percentile_90 = np.percentile(list(lv_rv_valu_to_id), 90)
            percentile_90 = np.percentile(lv_rv_vals, 90, interpolation="nearest")
            top_90p_vals = lv_rv_vals[lv_rv_vals >= percentile_90]
            print('percentile 90', percentile_90)
            for top90 in top_90p_vals:
                print(top90, lv_rv_valu_to_id[top90])
            print('top 5')
            max_five = sorted(lv_rv_valu_to_id)[-5:]  # get 5 highest values
            for max in max_five:
                print(max, lv_rv_valu_to_id[max])


if __name__ == '__main__':
    main()

