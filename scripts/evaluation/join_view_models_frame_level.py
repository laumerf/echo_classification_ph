from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from echo_ph.evaluation.metrics import read_results, Metrics
import os
import numpy as np

"""
This script gets results from models of different views, by joining the frame-level prediction, 
before concatenating for subject-level prediction.
"""

parser = ArgumentParser(
    description='Join frame-level predictions, and get metrics',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('base_dir', help='Path to the base directory where model files are stored')
parser.add_argument('--res_files', help='Path to the results file of kapap view', nargs='+')
parser.add_argument('--views', help='Path to the results file of kapap view', nargs='+')
parser.add_argument('--no_folds', type=int, default=10, help='Number of folds')


def main():
    args = parser.parse_args()
    no_folds = args.no_folds
    res_files = args.res_files
    res_paths = [os.path.join(args.base_dir, res_file) for res_file in res_files]
    views = args.views
    all_res = {key: [] for key in ['Video bACC', 'Video F1 (micro)', 'Video CI']}
    for fold in range(0, no_folds):
        print('fold', fold)
        samples = []
        targets = []
        preds = []
        for res_path, view in zip(res_paths, views):
            fold_path = sorted(os.listdir(res_path))[fold]
            fold_dir = os.path.join(res_path, fold_path)
            fold_preds, fold_probs, fold_targets, fold_samples, fold_outs = read_results(fold_dir)
            samples.extend(fold_samples)
            targets.extend(fold_targets)
            preds.extend(fold_preds)
        m = Metrics(targets, samples, preds=preds, binary=False, tb=False)
        res = m.get_per_subject_scores()
        for metric, val in res.items():
            all_res[metric].append(val)
    for metric, metric_values in all_res.items():
        mean = np.mean(metric_values)
        std = np.std(metric_values)
        metric_str = f'{mean:.2f} (std: {std:.2f})'
        print(metric, ':', metric_str)


if __name__ == '__main__':
    main()
