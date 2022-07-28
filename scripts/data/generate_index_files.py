import numpy as np
import os
from sklearn.model_selection import train_test_split
import pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

"""
This script generates index files for training and validation data in a stratified way, keeping class ratios - according
to a given class formulation (i.e. the method used to convert raw labels to classes).
When --k is specified > 1, this generates train-valid splits for k-fold cross-validation.
"""

parser = ArgumentParser(
    description='Generates index files for train and validation.',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--label_file_path', type=str, default=None,
                    help='Full path to the pickle file containing the desired labels - to split evenly.'
                         'E.g.: label_files/labels_3class.pkl. If None - apply for all files in --label_dir instead')
parser.add_argument('--label_dir', type=str, default='label_files',
                    help='Path to directory containing all label files - in order to create a pair of index files for '
                         'each label file. Set to None, in order to only create index files for a given label file')
parser.add_argument('--valid_ratio', type=float, default=0.2,
                    help='Ratio of total data used for validation')
parser.add_argument('--test_ratio', type=float, default=0.1,
                    help='Ratio of total data used for separate test set. If not have a test set - set to 0')
parser.add_argument('--video_cache_dir', type=str, default='~/.heart_echo',
                    help='Path of the video cache dir (usually: ~/.heart_echo).')
parser.add_argument('--scale_factor', default=0.25,
                    help='Scaling factor of the cached videos')
parser.add_argument('--out_dir', default='index_files',
                    help='Path to directory where results should be stored')
parser.add_argument('--no_folds', type=int, default=None,
                    help='How many folds to use for K-fold validation, i.e. how many different train-val splits.'
                         'If None, only single split is created')

TEST_SEED = 1337


def print_res(train_labels, valid_labels, test_labels=None):
    """
    Prints the results, i.e. the ratio of each label.
    :param train_labels: List of labels for train samples
    :param valid_labels: List of labels for valid samples
    :param test_labels: (Optional) list of labels for test samples, if seperate test set.
    :return:
    """
    cnt_val = dict()
    cnt_train = dict()
    cnt_test = dict()
    train_total_cnt = 0
    valid_total_cnt = 0
    test_total_cnt = 0
    for val_label in valid_labels:
        if val_label in cnt_val:
            cnt_val[val_label] += 1
        else:
            cnt_val[val_label] = 1
        valid_total_cnt += 1
    for train_label in train_labels:
        if train_label in cnt_train:
            cnt_train[train_label] += 1
        else:
            cnt_train[train_label] = 1
        train_total_cnt += 1
    if test_labels is not None:
        for test_label in test_labels:
            if test_label in cnt_test:
                cnt_test[test_label] += 1
            else:
                cnt_test[test_label] = 1
            test_total_cnt += 1

    print()
    print('Number of test samples:', test_total_cnt)
    print('Number of training samples:', train_total_cnt)
    print('Number of valid samples:', valid_total_cnt)
    for label in cnt_train.keys():
        print(f'Train ratio label {label}: {cnt_train[label] / train_total_cnt}')
    print('Test distribution:')
    print(cnt_test)
    print('Train distribution:')
    print(cnt_train)
    print('Valid distribution:')
    print(cnt_val)


def main():
    args = parser.parse_args()
    video_cache_dir = os.path.join(os.path.expanduser(args.video_cache_dir), str(args.scale_factor))
    video_ending = 'KAPAP.npy'
    video_ending_len = len(video_ending)
    kapap_cache_videos = [video for video in os.listdir(video_cache_dir) if video.endswith(video_ending)]
    label_dicts = {}
    if args.label_file_path is None:  # create index files for all label files in label directory
        for label_file in os.listdir(args.label_dir):
            label_file_path = os.path.join(args.label_dir, label_file)
            with open(label_file_path, "rb") as file:
                label_dict = pickle.load(file)
            label_dicts[label_file] = label_dict
    else:
        with open(args.label_file_path, "rb") as file:
            label_dicts[os.path.basename(args.label_file_path)] = pickle.load(file)

    no_splits = 1 if args.no_folds is None else args.no_folds

    if args.test_ratio > 0:
        # Create a separate test set, that is common for all label types
        # (Create acc. to class imbalance from labels_2class_drop_ambiguous.pkl), as it is most restrictive on videos.
        labels_in_use = []
        video_ids_in_use = []
        for video in kapap_cache_videos:
            video_id = int(video[:-video_ending_len])
            if video_id not in label_dicts['labels_2class_drop_ambiguous.pkl']:
                print(f'video {video_id} does not have a legal label - skipping')
            else:
                label = label_dicts['labels_2class_drop_ambiguous.pkl'][video_id]
                labels_in_use.append(label)
                video_ids_in_use.append(video_id)
        # Split samples into train and test, stratified according to the labels
        samples_rest, samples_test, labels_rest, labels_test = train_test_split(np.asarray(video_ids_in_use),
                                                                                labels_in_use,
                                                                                test_size=args.test_ratio,
                                                                                shuffle=True,
                                                                                stratify=labels_in_use,
                                                                                random_state=TEST_SEED)
        np.save(os.path.join(args.out_dir, 'test_samples.npy'), samples_test)
        print("Have created test set. Will now create validation sets for each label dict")
    # for label_dict, label_file in zip(label_dicts, label_files):
    for label_file, label_dict in label_dicts.items():
        for fold in range(no_splits):
            print("\nResults for", label_file)
            labels_in_use = []
            video_ids_in_use = []
            for video in kapap_cache_videos:
                video_id = int(video[:-video_ending_len])
                if video_id not in label_dict:
                    print(f'video {video_id} does not have a legal label - skipping')
                elif args.test_ratio > 0 and video_id in samples_test:  # Can't include samples that are already in test
                    print(f'video {video_id} is in test set')
                else:
                    label = label_dict[video_id]
                    labels_in_use.append(label)
                    video_ids_in_use.append(video_id)
            # Split samples into train and test, stratified according to the labels
            samples_train, samples_val, y_train, y_val = train_test_split(np.asarray(video_ids_in_use), labels_in_use,
                                                                          test_size=args.valid_ratio,
                                                                          shuffle=True, stratify=labels_in_use,
                                                                          random_state=fold)  # seed is the fold id
            # Save index files for train and test and valid
            file_name = label_file.split('labels_')[1][:-4]
            file_ending = '' if args.no_folds is None else '_' + str(fold)
            out_dir = args.out_dir if args.no_folds is None else os.path.join(args.out_dir, 'k' + str(args.no_folds))
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, 'train_samples_' + file_name + file_ending + '.npy'), samples_train)
            np.save(os.path.join(out_dir, 'valid_samples_' + file_name + file_ending + '.npy'), samples_val)
            # Print results
            if args.test_ratio > 0:
                print_res(y_train, y_val, labels_test)
            else:
                print_res(y_train, y_val)


if __name__ == '__main__':
    main()
