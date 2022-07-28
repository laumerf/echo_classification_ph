import tensorflow as tf
import os
import numpy as np
import json
import pickle
import bz2
import multiprocessing as mp
from functools import partial
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from echo_ph.models import Unet
from echo_ph.data.segmentation import model_dict, segmentation_labels

"""
A script to segments one ECHO video at a time from out database, using pre-trained models from the echo-cv repo.
(bitbucket.org/rahuldeo/echocv).
TODO: Parallelize the code
"""


parser = ArgumentParser(
    description='Segment echo videos',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--models_path', default='/cluster/home/XXXX-6/segmentation_models',
                    help='Path to the directory storing the pre-trained models (checkpoints)')
parser.add_argument('--cache_dir', default='~/.heart_echo',
                    help='Path to the root directory storing the cached, processed echo videos to be segmented.')
parser.add_argument('--scaling_factor', default=0.25,
                    help='Scaling factor of cached videos to segment.')
parser.add_argument('--out_root_dir', default='segmented_results',
                    help='Path to the directory that should store the resulting segmentation maps')
parser.add_argument('--procs', type=int, default=32, help='Number of processes')
parser.add_argument('--max_frames', type=int, default=1000, help='Max number of frames to do segmentation prediction on')
parser.add_argument('--sampling_period', type=int, default=1,
                    help='If sample each frame, set to 1 (default). To sample every x-th frame, set to x.')
parser.add_argument('--samples', default=None, nargs='+',
                    help='Set this flag to segment only specific samples (videos) instead of all. Input space seperated'
                         'names of the desired samples/file names (including ending) after this flag')
parser.add_argument('--model_view', default='psax', choices=['psax', 'plax', 'a4c'],
                    help='What echocv-views to use for segmentation. Currently only psax, plax and a4c supported.')
parser.add_argument('--our_view', default='KAPAP', choices=['KAPAP', 'KAAP', 'CV'],
                    help='What view to use for our data. Currently only KAPAP, KAAP and CV supported.')
parser.add_argument('--save_visualisations', action='store_true', help='set this flag to save the visuals of '
                                                                       'segmentation mask for each image / frame ')


def save_segmentation_visuals(index, orig_image, segm_frame, outpath, videofile):
    """
    Saves each frame / image of an echo video, as well as an overlay of it's corresponding segmentation mask
    :param index: Index of this frame (e.g. if 4th frame in a video, index=4)
    :param orig_image: The original frame / image
    :param segm_frame: The predicted segmentation map for the frame
    :param outpath: The path to the directory storing the results
    :param videofile: Name of the video / sample name.
    :return:
    """
    segm_frame = segm_frame.astype(np.float32)
    segm_frame[segm_frame == 0] = np.nan
    os.makedirs(outpath, exist_ok=True)

    plt.figure(figsize=(7, 7))
    plt.axis('off')
    plt.imshow(segm_frame, cmap='Set3')
    plt.savefig(outpath + '/' + videofile + '_' + str(index) + '_' + 'seg.png')
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.axis('off')
    plt.imshow(orig_image, cmap='Greys_r')
    plt.savefig(outpath + '/' + videofile + '_' + str(index) + '_' + 'orig.png')
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.axis('off')
    plt.imshow(orig_image, cmap='Greys_r')
    plt.imshow(segm_frame, alpha=0.3)
    plt.savefig(outpath + '/' + videofile + '_' + str(index) + '_' + 'overlay.png')
    plt.close()
    print('')


def save_segm_map(segm_frames, view, outpath, videofile):
    """
    Saves the predicted segmentation maps for each frame in a video as a compressed pickle file.
    :param segm_frames: The predictions (segmented frames), with shape [num_frames, w, h, 1]
    :param view: The ehco-cv view used for this segmentation.
    :param outpath: Path to store the results.
    :param videofile: Name of the video-file / sample.
    """

    with bz2.BZ2File(os.path.join(outpath, videofile + "-frames.segment_pbz2"), 'wb') as file:
        pickle.dump(segm_frames, file)

    with open(os.path.join(outpath, videofile + "-segmentation_label.json"), 'w') as file:
        json.dump(segmentation_labels[view], file)


#TODO: Move dictionaries to a different file - maybe joint label file for segment and PH.


# NN Parameters
mean = 24
weight_decay = 1e-12
learning_rate = 1e-4
maxout = False
frame_size = 384

transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Grayscale(),
                                transforms.Resize(size=(frame_size, frame_size),
                                                  interpolation=InterpolationMode.BICUBIC)])


def segment_video(args, video_path):
    video_name = os.path.basename(video_path)[:-4]
    print('segmenting video', video_name)
    out_dir = os.path.join(args.out_root_dir, video_name, args.model_view)
    os.makedirs(out_dir, exist_ok=True)
    print('out_dir', out_dir)
    # === Get model ===
    graph = tf.Graph()
    with graph.as_default():
        num_labels = model_dict[args.model_view]['label_dim']
        checkpoint_path = model_dict[args.model_view]['restore_path']
        sess = tf.compat.v1.Session()
        model = Unet(mean, weight_decay, learning_rate, num_labels, maxout=maxout)
        sess.run(tf.compat.v1.local_variables_initializer())
    with graph.as_default():
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, os.path.join(args.models_path, checkpoint_path))

    # === Get data ===
    video_frames = np.load(video_path)
    num_frames = len(video_frames)
    num_frames = min(num_frames, args.max_frames)  # If only process specified number of frames
    # Resizing each frame that we want to process
    frames_resized = []
    for i in range(0, num_frames * args.sampling_period, args.sampling_period):
        img_frame = video_frames[i]
        transformed_frame = np.array(transform(img_frame))
        frames_resized.append(transformed_frame)
    # print('Starting to predict')
    frames_to_predict = np.array(frames_resized, dtype=np.float64).reshape(
        (len(frames_resized), frame_size, frame_size, 1))
    predicted_frames = []
    for frame in frames_to_predict:  # predict one frame at a time, to save memory
        frame_to_predict = np.expand_dims(frame, 0) # add 1 in front for batch_size 1
        pred_frame = np.argmax(model.predict(sess, frame_to_predict), -1)  # argmax max over last dim
        predicted_frames.append(np.squeeze(pred_frame, 0))
    predicted_frames = np.asarray(predicted_frames)
    print('saving video', video_name)
    save_segm_map(predicted_frames, args.model_view, out_dir, video_name)
    if args.save_visualisations:
        for i in range(num_frames):
            save_segmentation_visuals(i, frames_resized[i], predicted_frames[i], out_dir, video_name)


def main():
    args = parser.parse_args()
    videos_path = os.path.join(os.path.expanduser(args.cache_dir), str(args.scaling_factor))
    video_ending = args.our_view + '.npy'
    if args.samples is None:
        video_files = [file for file in os.listdir(videos_path) if file.endswith(video_ending)]
    else:
        video_files = [sample_name + video_ending for sample_name in args.samples]
    video_paths = [os.path.join(videos_path, video_file) for video_file in video_files]
    with mp.Pool(processes=args.procs) as pool:
        pool.map(partial(segment_video, args), video_paths)


if __name__ == '__main__':
    main()
