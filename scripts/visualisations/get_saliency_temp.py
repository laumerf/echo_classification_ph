import os
import torch
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from medcam import medcam
from echo_ph.data.transforms import get_transforms
from utils.helpers import get_index_file_path, get_temp_model
from echo_ph.data import EchoDataset
from echo_ph.visual.video_saver import VideoSaver
import cv2
import matplotlib.cm as cm
from scipy.ndimage import zoom

"""
A script to create saliency map visualisations on spatio-temporal models, either saving as frames or videos.
Can let it run on all frames in an entire dataset, or specified videos. Supports grad-cam, cam or guided backprop.
"""


parser = ArgumentParser(
    description='Arguments for visualising 3D grad cam',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_path', default=None, help='set to path of a model state dict, to evaluate on. '
                                                       'If None, use resnet18 pretrained on Imagenet only.')
parser.add_argument('--id', default='', type=str, help='Identify this model / run, for the output name')
parser.add_argument('--out_dir', default='vis_3d', help='Name of directory storing the results')
parser.add_argument('--model', default='r3d_18', choices=['r2plus1d_18', 'mc3_18', 'r3d_18', 'r3d_50',
                                                          'saliency_r3d_18', 'r3d_18_multi_view'],
                    help='What model architecture to use.')
parser.add_argument('--label_type', default='2class_drop_ambiguous',
                    choices=['2class', '2class_drop_ambiguous', '3class'])
parser.add_argument('--cache_dir', default='~/.heart_echo')
parser.add_argument('--scale', default=0.25)
parser.add_argument('--view', nargs='+', type=str, default=['KAPAP'], help='What view (s) to use')
parser.add_argument('--fold', default=0, type=int,
                    help='In case of k-fold cross-validation, set the current fold for this training.'
                         'Will be used to fetch the relevant k-th train and valid index file')
parser.add_argument('--k', default=10, type=int,
                    help='In case of k-fold cross-validation, set the k, i.e. how many folds all in all. '
                         'Will be used to fetch the relevant k-th train and valid index file')
parser.add_argument('--n_workers', default=8, type=int)
parser.add_argument('--vis_type', default='gcam', choices=['gcam', 'gbp', 'cam'],
                    help='what visualis. type (backend) to use')

# Amount of samples / clips to pick
parser.add_argument('--num_rand_samples', type=int, default=None,
                    help='Set number of samples (vide-clips) to evaluate for each model, if not all frames.')
parser.add_argument('--all_frames', action='store_true', default=None,
                    help='Get all frames of a video')
parser.add_argument('--max_frame', type=int, default=50,
                    help='Only valid in combination with m flag. '
                         'Get sequential frames of a video from frame 0, but limit len to max_frame.')

# Other
parser.add_argument('--train_set', action='store_true', help='Also get grad cam for the images in the training set, '
                                                             'with random augmentation (type 3)')
parser.add_argument('--video_ids', default=None, nargs='+', type=int,
                    help='Instead of getting results acc.to index file, get results for specific video ids')

# What to save / show
parser.add_argument('--save_video_clips', action='store_true', help='Save individual video clips')
parser.add_argument('--save_video', action='store_true', help='Save entire video, batching up video clips')
parser.add_argument('--save_frames', action='store_true', help='Save individual frames')
parser.add_argument('--show', action='store_true', help='If to show grad cam images')
parser.add_argument('--img_size', type=int, default=224, help='Size that frames are resized to')
parser.add_argument('--corr_thres', default=None, type=float,
                    help='If only save videos that have corr ratio over certain threshold. E.g. 0.9')
parser.add_argument('--wrong_thres', default=None, type=float,
                    help='If only save "bad" video predictions that have corr ratio below certain threshold. E.g. 0.3')
# Temporal param
parser.add_argument('--clip_len', type=int, default=12, help='How many frames to select per video')
parser.add_argument('--period', type=int, default=1, help='Sample period, sample every n-th frame')


def get_data_loader(train=False):
    """
    Get data loader for temporal data.
    :param train: True if training data loader
    :return: data loader
    """
    if args.video_ids is None:
        index_file_path = get_index_file_path(args.k, args.fold, args.label_type, train=train)
    else:
        index_file_path = None
    label_path = os.path.join('label_files', 'labels_' + args.label_type + '.pkl')
    aug_type = 4 if train else 0
    transforms = get_transforms(index_file_path, dataset_orig_img_scale=args.scale, resize=args.img_size,
                                augment=aug_type, fold=args.fold, valid=(not train), view=args.view)
    dataset = EchoDataset(index_file_path, label_path, cache_dir=args.cache_dir,
                          transform=transforms, scaling_factor=args.scale, procs=args.n_workers,
                          view=args.view,  num_rand_frames=args.num_rand_samples, temporal=True,
                          clip_len=args.clip_len, period=args.period, video_ids=args.video_ids,
                          all_frames=args.all_frames, max_frame=args.max_frame)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.n_workers)
    return data_loader


def normalize(x):
    """
    Normalizes data both numpy or tensor data to range [0,1].
    """
    if isinstance(x, torch.Tensor):
        if torch.min(x) == torch.max(x):
            return torch.zeros(x.shape)
        return (x-torch.min(x))/(torch.max(x)-torch.min(x))
    else:
        if np.min(x) == np.max(x):
            return np.zeros(x.shape)
        return (x - np.min(x)) / (np.max(x) - np.min(x))


def overlay(raw_input, saliency_map):
    """
    Overlay attention map on top of a raw image.
    :param raw_input: The original input frame (resized or original size). Shape should be ch x w x h.
    :param saliency_map: The attention map of that frame. Shape should be ch x w x h.
    :return: The overlayed image of shape (W X H X 3) => needed to save as image.
    """
    saliency_map = np.asarray(saliency_map)
    raw_input = np.asarray(raw_input)
    raw_input = raw_input.transpose(1, 2, 0)  # Reshape to : W X H X CH
    saliency_map = normalize(saliency_map.astype(np.float))
    if np.max(raw_input) > 1:
        raw_input = raw_input.astype(float)
        raw_input /= 255
    saliency_map = cm.jet_r(saliency_map)[..., :3].squeeze()
    saliency_map = (saliency_map.astype(float) + raw_input.astype(float)) / 2
    saliency_map \
        *= 255
    return saliency_map.astype(np.uint8)


def get_cam_saliency_tubes(model, layerout, label):
    """
    Get the CAM saliency spatio-temporal tubes from last conv layer of model (layerout), and model weights.
    :param model: The model
    :param layerout: The last conv layer of the model.
    :param label: Label of this subject.
    :return: CAM saliency clip
    """
    pred_weights = model.fc.weight.data.detach().cpu().numpy().transpose()
    layerout = layerout.detach().cpu()[0].numpy()  # remove batch part
    layerout = layerout.swapaxes(0, 3)
    cam = np.zeros(dtype=np.float32, shape=layerout.shape[0:3])
    for i, w in enumerate(pred_weights[:, label]):
        cam += w * layerout[:, :, :, i]

    # Resize CAM to frame level
    cam = cam.swapaxes(1, 0)
    cam = zoom(cam, (6, 16, 16))  # out cam is 2x14x14, so multiply by 6x16x16 to get 12x224z224 (out inp size)

    # Normalize
    cam = normalize(cam)
    return cam


def visualise_save_saliency(data_loader, model, device, subset='valid'):
    """
    Get the saliency map of choice, overlay with original ECHO video and save the resulting clips or frames.
    :param data_loader: Data loader
    :param model: Model
    :param device: Device
    :param subset: valid or train
    :return:
    """
    id = '' if args.id == '' else '_' + args.id
    out_dir = f'{args.out_dir}{id}_{args.vis_type}_fold_{args.fold}'
    video_clips = {}
    for batch in data_loader:
        inp = batch['frame']
        if len(args.view) == 1:  # single view
            view = args.view[0]
            inp = inp[view].to(device).transpose(2, 1)  # Reshape to: (batch_size, channels, seq-len, W, H)
        sample_name = batch['sample_name'][0]
        video_id = int(sample_name.split('_')[0])
        label = batch['label'][0].item()

        # Using model to predict, and get also saliency map.
        out, saliency_map = model(inp)
        pred = torch.max(out, dim=1).indices[0].item()
        corr = 'CORR' if label == pred else 'WRONG'
        print(sample_name, corr)
        title = f'{sample_name}-{corr}-{label}.jpg'
        video = np.swapaxes(inp, 1, 2)[0]  # re-shape to (no_frames, ch, W, H)
        raw_vid_clip = video.cpu().detach().numpy()
        if args.vis_type == 'cam':  # Approach acc. to paper https://arxiv.org/abs/1902.01078
            saliency_clip = get_cam_saliency_tubes(model, saliency_map, label)
        else:
            saliency_clip = np.swapaxes(saliency_map, 1, 2)[0]  # re-shape to (no_frames, ch, W, H)
            saliency_clip = saliency_clip.squeeze(1).cpu().detach().numpy()
        if video_id not in video_clips:
            video_clips[video_id] = (saliency_clip, raw_vid_clip, [title])
        else:
            extended_attention = np.append(video_clips[video_id][0], saliency_clip, axis=0)
            extended_video = np.append(video_clips[video_id][1], raw_vid_clip, axis=0)
            extended_title = np.append(video_clips[video_id][2], title)
            video_clips[video_id] = (extended_attention, extended_video, extended_title)

        # Saving and / or showing the saliency frames / clips
        if args.save_video_clips:
            overlay_clip = [overlay(frame, np.expand_dims(att_frame, axis=0)) for (frame, att_frame) in
                            zip(raw_vid_clip, saliency_clip)]
            clip_out_dir = os.path.join(out_dir + '_clip', subset)
            print("save to folder", clip_out_dir)
            vs = VideoSaver(title, overlay_clip, out_dir=clip_out_dir, fps=7)
            vs.save_video()
        if args.show or args.save_frames:
            if args.save_frames:
                out_frame_dir = os.path.join(out_dir + '_frames', str(video_id))
                os.makedirs(out_frame_dir, exist_ok=True)
                print('saving frames to out_dir', out_frame_dir)
            for i in range(len(video)):  # For each frame in current clip
                img = video[i]
                saliency_frame = saliency_clip[i]
                title = f'{sample_name}-{i}-{corr}-{label}.jpg'
                if args.show:
                    # Show overlay
                    plt.imshow(img.squeeze().cpu(), cmap='Greys_r')
                    plt.imshow(saliency_frame.squeeze().cpu(), cmap='jet', alpha=0.5)
                    plt.title(title)
                    plt.show()
                    # Show only original
                    plt.imshow(img.squeeze().cpu(), cmap='Greys_r')
                    plt.title(title)
                    plt.show()
                    # Show only saliency map
                    plt.title(title)
                    plt.imshow(saliency_frame.squeeze().cpu(), cmap='jet')
                    plt.title(title)
                    plt.show()
                if args.save_frames:
                    x = overlay(img, saliency_frame)
                    cv2.imwrite(os.path.join(out_frame_dir, title), x)
                    cv2.imwrite(os.path.join(out_frame_dir, 'orig_' + title), img.squeeze().cpu().numpy() * 255.0)
    # Save entire saliency overlayed video, combining all clips
    if args.save_video:
        save_all = args.corr_thres is None and args.wrong_thres is None
        good = args.corr_thres is not None
        bad = args.wrong_thres is not None
        for video_id in video_clips:
            overlay_clips = [overlay(vid, np.expand_dims(att_vid, axis=0)) for (att_vid, vid) in zip(video_clips[video_id][0],
                                                                                         video_clips[video_id][1])]
            clip_titles = video_clips[video_id][2]  # get titles for clips in video, to extract no. corrs & label
            clips_corrs = np.asarray([frame_title.split('-')[1] for frame_title in clip_titles])
            ratio_corr = (clips_corrs == 'CORR').sum() / len(clip_titles)
            if save_all or (good and ratio_corr >= args.corr_thres) or (bad and ratio_corr <= args.wrong_thresh):
                true_label = clip_titles[0].split('-')[-1][:-4]
                video_title = f'{video_id}-{ratio_corr:.2f}-{true_label}.jpg'
                vs = VideoSaver(video_title, overlay_clips, out_dir=os.path.join(out_dir + '_video', subset), fps=7)
                vs.save_video()


def main():
    # Argument checks
    if args.num_rand_samples is None and not args.all_frames:
        print('Must specify either number of random samples or set all_frames flag')
        exit()
    if args.vis_type == 'cam' and args.model != 'saliency_r3d_18':
        print("In order to visualise with CAM, must have saliency model")
        exit()
    if args.model_path is not None:
        model_name = os.path.basename(args.model_path)
        if not model_name.startswith('fold' + str(args.fold)):
            print('Fold for getting data does not match fold of model')
            exit()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Get data
    val_data_loader = get_data_loader()
    print("Done loading valid data")

    # Get and evaluate model
    num_classes = 2 if args.label_type.startswith('2') else 3
    return_last = True if args.vis_type == 'cam' else False
    model = get_temp_model(args.model, num_classes, pretrained=True, device=device,
                           return_last_saliency=return_last, views=args.view)
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    # Instantiate model with grad cam & evaluate
    if args.vis_type != 'cam':
        model = medcam.inject(model, return_attention=True, backend=args.vis_type)
    model.eval()

    # Get visualisations
    visualise_save_saliency(val_data_loader, model, device, subset='valid')
    if args.train_set:
        train_data_loader = get_data_loader(train=True)
        visualise_save_saliency(train_data_loader, model, device, subset='train')


if __name__ == '__main__':
    args = parser.parse_args()
    main()