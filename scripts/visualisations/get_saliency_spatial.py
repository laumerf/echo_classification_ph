import torch
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from echo_ph.data.transforms import get_transforms
from utils.helpers import get_index_file_path
from echo_ph.data import EchoDataset
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from echo_ph.models.resnets import get_resnet18
from echo_ph.visual.video_saver import VideoSaver
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import os

"""
A script to create grad-cam visualisations on spatial models, either saving as frames or videos.
Can let it run on all frames in an entire dataset, or specified videos.
"""

parser = ArgumentParser(
    description='Arguments for visualising grad cam',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_path', default=None, help='set to path of a model state dict, to evaluate on. '
                                                       'If None, use resnet18 pretrained on Imagenet only.')
parser.add_argument('--label_type', default='2class_drop_ambiguous',
                    choices=['2class', '2class_drop_ambiguous', '3class'])
parser.add_argument('--cache_dir', default='~/.heart_echo')
parser.add_argument('--scale', default=0.25)
parser.add_argument('--view', default='KAPAP')
parser.add_argument('--img_size', type=int, default=224, help='Size that frames are resized to')
parser.add_argument('--fold', default=0, type=int,
                    help='In case of k-fold cross-validation, set the current fold for this training.'
                         'Will be used to fetch the relevant k-th train and valid index file')
parser.add_argument('--k', default=10, type=int,
                    help='In case of k-fold cross-validation, set the k, i.e. how many folds all in all. '
                         'Will be used to fetch the relevant k-th train and valid index file')
parser.add_argument('--n_workers', default=8, type=int)
parser.add_argument('--max_p', default=95, type=int)
# Optional additional arguments
parser.add_argument('--min_expansion', action='store_true',
                    help='Percentile for min expansion frames instead of maximum')
parser.add_argument('--num_rand_frames', type=int, default=None,
                    help='Set this only if get random frames instead of max/min')
parser.add_argument('--all_frames', action='store_true', default=None,
                    help='Get all frames of a video')
parser.add_argument('--max_frame', type=int, default=50,
                    help='Only valid in combination with all_frames flag. '
                         'Get sequential frames of a video from frame 0, but limit len to max_frame')
parser.add_argument('--save_frames', action='store_true', help='If save grad cam images on a frame-level')
parser.add_argument('--save_video', action='store_true', help='If also to save video visualisations')
parser.add_argument('--show', action='store_true', help='If to show grad cam images')
parser.add_argument('--train_set', action='store_true', help='Also get grad cam for the images in the training set, '
                                                             'with random augmentation (type 3)')
parser.add_argument('--segm_only', action='store_true', help='Only evaluate on the segmentation masks')
parser.add_argument('--video_ids', default=None, nargs='+', type=int,
                    help='Instead of getting results acc.to index file, get results for specific video ids')


def get_data_loader(train=False):
    """
    Get data loader for sptial data.
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
                                augment=aug_type, fold=args.fold, valid=(not train), view=args.view,
                                crop_to_corner=args.crop, segm_mask_only=args.segm_only)
    dataset = EchoDataset(index_file_path, label_path, cache_dir=args.cache_dir,
                          transform=transforms, scaling_factor=args.scale, procs=args.n_workers,
                          percentile=args.max_p, view=args.view, min_expansion=args.min_expansion,
                          num_rand_frames=args.num_rand_frames, segm_masks=args.segm_only, video_ids=args.video_ids,
                          all_frames=args.all_frames, max_frame=args.max_frame)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.n_workers)
    return data_loader


def visualise_save_saliency_frames(data_loader, model, saliency_map, device, subset='valid'):
    """
    Get the saliency map of choice, overlay it with original ECHO and save or visualise the resulting frames.
    :param data_loader: Data loader
    :param model: Model
    :param saliency_map: The raw saliency map
    :param device: Device
    :param subset: valid or train
    :return:
    """
    target_category = None
    if args.save_frames or args.save_video:
        # prepare output directory
        model_name = os.path.basename(args.model_path)[:-3]
        output_dir = os.path.join('grad_cam_vis', model_name, subset)
        os.makedirs(output_dir, exist_ok=True)
    video_frames = {}
    for batch in data_loader:
        img = batch['frame'][args.view].to(device)
        sample_name = batch['sample_name'][0]  # get first, because batch size 1
        video_id = sample_name.split('_')[0]  # get first, because batch size 1
        print('Processing video_id', video_id)
        label = batch['label'][0].item()  # get first, because batch size 1
        pred = torch.max(model(img), dim=1).indices[0].item()
        corr = 'CORR' if label == pred else 'WRONG'
        title = f'{sample_name}-{corr}-{label}.jpg'
        grayscale_cam = saliency_map(input_tensor=img,
                                     target_category=target_category)
        img = np.stack((img.squeeze().cpu(),) * 3, axis=-1)  # create a 3-channel image from the grayscale img
        try:
            cam_image = show_cam_on_image(img, grayscale_cam[0])
        except:
            print(f'failed for sample {sample_name}, max is {img.max()}, min is {img.min()}')
        if video_id not in video_frames:
            video_frames[video_id] = ([cam_image], [title])
        else:
            video_frames[video_id][0].append(cam_image)
            video_frames[video_id][1].append(title)
        if args.show:
            plt.imshow(cam_image)
            plt.title(title)
            plt.show()
    if args.save_frames or args.save_video:
        for video_id in video_frames:
            frame_titles = video_frames[video_id][1]  # get titles for frames in video, to extract no. corrs & label
            frame_corrs = np.asarray([frame_title.split('-')[1] for frame_title in frame_titles])
            ratio_corr = (frame_corrs == 'CORR').sum() / len(frame_titles)
            print('ratio corr', ratio_corr)
            if args.save_frames:
                out_dir = os.path.join(output_dir, str(video_id))
                print('saving frames to out_dir', out_dir)
                os.makedirs(out_dir, exist_ok=True)
                for grad_cam_frame, title in zip(video_frames[video_id][0], video_frames[video_id][1]):
                    cv2.imwrite(os.path.join(out_dir, title), grad_cam_frame)
            if args.save_video:
                true_label = frame_titles[0].split('-')[-1][:-4]
                video_title = f'{video_id}-{ratio_corr:.2f}-{true_label}.jpg'
                out_dir = output_dir + '_video'
                vs = VideoSaver(video_title, video_frames[video_id][0], out_dir=out_dir, fps=10)
                vs.save_video()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Get data
    val_data_loader = get_data_loader()
    if args.train_set:
        train_data_loader = get_data_loader(train=True)
    print("Done loading data")

    # Get and evaluate model, and raw saliency map
    num_classes = 2 if args.label_type.startswith('2') else 3
    model = get_resnet18(num_classes=num_classes)
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    model = model.to(device)
    target_layers = [model.layer4[-1]]
    saliency = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    print("Done initialising grad cam with model")

    # Get and save / visualise the saliency frames
    visualise_save_saliency_frames(val_data_loader, model, saliency, device)
    if args.train_set:
        visualise_save_saliency_frames(train_data_loader, model, saliency, device, subset='train')


if __name__ == '__main__':
    args = parser.parse_args()
    main()
