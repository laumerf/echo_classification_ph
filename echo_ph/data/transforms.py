from torchvision import transforms
import torch
import cv2
import os
import math
from pathlib import Path
import random
import numpy as np
from functools import partial
import torchvision.transforms.functional as F
from sympy import Line, Circle
from echo_ph.data.segmentation import segmentation_labels, our_view_to_segm_view

# Imports for mask generation
from scipy.spatial import ConvexHull
import multiprocessing as mp
FILL_VAL = 0.3
TORCH_SEED = 0
torch.manual_seed(TORCH_SEED)  # Fix a seed, to increase reproducibility
torch.cuda.manual_seed(TORCH_SEED)
torch.cuda.manual_seed_all(TORCH_SEED)
np.random.seed(TORCH_SEED)
random.seed(TORCH_SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False

from PIL import Image, ImageChops


def get_mask_fn(mask_dir, size, img_scale, label_type, fold):
    fn = os.path.join(mask_dir,  # view is already defined in the mask path!
                 f'{size}_{int(100 * float(img_scale))}_percent_'
                 f'label{label_type}_fold{fold}.pt')
    return fn


class Trim():
    def __init__(self, border=0):
        self.border = border
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    def __call__(self, sample):
        sample, p_id = sample
        _, W, H = sample.size()
        pimg = self.to_pil(sample)
        bg = Image.new(pimg.mode, (W, H), self.border)
        diff = ImageChops.difference(pimg, bg)
        bbox = diff.getbbox()
        if bbox:
            return self.to_tensor(pimg.crop(bbox)), p_id


class CropToCorners():
    """
    Crop echo to have the its four corners at the border
    """
    def _get_masks(self):
        mask_fn = get_mask_fn(self.mask_path, -1, self.orig_img_scale, self.label_type, self.fold)
        if not os.path.exists(mask_fn):
            gen_masks(mask_fn, -1, self.orig_img_scale, self.index_file_path, view=self.view)
        return torch.load(mask_fn)

    def _get_mask_corners(self):
        corners_fn = os.path.join(self.corner_path, f'{int(100 * float(self.orig_img_scale))}_percent_fold{self.fold}.pt')
        if not os.path.exists(corners_fn):
            gen_mask_corners(self.mask_path, corners_fn, self.orig_img_scale, self.index_file_path,
                             fold=self.fold, view=self.view, label_type=self.label_type)
        return torch.load(corners_fn)

    def __init__(self, mask_path, corner_path, index_file_path, orig_img_scale=0.25, fold=0, view='KAPAP',
                 label_type='2class'):
        self.orig_img_scale = orig_img_scale
        self.index_file_path = index_file_path
        self.fold = fold
        self.view = view
        self.label_type = label_type
        self.mask_path = mask_path
        self.corner_path = corner_path


        # Generate pre-computation libraries for masks and corners
        Path(self.mask_path).mkdir(parents=True, exist_ok=True)
        Path(self.corner_path).mkdir(parents=True, exist_ok=True)

        # Get masks and corners
        self.masks = self._get_masks()
        self.corners = self._get_mask_corners()

    def __call__(self, sample):
        sample, p_id = sample
        T, R, B, L = self.corners[p_id]
        if L[1] < 0:  # can't have negative at the beginning of range
            L[1] = 0
        if T[0] < 0:  # can't have negative at the beginning of range
            T[0] = 0
        cropped_sample = sample[:, T[0]:B[0], L[1]:R[1]]
        return cropped_sample, p_id


class Identity():
    """
    Identity transformation
    """

    def __call__(self, sample):
        return sample


class ConvertToTensor():
    """
    Convert numpy array to Tensor
    """

    def __call__(self, sample):
        sample, p_id = sample
        if isinstance(sample, list):
            return torch.stack([torch.from_numpy(s).float().unsqueeze(0) for s in sample]), p_id
        else:
            return torch.from_numpy(sample).float().unsqueeze(0), p_id


class HistEq():
    """
    Perform Histogram Equalization
    """

    def __call__(self, sample):
        sample, p_id = sample
        if isinstance(sample, list):
            sample = [cv2.equalizeHist(s) for s in sample]
        else:
            sample = cv2.equalizeHist(sample)
        return sample, p_id


class RandResizeCrop():
    """
    Randomly resize image to given scale and crop it back to original scale
    """

    def __init__(self, scale):
        assert scale > 1, "Scale must be greater than 1 for RandResizeCrop"
        self.max_scale = scale

    def __call__(self, sample):
        H, W = sample.shape[-2], sample.shape[-1]
        rand_scale = torch.rand(1) * (self.max_scale - 1) + 1
        sample = transforms.functional.resize(sample, size=(int(rand_scale * H), int(rand_scale * W)))
        sample = transforms.functional.center_crop(sample, output_size=(H, W))
        return sample


class RandResizePad():
    """
    Randomly resize image to given scale and pad to arrive back at original scale
    """

    def __init__(self, scale, pad_noise=False, fill_val=FILL_VAL):
        assert scale < 1, "Scale must be greater than 1 for RandResizeCrop"
        self.pad_noise = pad_noise
        self.min_scale = scale
        self.fill_val = fill_val

    def _pad_noise(self, sample, pad_up_down, pad_left_right):
        background = torch.rand(sample.shape) + sample
        background[sample != 0] = 0
        sample[:, :pad_up_down, :] = background[:, :pad_up_down, :]
        sample[:, -pad_up_down:, :] = background[:, -pad_up_down:, :]
        sample[:, :, :pad_left_right] = background[:, :, :pad_left_right]
        sample[:, :, -pad_left_right:] = background[:, :, -pad_left_right:]
        return sample

    def __call__(self, sample):
        H, W = sample.shape[-2], sample.shape[-1]
        rand_scale = torch.rand(1) * (1 - self.min_scale) + self.min_scale
        sample = transforms.functional.resize(sample, size=(int(rand_scale * H), int(rand_scale * W)))
        new_H, new_W = sample.shape[-2], sample.shape[-1]
        pad_up_down, pad_left_right = (H - new_H) // 2, (W - new_W) // 2
        sample = transforms.functional.pad(sample, padding=(pad_left_right, pad_up_down), fill=self.fill_val)
        assert H - 2 <= sample.shape[-2] <= H + 2 and W - 2 <= sample.shape[
            -1] <= W + 2, f"Wrong dimension after padding, original was {(H, W)}, new is {sample.shape[-2:]}"
        if self.pad_noise:
            samples = self._pad_noise(sample, pad_up_down, pad_left_right)
        sample = transforms.functional.resize(sample, size=(H, W))
        return sample


class RandomResize():
    """
    Randomly scale and crop or scale and pad the image
    """

    def __init__(self, pad_noise=False, fill_val=FILL_VAL):
        self.transforms = [
            # RandResizeCrop(1.4),
            RandResizeCrop(1.2),
            # RandResizePad(0.6, pad_noise),
            RandResizePad(0.8, pad_noise, fill_val)
        ]

    def __call__(self, sample):
        rand_transform = random.randint(0, 1)
        return self.transforms[rand_transform](sample)


class Normalize():
    """
    Standardize input image
    """

    def __init__(self, max_val=255.):
        self.max_val = max_val

    def __call__(self, sample):
        sample, p_id = sample
        return sample.float() / self.max_val, p_id


class SaltPepperNoise():
    """
    Add Salt and Pepper noise on top of the image
    """

    def __init__(self, thresh=0.005):
        self.thresh = thresh

    def __call__(self, sample):
        noise = torch.rand(sample.shape)
        sample[noise < self.thresh] = 0
        sample[noise > 1 - self.thresh] = 1
        return sample


class RandomBrightnessAdjustment():
    """
    Randomly adjust brightness
    """

    def __call__(self, sample):
        rand_factor = torch.rand(1) * 0.7 + 0.5
        sample = F.adjust_brightness(sample, brightness_factor=rand_factor)
        return sample


class RandomGammaCorrection():
    """
    Do Gamma Correction with random gamma as described in
    https://en.wikipedia.org/wiki/Gamma_correction
    """

    def __call__(self, sample):
        rand_gamma = torch.rand(1) * 1.75 + 0.25
        sample = F.adjust_gamma(sample, gamma=rand_gamma)
        return sample


class RandomSharpness():
    """
    Randomly increase oder decrease image sharpness
    """

    def __call__(self, sample):
        if 0.5 < torch.rand(1):
            rand_factor = torch.rand(1) * 7 + 1
        else:
            rand_factor = torch.rand(1)
        sample = F.adjust_sharpness(sample, sharpness_factor=rand_factor)
        return sample


class RandomNoise():
    """
    Add Gaussian noise to the image
    """

    def __init__(self):
        # Standarddeviation of noise is 5 pixel intensities
        self.std = 5. / 255.

    def __call__(self, sample):
        return torch.clamp(sample + self.std * torch.randn_like(sample), min=0, max=1)


class GaussianSmoothing():
    """
    Do Gaussian filtering with low sigma to smooth out edges
    """

    def __init__(self, kernel_size=11, sigma=0.5):
        # Define Gaussian Kernel
        self.gaussian_kernel = self._generate_gaussian_kernel(kernel_size=kernel_size, sigma=sigma)

    def _generate_gaussian_kernel(self, kernel_size, sigma):
        """
        Code from https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
        """

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                              (2 * variance)
                          )
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Set pytorch convolution from gaussian kernel
        gaussian = torch.nn.Conv2d(1, 1, kernel_size, padding=kernel_size // 2, padding_mode='zeros', bias=False)
        gaussian.weight.requires_grad = False
        gaussian.weight[:, :] = gaussian_kernel
        return gaussian

    def __call__(self, sample):
        sample, p_id = sample
        # Slightly smooth out edges with Gaussian Kernel
        with torch.no_grad():
            if len(sample.shape) == 3:
                sample = self.gaussian_kernel(sample.unsqueeze(0)).squeeze(0)
            else:
                sample = self.gaussian_kernel(sample)
        return sample, p_id


class Resize():
    """
    Resize image to given size, -1 for original size
    """

    def __init__(self, size, return_pid=True):
        self.return_pid = return_pid
        if size == -1:
            self.transform = Identity()
        else:
            self.transform = transforms.Resize((size, size))

    def __call__(self, sample):
        sample, p_id = sample
        if self.return_pid:
            return self.transform(sample), p_id
        else:
            return self.transform(sample)


class AugmentSegMasks():
    def __init__(self):
        self.augments = [transforms.RandomRotation(degrees=15),
                         transforms.RandomAffine(degrees=(0, 0), translate=(0.1, 0.1)),
                         RandomResize(fill_val=0)]  # only black-background

    def __call__(self, sample):
        # Get sample and corresponding mask
        sample, p_id = sample
        for aug in self.augments:
            sample = aug(sample)
        return sample


class Augment():
    """
    Randomly perform a range of data augmentation
    including translation, rotation, scaling, salt & pepper noise,
    brightness adjustment, Gamma Correction, blurring and sharpening the image
    """

    def __init__(self, mask_path, index_file_path, orig_img_scale=0.5, size=-1, return_pid=False, fold=0, valid=False,
                 view='KAPAP', aug_type=2, label_type='2class'):
        self.return_pid = return_pid
        self.mask_path = mask_path
        Path(self.mask_path).mkdir(parents=True, exist_ok=True)

        # Get masks and corners and set other parameters
        self.orig_img_scale = orig_img_scale
        self.index_file_path = index_file_path
        self.size = size
        self.view = view
        self.type = aug_type
        self.label_type = label_type
        self.masks = {}
        for curr_view in view:  # change to views
            self.masks[curr_view] = self._get_masks(fold, curr_view)
        # self.masks = self._get_masks(fold)

        # self.pad = 12
        self.pad = 18

        # Define augmentation transforms
        self.intensity_transformations = [
            RandomSharpness(),
            RandomBrightnessAdjustment(),
            RandomGammaCorrection(),
            SaltPepperNoise()
        ]
        self.positional_transformations = [
            None,
            transforms.RandomAffine(0, translate=(0.1, 0.1), fill=FILL_VAL),
            RandomResize(),
        ]

        self.augments = [transforms.RandomRotation(degrees=15),
                         RandomResize(fill_val=0)]  # only black-background

    def _get_masks(self, fold, view):
        print('in _get_masks')
        #mask_fn = os.path.join(self.mask_path,
        #                       f'{self.size}_{int(100 * float(self.orig_img_scale))}_percent_fold{fold}.pt')
        # mask_fn = os.path.join(self.mask_path,  # view is already defined in the mask path!
        #                        f'{self.size}_{int(100 * float(self.orig_img_scale))}_percent_'
        #                        f'label{self.label_type}_fold{fold}.pt')
        mask_path = self.mask_path if view == 'KAPAP' else self.mask_path + f'_{view}'
        # mask_fn = get_mask_fn(self.mask_path, self.size, self.orig_img_scale, self.label_type, fold)
        mask_fn = get_mask_fn(mask_path, self.size, self.orig_img_scale, self.label_type, fold)
        print(mask_fn)
        if not os.path.exists(mask_fn):
            # utilities.generate_masks(self.size, self.orig_img_scale)
            # gen_masks(mask_fn, self.size, self.orig_img_scale, self.index_file_path, view=self.view)
            gen_masks(mask_fn, self.size, self.orig_img_scale, self.index_file_path, view=view)
        return torch.load(mask_fn)

    def _apply_background_noise(self, sample, mask):
        background = FILL_VAL * torch.ones(sample.shape)
        background = background.to(sample.device)
        if len(sample.shape) == 4 and len(mask.shape) < 4:
            sample[:, mask == 0] = sample[:, mask == 0] + background[:, mask == 0]
        else:
            sample[mask == 0] = sample[mask == 0] + background[mask == 0]
        return sample

    def _apply_mask(self, sample, mask):
        return mask * sample

    def _cut_border(self, sample, mask):
        H, W = sample.shape[-2], sample.shape[-1]
        sample = transforms.functional.resize(sample, size=(int(1.05 * H), int(1.05 * W)))
        sample = transforms.functional.center_crop(sample, output_size=(H, W))
        return sample * mask

    def _add_background_speckle_noise(self, sample):
        std = 0.3
        sample[sample == FILL_VAL] = sample[sample == FILL_VAL] + \
                                     std * sample[sample == FILL_VAL] * torch.randn_like(sample[sample == FILL_VAL])
        sample = torch.clamp(sample, min=0, max=1)
        return sample

    def _apply_positional_transforms(self, sample, mask, p=0.4):
        # Cut off black border around echo
        sample = self._cut_border(sample, mask)
        # Add background noise
        sample = self._apply_background_noise(sample, mask)

        # Define Random Rotation around top corner
        if self.positional_transformations[0] is None:
            rot_center = (sample.shape[1] // 2, 0)
            rand_rot = transforms.RandomRotation(15, fill=FILL_VAL, center=rot_center)
            self.positional_transformations[0] = rand_rot

        # Apply positional transformations
        for t in self.positional_transformations:
            # if 0.7 < torch.rand(1):
            if torch.rand(1) < p:  # 4 transforms, each with p %  chance
                sample = t(sample)
        return sample

    def _apply_positional_transforms_black(self, sample):
        if len(sample.shape) == 4:  # temporal, i.e. video
            l, c, h, w = sample.shape
            temp = np.zeros((l, c, h + 2 * self.pad, w + 2 * self.pad), dtype=np.float32)
            temp[:, :, self.pad:-self.pad, self.pad:-self.pad] = sample  # pylint: disable=E1130
        else:  # spatial, i.e. single frames
            c, h, w = sample.shape
            temp = np.zeros((c, h + 2 * self.pad, w + 2 * self.pad), dtype=np.float32)
            temp[:, self.pad:-self.pad, self.pad:-self.pad] = sample  # pylint: disable=E1130
        i, j = np.random.randint(0, 2 * self.pad, 2)
        if len(sample.shape) == 4:
            sample = temp[:, :, i:(i + h), j:(j + w)]
            sample = torch.stack([torch.from_numpy(s).float() for s in sample])  # to tensor
        else:
            sample = temp[:, i:(i + h), j:(j + w)]
            sample = torch.from_numpy(sample).float()
        for aug in self.augments:
            sample = aug(sample)
        return sample

    def __call__(self, sample):
        # Get sample and corresponding mask
        sample, p_id = sample
        p_id, view = p_id.split('_')
        p_id = p_id + view  # without the '_'
        # In augment type 1, 25 % of images don't get any augmentation
        if self.type == 1 and torch.rand(1) < 0.25:
            if self.return_pid:
                return sample, p_id
            return sample

        # In augment type 4, 10 % of images don't get any augmentation
        if (self.type == 4 or self.type == 6) and torch.rand(1) < 0.1:
            if self.return_pid:
                return sample, p_id
            return sample

        # mask = self.masks[p_id].unsqueeze(0)
        mask = self.masks[view][p_id].unsqueeze(0)
        mask = mask.to(sample.device)         # Try moving mask to gpu if available

        # Apply intensity transformations to 50 % of images
        for t in self.intensity_transformations:
            if torch.rand(1) < 0.5:
                sample = t(sample)

        # Apply positional transformations
        if self.type == 1:
            sample = self._apply_positional_transforms(sample, mask, p=0.5)
        elif self.type == 3:
            sample = self._apply_positional_transforms(sample, mask, p=0.4)
        elif self.type == 5:
            sample = self._apply_positional_transforms_black(sample)
        elif self.type == 6:
            if torch.rand(1) < 0.5:  # 50 % get positional transforms on black background
                sample = self._apply_positional_transforms_black(sample)
            else:
                if torch.rand(1) < 0.75:  # 75 % of 50% get also positional transforms, each one with 60% chance
                    sample = self._apply_positional_transforms(sample, mask, p=0.6)
                    pos_tf = True
                else:  # Even when not doing positional transforms, gray out background for some (25% of) frames
                    if torch.rand(1) < 0.25:
                        # Cut off black border around echo & add background noise
                        sample = self._cut_border(sample, mask)
                        sample = self._apply_background_noise(sample, mask)
        else:  # for augmentation type 2 and 4, not always apply positional transforms
            pos_tf = False
            if torch.rand(1) < 0.75:  # 75 % get also positional transforms, each one with 60% chance
                sample = self._apply_positional_transforms(sample, mask, p=0.6)
                pos_tf = True
            else:  # Even when not doing positional transforms, gray out background for some (25% of) frames
                if torch.rand(1) < 0.25:
                    # Cut off black border around echo & add background noise
                    sample = self._cut_border(sample, mask)
                    sample = self._apply_background_noise(sample, mask)

        # Retrieve original shape, in some cases
        if self.type == 3:  # Always retrieve original shape
            sample = self._apply_mask(sample, mask)
        # Augment type 4 retrieves original shape for 50 % of frames with positional transforms
        elif self.type == 4 and pos_tf and torch.rand(1) < 0.5:
            sample = self._apply_mask(sample, mask)

        # Add speckle noise to background (always do this - only has effects when background is still gray)
        sample = self._add_background_speckle_noise(sample)

        if self.type == 4 and torch.rand(1) < 0.5:  # Augment type 4 also performs random noise to 50 % of images
            rn = RandomNoise()
            sample = rn(sample)

        if self.return_pid:
            return sample, p_id
        return sample


def get_transforms(
        index_file_path,
        fold=0,
        valid=False,
        view='KAPAP',
        resize=256,
        noise=False,
        augment=2,
        with_pid=False,
        crop_to_corner=False,
        dataset_orig_img_scale=0.25,
        segm_mask_only=False,
        label_type='2class'
):
    """
    Compose a set of prespecified transformation using the torchvision transform compose class
    """
    subset = 'valid' if valid else 'train'
    view_set = '' if view == 'KAPAP' else f'_{view}'  # Only specify separately if not default
    # mask_path = os.path.expanduser(os.path.join('~', '.echo-net', 'masks', subset + view_set))
    mask_path = os.path.expanduser(os.path.join('~', '.echo-net', 'masks', subset))
    print(mask_path)
    corner_path = os.path.expanduser(os.path.join('~', '.echo-net', 'mask_corners', subset + view_set))
    max_val = 255.
    if segm_mask_only:
        segm_view = our_view_to_segm_view[view]
        max_val = np.max(list(segmentation_labels[segm_view].values()))
    return transforms.Compose(
        [
            HistEq() if not segm_mask_only else Identity(),
            ConvertToTensor(),
            Normalize(max_val=max_val),
            CropToCorners(mask_path, corner_path, index_file_path, orig_img_scale=dataset_orig_img_scale, fold=fold,
                          view=view, label_type=label_type) if crop_to_corner and not segm_mask_only else Identity(),
            Trim() if crop_to_corner and segm_mask_only else Identity(),
            Resize(resize, return_pid=(with_pid or augment)),
            Augment(mask_path, index_file_path, orig_img_scale=dataset_orig_img_scale, size=resize, return_pid=with_pid,
                    fold=fold, valid=valid, view=view, aug_type=augment, label_type=label_type) if augment != 0 and
                                                                                                   not segm_mask_only
            else Identity(),
            # ConvertToTensor(),  # TODO: REMOVE
            AugmentSegMasks() if augment != 0 and segm_mask_only else Identity(),
            RandomNoise() if noise and not segm_mask_only else Identity()
        ]
    )


def gen_masks(mask_fn, resize, orig_scale_fac, index_file_path, view='KAPAP'):
    # Get a mask for each patient
    print("Assembling echo masks.")
    if not os.path.exists(os.path.dirname(mask_fn)):
        os.mkdir(os.path.dirname(mask_fn))
    masks = {}
    p_ids = [str(id) + view for id in np.load(index_file_path)]
    results = []
    with mp.Pool(processes=16) as pool:
        for result in pool.map(partial(_gen_mask, index_file_path, resize, orig_scale_fac), p_ids):
            results.append(result)
    # Join results
    for tup in results:
        if tup is not None:
            p_id = tup[0]
            masks[p_id] = tup[1]
    torch.save(masks, mask_fn)


def _gen_mask(index_file_path, resize, orig_scale_fac, p_id):
    """
    Generate mask for p_id
    """
    transform = get_transforms(index_file_path, resize=resize, augment=False)
    cache_dir = os.path.expanduser('~/.heart_echo')  # TODO: Make parameter !
    curr_video_path = os.path.join(cache_dir, str(orig_scale_fac), str(p_id) + '.npy')
    if not os.path.exists(curr_video_path):  # Skip unavailable patients
        return
    print('CREATING MASK FOR', curr_video_path)
    frames = np.load(curr_video_path)  # load numpy video frames
    frames = [transform((frame, p_id)).unsqueeze(0) for frame in frames]
    # Extract mask from patient
    H, W = frames[0][0].shape[1:]
    # H, W = frames[0].shape
    vid = torch.cat(tuple(frame[0] for frame in frames), dim=0)
    mask = torch.std(vid, dim=0)
    mask[mask < 0.01] = 0

    # Remove pixels without surrounding pixels
    nonzero = torch.nonzero(mask)
    for idx in nonzero:
        if idx[0] + 1 < H and idx[1] + 1 < W and \
                idx[0] - 1 >= 0 and idx[1] - 1 >= W and \
                mask[idx[0] + 1, idx[1]] == 0 and \
                mask[idx[0] - 1, idx[1]] == 0 and \
                mask[idx[0], idx[1] + 1] == 0 and \
                mask[idx[0], idx[1] - 1] == 0:
            mask[idx[0], idx[1]] = 0


    # Augment mask with convex hull
    hull = ConvexHull(torch.nonzero(mask))
    hull_mask = torch.ones(mask.shape)
    for i in range(hull_mask.shape[0]):
        for j in range(hull_mask.shape[1]):
            point = np.array([i, j, 1])
            for eq in hull.equations:
                if eq.dot(point) > 0:
                    hull_mask[i, j] = 0

    return p_id, hull_mask


def gen_mask_corners(mask_path, corners_fn, orig_scale_fac, index_file_path, fold=0, view='KAPAP', label_type='2class'):
    # Assemble pre-computation paths
    mask_fn = get_mask_fn(mask_path, -1, orig_scale_fac, label_type, fold)
    # Load masks
    if not os.path.exists(mask_fn):
        gen_masks(mask_path, -1, orig_scale_fac, index_file_path, view=view)
    masks = torch.load(mask_fn)

    # Generate Corners
    corners = {}
    for p_id in masks:
        corners[p_id] = np.int32(get_arc_points_from_mask(masks[p_id]))
    torch.save(corners, corners_fn)


def get_arc_points_from_mask(mask):
    # Get min and max nonzero entries
    mask_entries = torch.nonzero(mask)
    min_y, min_x = torch.min(mask_entries, dim=0).values
    max_y, max_x = torch.max(mask_entries, dim=0).values

    # Get top point
    top_candidates = mask_entries[mask_entries[:, 0] == min_y]
    top_left = torch.tensor([min_y, torch.min(top_candidates, dim=0).values[1]])
    top_right = torch.tensor([min_y, torch.max(top_candidates, dim=0).values[1]])

    left_candidates = mask_entries[mask_entries[:, 1] == min_x]
    left_top = torch.tensor([torch.min(left_candidates, dim=0).values[0], min_x])
    left_bot = torch.tensor([torch.max(left_candidates, dim=0).values[0], min_x])

    right_candidates = mask_entries[mask_entries[:, 1] == max_x]
    right_top = torch.tensor([torch.min(right_candidates, dim=0).values[0], max_x])
    right_bot = torch.tensor([torch.max(right_candidates, dim=0).values[0], max_x])

    left_line = Line(top_left, left_top)
    right_line = Line(top_right, right_top)
    top = np.array(left_line.intersection(right_line)[0])

    # Compute bottom point
    bot = []
    bot_candidates = mask_entries[mask_entries[:, 0] == max_y]
    bot.append(torch.tensor([[max_y, torch.min(bot_candidates, dim=0).values[1]]]))
    bot.append(torch.tensor([[max_y, torch.max(bot_candidates, dim=0).values[1]]]))
    bot = np.concatenate(bot)
    bot = 0.5 * (bot[0] + bot[1])

    # Create arc circle
    arc = Circle(left_bot, bot, right_bot)

    # Compute left and right points
    left_int = arc.intersection(left_line)
    right_int = arc.intersection(right_line)

    # Always take point with highest first coordinate (lowest)
    if len(left_int) == 1 or left_int[0][0] > left_int[1][0]:
        left = left_int[0]
    else:
        left = left_int[1]
    if len(right_int) == 1 or right_int[0][0] > right_int[1][0]:
        right = right_int[0]
    else:
        right = right_int[1]
    left = np.array(left)
    right = np.array(right)

    corners = np.array([top, right, bot, left])
    return corners