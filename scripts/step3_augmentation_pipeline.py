import numpy as np
from monai.transforms import (
    RandRotate90d,
    ScaleIntensityd,
    Compose,
    LoadImaged,
    RandAxisFlipd,
    RandGaussianNoised,
    RandBiasFieldd,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandGaussianSharpend,
    RandHistogramShiftd,
    RandSpatialCropd,
    Rand3DElasticd,
    EnsureTyped,
    RandGibbsNoised,
    EnsureChannelFirstd,
    RandKSpaceSpikeNoised,
    RandRicianNoised,
    RandCoarseDropoutd,
    RandZoomd,
    RandAffined,
)

import glob, os, natsort
import torch
from torch.utils.data import DataLoader
from skimage.util import random_noise
from skimage.measure import label as unique_label

import monai
from monai.data import list_data_collate

import argparse


def worker_init_fn(worker_id):
    """Used because MONAI / pytorch have some odd threading issue."""
    worker_info = torch.utils.data.get_worker_info()
    try:
        worker_info.dataset.transform.set_random_state(worker_info.seed % (2 ** 32))
    except AttributeError:
        pass


def get_transforms():
    """
    MONAI Composed transforms used for the A(.) function from the paper.
    """
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            ScaleIntensityd(keys="image"),
            RandSpatialCropd(
                keys=["image", "label"],
                roi_size=[64, 64, 64],
            ),
            RandZoomd(
                prob=1.0,
                keys=["image", "label"],
                min_zoom=0.4,
                max_zoom=1.4,
                mode=["trilinear", "nearest"],
                padding_mode="reflect",
            ),
            RandAffined(
                mode=("bilinear", "nearest"),
                keys=["image", "label"],
                rotate_range=(np.pi/4, np.pi/4, np.pi/4),
                translate_range=(16, 16, 16),
                prob=1.0,
                padding_mode="reflection",
            ),
            RandBiasFieldd(keys=["image"], prob=1.0, coeff_range=(0.0, 0.1)),
            RandGaussianNoised(keys=["image"], prob=0.25),
            RandKSpaceSpikeNoised(keys=["image"], prob=0.2),
            RandAdjustContrastd(keys=["image"], prob=0.8),
            RandGaussianSmoothd(
                keys=["image"],
                prob=0.8,
                sigma_x=(0.0, 0.333),
                sigma_y=(0.0, 0.333),
                sigma_z=(0.0, 0.333),
            ),
            RandRicianNoised(keys=["image"], prob=0.2, std=0.05),
            RandGibbsNoised(keys=["image"], prob=0.5, alpha=(0.0, 1.0)),
            RandGaussianSharpend(keys=["image"], prob=0.25),
            RandHistogramShiftd(keys=["image"], prob=0.1),
            RandAxisFlipd(keys=["image", "label"], prob=1.0),
            RandRotate90d(keys=["image", "label"], prob=1.0),
            Rand3DElasticd(
                keys=["image", "label"],
                sigma_range=(2, 5),
                magnitude_range=(8, 20),
                mode=("bilinear", "nearest"),
                scale_range=(0.3, 0.3, 0.3),
                spatial_size=(64, 64, 64),
                shear_range=(0.5, 0.5, 0.5),
                prob=1.0,
                padding_mode='zeros',
            ),
            ScaleIntensityd(keys="image"),
            RandCoarseDropoutd(
                prob=0.2,
                keys=["image", "label"],
                fill_value=0,
                holes=20,
                spatial_size=12,
            ),
        ]
    )
    return train_transforms


if __name__ == '__main__':
    np.random.seed(12345)

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_offline_augmentations', type=int, default=100)
    args = parser.parse_args()
    n_offline_augmentations = args.n_offline_augmentations

    train_transforms = get_transforms()

    baseimg = '../generative_model/outputs/training_images/'
    baseseg = '../generative_model/outputs/training_labels/'    
    baseimagepaths = sorted(glob.glob('../generative_model/outputs/gmm_perlin_images/*'))
    basesegpaths = sorted(glob.glob('../generative_model/outputs/gmm_perlin_labels/*'))
    
    assert len(baseimagepaths) == len(basesegpaths)
    
    for i in range(len(baseimagepaths)):
        print('Running {} which is {}/{}'.format(baseimagepaths[i], i+1, len(baseimagepaths)))
    
        images = natsort.natsorted(glob.glob(baseimagepaths[i] + '/*.nii.gz'))
        segs = natsort.natsorted(glob.glob(basesegpaths[i] + '/*.nii.gz'))
        train_files = [{"image": img, "label": seg} for img, seg in zip(images, segs)]
        
        # create a training data loader
        train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms)
        train_loader = DataLoader(
            train_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=list_data_collate,
            pin_memory=False,
            worker_init_fn=worker_init_fn
        )
    
        imagelocs = '{}/{}/'.format(baseimg, os.path.basename(baseimagepaths[i]))
        seglocs = '{}//{}/'.format(baseseg, os.path.basename(basesegpaths[i]))
        os.makedirs(imagelocs, exist_ok=True)
        os.makedirs(seglocs, exist_ok=True)

        # augment each gmmXperlin sampled image n_offline_augmentations times:
        for j in range(n_offline_augmentations):
            for batch_data in train_loader:
                inputs, labels = batch_data["image"], batch_data["label"]
                if inputs.squeeze().data.sum() == 0:
                    continue
                nonzero = (inputs.squeeze().data > 0).astype(np.float32)
                if np.random.uniform(0, 1) < 0.2:
                    noise_choice = np.random.choice(['gaussian', 'poisson', 'speckle'])
                    img = random_noise(inputs.squeeze().data, mode=noise_choice)
                    img = img * nonzero
                else:
                    img = 1.0 * inputs.squeeze().data
                monai.data.write_nifti(
                    img,
                    imagelocs + os.path.basename(inputs.meta['filename_or_obj'][0])[:-7] + '_v{}.nii.gz'.format(j),
                    affine=np.eye(4),
                )
                separate_labels = unique_label(labels.squeeze().data).astype(np.uint16)
                monai.data.write_nifti(
                    separate_labels,
                    seglocs + os.path.basename(labels.meta['filename_or_obj'][0])[:-7] + '_v{}.nii.gz'.format(j),
                    affine=np.eye(4),
                )

