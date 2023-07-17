import numpy as np
import argparse
import nibabel as nib
import tensorflow as tf
import glob, os, natsort
from scipy import ndimage as ndi

from skimage.transform import resize
from skimage.measure import label as unique_label

import sys
sys.path.append('../')
from utils.synthseg_utils import draw_value_from_distribution, SampleConditionalGMM

import voxelmorph as vxm
import neurite as ne


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n_imgs',
        type=int,
        default=300,
        help='number of GMM images to sample from each synthesized label map'
    )
    args = parser.parse_args()
    nimgs = args.n_imgs

    segpath_base = "../generative_model/outputs/initial_labels/"  # step 1 initial synth labels
    imgpath_base = "../generative_model/outputs/gmm_perlin_images/"  # location for step2 imgs
    labpath_base = "../generative_model/outputs/gmm_perlin_labels/"  # location for step2 labs
    segs = natsort.natsorted(glob.glob(segpath_base + "/*.nii.gz"))

    for i in range(len(segs)):
        print('image {}/{}'.format(i + 1, len(segs)))
    
        # For each initial label map, create folders to store synth images:
        os.makedirs(
            imgpath_base + '/{}'.format(os.path.basename(segs[i])[:-7]),
            exist_ok=True,
        )
        os.makedirs(
            labpath_base + '/{}'.format(os.path.basename(segs[i])[:-7]),
            exist_ok=True,
        )
        complete_labels = nib.load(segs[i]).get_fdata()
        
        # for each label map, generate initial gmmXperlin images:
        for j in range(nimgs):
            # First, we pad and resize back in order to simulate varying object
            # densities.
    
            # Mode with which to pad initial label maps:
            # Constant just zero pads, reflect adds more instances
            randmode = np.random.choice(['constant', 'reflect'])
            
            # Amount to pad initial label maps along each axis:
            randpad_x = np.random.choice([0, 32, 64, 96])
            randpad_y = np.random.choice([0, 32, 64, 96])
            randpad_z = np.random.choice([0, 32, 64, 96])
    
            # Pad:
            current_labels = np.pad(
                complete_labels,
                [
                    [randpad_x, randpad_x],
                    [randpad_y, randpad_y],
                    [randpad_z, randpad_z]
                ],
                mode=randmode,
            )
            
            # Resize:
            current_labels = resize(
                current_labels,
                (128, 128, 128),
                preserve_range=True,
                anti_aliasing=False,
                order=0,
            )
            # Make sure each label is unique:
            current_labels = unique_label(current_labels).astype(np.uint16)
    
            # Second, we begin the GMM procedure.
            # Sample means and standard deviations for each object:
            means = draw_value_from_distribution(
                None,
                len(np.unique(current_labels)),
                'uniform',
                125,
                100,
                positive_only=True,
            )
            stds = draw_value_from_distribution(
                None, 
                len(np.unique(current_labels)),
                'uniform',
                15,
                10,
                positive_only=True,
            )
    
            # Background processing. This generates 'AS-Mix'.
            backgnd_mode = np.random.choice(['plain', 'rand', 'perlin'])
            if backgnd_mode == 'plain':
                min_mean = means.min() * np.random.rand(1)
                means[0] = 1.0 * min_mean
                stds[0] = np.random.uniform(0., 5., 1)
            elif backgnd_mode == 'perlin':
                # Inspired by the sm-shapes generative model from
                # https://martinos.org/malte/synthmorph/
                n_texture_labels = np.random.randint(1, 21)
                idx_texture_labels = np.arange(0, n_texture_labels, 1)
                im = ne.utils.augment.draw_perlin(
                    out_shape=(128, 128, 128, n_texture_labels),
                    scales=(32, 64), max_std=1,
                )
                try:
                    warp = ne.utils.augment.draw_perlin(
                        out_shape=(128, 128, 128, n_texture_labels, 3),
                        scales=(16, 32, 64), max_std=16,
                    )
                except:
                    continue
        
                # Transform and create background label map.
                im = vxm.utils.transform(im, warp)
                background_struct = np.uint8(tf.argmax(im, axis=-1))
    
                # Background moments for GMM:
                background_means = draw_value_from_distribution(
                    None, len(np.unique(idx_texture_labels)), 'uniform',
                    125, 100, positive_only=True,
                )
                background_stds = draw_value_from_distribution(
                    None, len(np.unique(idx_texture_labels)), 'uniform',
                    15, 10, positive_only=True,
                )
    
            # Sample perlin noise for cell texture here
            randperl = ne.utils.augment.draw_perlin(
                out_shape=(128, 128, 128, 1),
                scales=(2, 4, 8, 16, 32),
                max_std=5.,
            )[...,  0].numpy()
            randperl = (
                (randperl - randperl.min()) / (randperl.max() - randperl.min())
            )
    
            # Create foreground:
            synthlayer = SampleConditionalGMM(np.unique(current_labels))
            synthimage = synthlayer(
                [
                    tf.convert_to_tensor(
                        current_labels[np.newaxis, ..., np.newaxis],
                        dtype=tf.float32,
                    ),
                    tf.convert_to_tensor(
                        means[tf.newaxis, ..., tf.newaxis],
                        dtype=tf.float32,
                    ),
                    tf.convert_to_tensor(
                        stds[tf.newaxis, ..., tf.newaxis],
                        dtype=tf.float32,
                    ),
                ]
            )[0, ..., 0].numpy()

            # Use multiplicative Perlin noise on foreground:
            synthimage[current_labels>0] = (
                synthimage[current_labels>0] * randperl[current_labels>0]
            )
            del synthlayer
    
            # Create background:
            if backgnd_mode == 'plain' or backgnd_mode == 'rand':
                synthimage[current_labels==0] = (
                    synthimage[current_labels==0] *
                    np.mean(randperl[current_labels==0])
                )
            elif backgnd_mode == 'perlin':
                synthlayer = SampleConditionalGMM(idx_texture_labels)
                synthbackground = synthlayer(
                    [
                        tf.convert_to_tensor(
                            background_struct[np.newaxis, ..., np.newaxis],
                            dtype=tf.float32,
                        ),
                        tf.convert_to_tensor(
                            background_means[tf.newaxis, ..., tf.newaxis],
                            dtype=tf.float32,
                        ),
                        tf.convert_to_tensor(
                            background_stds[tf.newaxis, ..., tf.newaxis],
                            dtype=tf.float32,
                        ),
                    ]
                )[0, ..., 0].numpy()
                synthimage[current_labels==0] = (
                    synthbackground[current_labels==0] *
                    np.mean(randperl[current_labels==0])
                )
                del synthlayer
        
            nib.save(
                nib.Nifti1Image(synthimage, affine=np.eye(4)),
                imgpath_base + '/{}/{}_{:04}.nii.gz'.format(
                    os.path.basename(segs[i])[:-7],
                    os.path.basename(segs[i])[:-7],
                    j,
                ),
            )
            nib.save(
                nib.Nifti1Image(current_labels, affine=np.eye(4)),
                labpath_base + '/{}/{}_{:04}.nii.gz'.format(
                    os.path.basename(segs[i])[:-7],
                    os.path.basename(segs[i])[:-7],
                    j,
                ),
            )
