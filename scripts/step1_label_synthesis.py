"""
We gratefully acknowledge Martin Weigert for providing a reference OpenCL
implementation of a nucleus generator upon which this python script is based.
"""

import numpy as np
import os
import nibabel as nib
import argparse

from itertools import product
from perlin_numpy import generate_perlin_noise_3d


def calculate_distance(center_coords, grid_size=128):
    """
    Parameters
    ----------
    center_coords : np.array
        (N_spheres x 3) matrix containing coordinates of sphere centers.
    grid_size : int, optional
        Size along 1 dim of image. The default is 128.

    Returns
    -------
    distances : np.array
        (N_voxels X N_spheres) distance matrix.
    """
    # Assign n_vox x n_spheres array:
    distances = np.zeros((grid_size**3, len(center_coords)))

    # list of all voxel indices:
    vox_coord = list(
        product(range(grid_size), range(grid_size), range(grid_size)),
    )

    # Calc. distance between each voxel and each sphere, loop through voxels:
    # There should be a much faster way using vectorization.
    for i in range(distances.shape[0]):
        distances[i] = np.sqrt(
            np.sum((np.array(vox_coord[i]) - center_coords)**2, axis=1)
        )
    
    return distances


def initial_label_generator(grid_size=128, r_mean=12):
    """
    Parameters
    ----------
    grid_size : int, optional
        Size along 1 dim of image. The default is 128.
    r_mean : int, optional
        Average sphere radius in voxels. The default is 12.

    Returns
    -------
    labelmap : np.array
        (grid_size x grid_size x grid_size) initial labels for synthesis.
    """

    Nz, Ny, Nx = (grid_size,) * 3
    
    x = np.arange(0, grid_size, 2*r_mean)[1:-1]
    Z, Y, X = np.meshgrid(x, x, x, indexing='ij')  # center coordinates
    
    # Randomly translate centers:
    points = np.stack(
        [Z.flatten(), Y.flatten(), X.flatten()]
    ).T # center coordinates flattened
    points = (points).astype(np.float32) 
    points_perturbed = points + .5*r_mean*np.random.uniform(-1,1,points.shape)
    
    # Randomly drop between 0--33% of spheres:
    ind = np.arange(len(points_perturbed))  # index of individual point
    np.random.shuffle(ind)  # randomly shuffle indices
    
    ind_keep = ind[:np.random.randint(2*len(ind)//3,len(ind))]  # drop indices
    points_perturbed_kept = points_perturbed[ind_keep]  # drop spheres
    
    # Randomly scale radii:
    rads = r_mean * np.random.uniform(.6, 1.2, len(points))  # randomly scale
    rads_kept = rads[ind_keep]  # randomly drop radii

    # Compute n_vox x n_spheres matrix:
    dist_mtx = calculate_distance(points_perturbed_kept)

    # Sample perlin noise:
    noise_sample = generate_perlin_noise_3d((Nz, Ny, Nx), res=(8, 8, 8))

    # Corrupt distance matrix:
    corr_dist_mtx = (
        dist_mtx + 0.9 * r_mean * noise_sample.flatten()[:, np.newaxis]
    )

    # Label assignment:
    labelmap = np.zeros(grid_size**3, dtype=np.uint16)  # initialize
    for j in range(dist_mtx.shape[0]): 
        finder = np.where(corr_dist_mtx[j, :] < rads_kept)[0]
        if len(finder) > 0:
            # in case of match with more than label, assign to closest:
            value = finder[np.argmin(corr_dist_mtx[j, finder])]
            labelmap[j] = value + 1

    labelmap = np.reshape(labelmap, (grid_size, grid_size, grid_size))
    return labelmap


if __name__ == '__main__':
    np.random.seed(12345)

    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--grid_size',
        type=int,
        default=128,
        help='side-length in voxels of the synthesized volume',
    )
    parser.add_argument(
        '--r_mean',  # one or both r_max & r_min have to be None to use this
        type=int,
        default=12,
        help='average radius in voxels of initial spheres',
    )
    parser.add_argument(
        '--r_max',
        type=int,
        default=None,
        help='Used if radius randomized. Specify min sphere radius in voxels',
    )
    parser.add_argument(
        '--r_min',
        type=int,
        default=None,
        help='Used if radius randomized. Specify min sphere radius in voxels',
    )
    parser.add_argument(
        '--n_images', 
        type=int,
        default=27,
        help='number of label maps to synthesize',
    )

    args = parser.parse_args()
    for i in range(args.n_images):
        print("Generating label {}/{}".format(i + 1, args.n_images))
        if args.r_min is None or args.r_max is None:
            label = initial_label_generator(args.grid_size, args.r_mean)
        else:
            radius = np.random.randint(args.r_min, args.r_max + 1)
            label = initial_label_generator(args.grid_size, radius)
        os.makedirs(
            "../generative_model/outputs/initial_labels/", exist_ok=True
        )
        nib.save(
            nib.Nifti1Image(label, np.eye(4)),
            "../generative_model/outputs/initial_labels/stack_{:04d}.nii.gz".format(i),
        )

