"""
Taken and modified from:
https://github.com/stardist/stardist/blob/master/examples/3D/2_training.ipynb
"""

import numpy as np
import nibabel as nib

from tensorflow.keras.utils import Sequence

class FileData(Sequence):
    def __init__(self, filenames, label_mode=False):
        """
        Sequence data structure for StarDist data loader.

        Parameters
        ----------
        filenames : list
            List of file paths pointing to the training image or label map.
            Assumes storage in nifti file formats.
        label_mode : bool
            Whether `filenames` contains paths pointing to images or labels.
            The default is False.

        """
        super().__init__() 
        self._filenames = filenames
        self.label_mode = label_mode

    def __len__(self):
        return len(self._filenames)

    def __getitem__(self, i):
        if self.label_mode is False:
            x = nib.load(self._filenames[i]).get_fdata()
        else:
            # If label map, use uint16
            x = nib.load(self._filenames[i]).get_fdata().astype(np.uint16)
        return x


def random_fliprot(img, mask, axis=None):
    """
    Augment on-the-fly with flips and rotations. Other augmentations done
    offline for the proposed model for speed.

    Parameters
    ----------
    img : np.array
        Image to augment.
    mask : TYPE
        Segmentation mask associated with `img`.
    axis : tuple, optional
        Axis along which to augment.

    Returns
    -------
    img : np.array
        Augmented image.
    mask : np.array
        Augmented segmentation mask.

    """
    if axis is None:
        axis = tuple(range(mask.ndim))
    axis = tuple(axis)
            
    assert img.ndim>=mask.ndim

    perm = tuple(np.random.permutation(axis))
    transpose_axis = np.arange(mask.ndim)

    for a, p in zip(axis, perm):
        transpose_axis[a] = p

    transpose_axis = tuple(transpose_axis)
    img = img.transpose(transpose_axis + tuple(range(mask.ndim, img.ndim))) 
    mask = mask.transpose(transpose_axis) 
    for ax in axis: 
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask


def augmenter(img, seg_label):
    img, seg_label = random_fliprot(img, seg_label, axis=None)
    return img, seg_label

