import numpy as np
import os
import tensorflow as tf
from glob import glob
from tifffile import imread
import nibabel as nib
from natsort import natsorted
from stardist.models import StarDist3D
from tensorflow.keras.utils import Sequence
import argparse

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


class FileData(Sequence):
    def __init__(self, filenames, ftype='nii.gz'):
        """
        Parameters
        ----------
        filenames : list
            list of filepaths to images to load.
        ftype : str, optional
            Image file extension. One of {'nii.gz', 'nii', 'tiff', 'tif'}.
        """
        super().__init__() 
        self.filenames = filenames
        self.ftype = ftype

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        if self.ftype == 'nii.gz' or 'nii':
            x = nib.load(self.filenames[i]).get_fdata().astype(np.float32)
        elif self.ftype == 'tiff' or self.ftype == 'tif':
            x = imread(self.filenames[i]).astype(np.float32)
        else:
            raise NotImplementedError
        upper = np.percentile(x, 99.9)
        x = np.clip(x, 0, upper)
        x = (x - x.min()) / (x.max() - x.min())
        fname = os.path.basename(self.filenames[i])
        return x, fname


def load_images(image_folder, extension):
    img_paths = natsorted(
        glob(image_folder + '/*.{}'.format(extension))
    )
    imgs = FileData(img_paths, ftype=extension)
    return imgs


def load_model(model_name, model_folder):
    model = StarDist3D(None, name=model_name, basedir=model_folder)
    model.load_weights(name='weights_best.h5')
    model.trainable = False
    model.keras_model.trainable = False
    return model


def inference_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--image_folder',
        type=str,
        help='path containing images to segment. nii.gz or tiff imgs accepted.',
    )
    parser.add_argument(
        '--image_extension',
        type=str,
        default='nii.gz',
        help='nii.gz or tiff imgs accepted.',
    )
    parser.add_argument(
        '--model_folder',
        type=str,
        default='models',
        help='name of folder containing multiple model subfolders',
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='anystar-mix',
        help='name of model subfolder',
    )
    parser.add_argument(
        '--prob_thresh',
        type=float,
        default=0.5,
        help='Softmax detection threshold. Lower detects more and vice versa.',
    )
    parser.add_argument(
        '--nms_thresh',
        type=float,
        default=0.3,
        help='Non-max suppression threshold. Lower value suppresses more.',
    )
    parser.add_argument(
        '--scale',
        type=float,
        nargs=3,
        default=[1., 1., 1.],
        help='Resizing ratios per dimension for inference',
    )
    parser.add_argument(
        '--n_tiles',
        type=int,
        nargs=3,
        default=None,
        help='N. tiles/dim for sliding window. Default: let StarDist decide',
    )
    return parser.parse_args()


if __name__ == '__main__':
    np.random.seed(12345)

    # Get inference arguments:
    args = inference_args()

    # Setup and load pretrained model:
    model = load_model(args.model_name, args.model_folder)

    # Load images to segment:
    data = load_images(args.image_folder, args.image_extension)

    # Create output directory to store segmentations:
    output_dir = 'outputs/{}_{}'.format(
        args.model_name,
        os.path.basename(os.path.normpath(args.image_folder)),
    ) 
    os.makedirs(output_dir, exist_ok=True)

    # Segment all images:
    for i in range(len(data)):
        image, name = data[i]
        # Segment image:
        labels, details = model.predict_instances(
            image,
            prob_thresh=args.prob_thresh,
            nms_thresh=args.nms_thresh,
            n_tiles=args.n_tiles,
            scale=args.scale,
        )
        # Save segmentations:
        nib.save(
            nib.Nifti1Image(labels, np.eye(4)),
            output_dir + '/{}'.format(name)
        )
        