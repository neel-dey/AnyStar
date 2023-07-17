import argparse


def training_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--epochs', type=int, default=600, help='number of training epochs'
    )
    parser.add_argument(
        '--steps', type=int, default=300, help='iterations per epoch'
    )
    parser.add_argument(
        '--batch_size', type=int, default=2, help='Batch size per iteration'
    )
    parser.add_argument(
        '--parent_dir',
        type=str,
        default='./generative_model/outputs/',
        help='path to folder containing images and labels subfolders'
    )
    parser.add_argument(
        '--dataset_ims', type=str, help='base name of training image folder'
    )
    parser.add_argument(
        '--dataset_labs', type=str, help='base name of training label folder'
    )
    parser.add_argument(
        '--name', type=str, help='folder name to store run results'
    )
    parser.add_argument(
        '--lr', type=float, default=2e-4, help='Adam learning rate'
    )
    parser.add_argument(
        '--losswt_1', type=float, default=1, help='centerness loss weight'
    )
    parser.add_argument(
        '--losswt_2', type=float, default=0.2, help='distance map loss weight'
    )
    parser.add_argument(
        '--nrays', type=int, default=96, help='number of 3D rays for polyhedra'
    )
    parser.add_argument(
        '--val_samples',
        type=int,
        default=100,
        help='number of synthetic samples to use for early stopping',
    )
    parser.add_argument(
        '--rng_seed', type=int, default=12345, help='NumPy RNG seed'
    )

    return parser.parse_args()