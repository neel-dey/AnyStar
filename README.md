# AnyStar

![Sample results on 5 datasets](https://www.neeldey.com/files/arxiv23_anystar_results.png)

AnyStar is a zero-shot 3D instance segmentation framework trained on purely
synthetic data. It is meant to segment star-convex (e.g. nuclei and nodules)
instances in 3D bio-microscopy and radiology. It is generally invariant to the
appearance (blur, noise, intensity, contrast) and environment of the instance
and requires no retraining or adaptation for new datasets.

This repository contains:
- Scripts (in `./scripts/`) to generate offline samples from the AnyStar-mix
generative model. While this can be run online with training, it significantly
CPU-bottlenecks training. Instead we sample a ~million samples offline first
and then use further fast augmentations during training.
- A training script `./train.py` to train a 3D [StarDist](https://github.com/stardist)
network on the synthesized data.

## Citation
```
@misc{dey2023anystar,
      title={AnyStar: Domain randomized universal star-convex 3D instance segmentation}, 
      author={Neel Dey and S. Mazdak Abulnaga and Benjamin Billot and Esra Abaci Turk and P. Ellen Grant and Adrian V. Dalca and Polina Golland},
      year={2023},
      eprint={2307.07044},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Roadmap
- [x] Add data generation and training code.
- [ ] Upload weights for the current paper model.
- [ ] Add example inference notebook.
- [ ] Add best practices doc.
- [ ] Upload improved weights.

## Set up dependencies
We create two separate tensorflow and pytorch conda environments for data generation
because the various VoxelMorph / MONAI / CSBDeep / etc. repos dont play well together.
We then have a separate training environment.

```bash
# Create environment for initial label and appearance synthesis:
conda create --name datagen_initial python=3.8
conda activate datagen_initial
pip install tensorflow==2.6 voxelmorph natsort keras==2.6
pip install --upgrade "protobuf<=3.20.1"
pip install git+https://github.com/pvigier/perlin-numpy
conda deactivate

# Create environment for offline augmentation pipeline:
conda create --name datagen_augment python=3.9.15
conda activate datagen_augment
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install monai==1.1.0 natsort scikit-image nibabel tqdm
conda deactivate

# Create environment for StarDist training:
conda env create -f https://raw.githubusercontent.com/CSBDeep/CSBDeep/master/extras/environment-gpu-py3.8-tf2.4.yml
conda activate csbdeep
pip install stardist
pip install nibabel
conda install -c conda-forge cudatoolkit-dev
conda install -c conda-forge libstdcxx-ng
pip install edt
conda deactivate
```

## Synthesize offline samples
![Sample synthesized training samples](https://www.neeldey.com/files/arxiv23_anystar_samples.png)

NOTE: Running the data synthesizer with the default settings will use a few
terabytes of storage. You may want to reduce `--n_stacks` in step1,
`--n_imgs` in step2, and `--n_offline_augmentations` in step3.

```bash
conda activate datagen_initial
cd ./scripts/
python step1_label_synthesis.py
python step2_GMM_Perlin_image_synthesis.py
conda deactivate

conda activate datagen_augment
python step3_augmentation_pipeline.py
conda deactivate
```

## Train
Sample training run, assuming that data is in `./generative_model/outputs/`.
```bash
conda activate csbdeep
python train.py --epochs 180 --steps 1000 --name sample_run
```

Full CLI:
```bash
usage: train.py [-h] [--epochs EPOCHS] [--steps STEPS]
                [--batch_size BATCH_SIZE] [--parent_dir PARENT_DIR]
                [--dataset_ims DATASET_IMS] [--dataset_labs DATASET_LABS]
                [--name NAME] [--lr LR] [--losswt_1 LOSSWT_1]
                [--losswt_2 LOSSWT_2] [--nrays NRAYS]
                [--val_samples VAL_SAMPLES] [--rng_seed RNG_SEED]

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of training epochs
  --steps STEPS         iterations per epoch
  --batch_size BATCH_SIZE
                        Batch size per iteration
  --parent_dir PARENT_DIR
                        path to folder containing images and labels subfolders
  --dataset_ims DATASET_IMS
                        base name of training image folder
  --dataset_labs DATASET_LABS
                        base name of training label folder
  --name NAME           folder name to store run results
  --lr LR               Adam learning rate
  --losswt_1 LOSSWT_1   centerness loss weight
  --losswt_2 LOSSWT_2   distance map loss weight
  --nrays NRAYS         number of 3D rays for polyhedra
  --val_samples VAL_SAMPLES
                        number of synthetic samples to use for early stopping
  --rng_seed RNG_SEED   NumPy RNG seed
```

## Acknowledgements
We make extensive use of the [StarDist](https://github.com/stardist) repository
for several training and helper functions and thank its authors.
