# Peekaboo

**Concordia University**

Hasib Zunair, A. Ben Hamza

[[`Paper`](link)] [[`Project`](link)] [[`Demo`](#4-demo)] [[`BibTeX`](#5-citation)]

This is official code for our **BMVC 2024 paper**:<br>
[PEEKABOO: Hiding Parts of an Image for Unsupervised Object Localization](Link)
<br>

![MSL Design](./media/figure.jpg)

We propose a method for unsupervised object localization by learning context-based representations. This is done at both the pixel-level by making predictions on masked images and at shape-level by matching the predictions of the masked input to the unmasked one.

## 1. Specification of dependencies

This code requires Python 3.8 and CUDA 11.2. Create and activate the following conda envrionment.

```
conda update conda
conda env create -f environment.yml
conda activate peekaboo
```

Or, you can also create a fresh environment and install the project requirements inside that environment by:

```
# create fresh env
conda create -n peekaboo python=3.8     
conda activate peekaboo
# Example of pytorch installation
pip install torch===1.8.1 torchvision==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install pycocotools
# install reqs
pip install -r requirements.txt
```

And then, install [DINO](https://arxiv.org/pdf/2104.14294.pdf):

```bash
git clone https://github.com/facebookresearch/dino.git
cd dino; 
touch __init__.py
echo -e "import sys\nfrom os.path import dirname, join\nsys.path.insert(0, join(dirname(__file__), '.'))" >> __init__.py; cd ../;
```

## 2a. Training code

### Dataset details

We train on [DUTS-TR](http://saliencydetection.net/duts/) dataset that should be downloaded and put in the directory `datasets_local`.

We evaluate on two tasks: unsupervised saliency detection and single object discovery.

#### Unsupervised Saliency Detection

We use the following datasets:

- [DUT-OMRON](http://saliencydetection.net/dut-omron/): `--dataset-eval DUT-OMRON`
- [DUTS-TEST](http://saliencydetection.net/duts/): `--dataset-eval DUTS-TEST`
- [ECSSD](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html): `--dataset-eval ECSSD`.

Download and keep them in `datasets_local` and provide the dataset directory using the argument `--dataset-dir`.

#### Single Object Discovery

In order to evaluate on the single object discovery task, we follow the framework used in [LOST](https://github.com/valeoai/LOST). Download the datasets and put them in the folder `datasets_local`.

- [VOC07](http://host.robots.ox.ac.uk/pascal/VOC/): `--dataset-eval VOC07`
- [VOC12](http://host.robots.ox.ac.uk/pascal/VOC/): `--dataset-eval VOC12`
- [COCO20k](https://cocodataset.org/#home): `--dataset-eval COCO20k`

## DUTS-TR training

```bash
export DATASET_DIR=datasets_local # Root directory of all datasets, both training and evaluation

python train.py --exp-name peekaboo --dataset-dir $DATASET_DIR
```

See tensorboard logs by running: `tensorboard --logdir=outputs`.

## 2b. Evaluation code

After training, the model checkpoint and logs are available in `outputs` folder.

```bash
export MODEL="outputs/peekaboo-DUTS-TR-vit_small8/decoder_weights_niter500.pt"
```

### Unsupervised saliency detection eval

```bash
source evaluate_saliency.sh $MODEL $DATASET_DIR single
source evaluate_saliency.sh $MODEL $DATASET_DIR multi
```

### Single object discovery eval

```bash
source evaluate_uod.sh $MODEL $DATASET_DIR
```

All experiments are conducted on a single NVIDIA 3080Ti GPU. For additional implementation details and results, please refer to the supplementary materials section in the paper.

## 3. Pre-trained models

We provide pretrained models on [GitHub Releases](TBA) for reproducibility.

|Dataset      | Backbone  |   mAP (%)  |   Download   |
|  ---------- | -------   |  ------ |  --------   |
| VOC2007 | MSL-C  | 86.4 | [download](https://github.com/hasibzunair/msl-recognition/releases/download/v1.0-models/msl_c_voc.pth) |
| COCO2014 | MSL-C | 96.1 | [download](https://github.com/hasibzunair/msl-recognition/releases/download/v1.0-models/msl_c_coco.pth) |
| Wider-Attribute | MSL-V | 90.6 | [download](https://github.com/hasibzunair/msl-recognition/releases/download/v1.0-models/msl_v_wider.pth) |

## 4. Demo

We provide prediction demos of our models. The following applies and visualizes our method on a single image.

```bash
# infer on one images
python demo.py --img-path ./datasets_local/ECSSD/images/0009.jpg
```

## 5. Citation

```bibtex
 @inproceedings{zunair2024peekaboo,
    title={PEEKABOO: Hiding Parts of an Image for Unsupervised Object Localization},
    author={Zunair, Hasib and Hamza, A Ben},
    booktitle={Proc. British Machine Vision Conference},
    year={2024}
  }
```

## Project Notes

<details><summary>Click to view</summary>
<br>

**[Mar 18, 2024]** Infer on image folders.

```python
# infer on folder of images
python visualize_outputs.py --model-weights outputs/msl_a1.5_b1_g1_reg4-MSL-DUTS-TR-vit_small8/decoder_weights_niter500.pt --img-folder ./datasets_local/DUTS-TR/DUTS-TR-Image/ --output-dir outputs/visualizations/msl_masks
```

**[Nov 10, 2023]** Reproduced FOUND results.

**[Nov 10, 2023]** Added project notes section.

</details>

## Acknowledgements

This repository was built on top of [FOUND](https://github.com/valeoai/FOUND), [SelfMask](https://github.com/NoelShin/selfmask), [TokenCut](https://github.com/YangtaoWANG95/TokenCut) and [LOST](https://github.com/valeoai/LOST). Consider acknowledging these projects.
