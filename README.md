### Env install 

```bash
# Create conda environment
conda create -n found python=3.7
conda activate found
# Example of pytorch installation
pip install torch===1.8.1 torchvision==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install pycocotools
# Install dependencies
pip install -r requirements.txtt
```

Install [DINO](https://arxiv.org/pdf/2104.14294.pdf):

```bash
git clone https://github.com/facebookresearch/dino.git
cd dino; 
touch __init__.py
echo -e "import sys\nfrom os.path import dirname, join\nsys.path.insert(0, join(dirname(__file__), '.'))" >> __init__.py; cd ../;
```

## Dataset details

Saliency object detection

- [DUT-OMRON](http://saliencydetection.net/dut-omron/): `--dataset-eval DUT-OMRON`
- [DUTS-TEST](http://saliencydetection.net/duts/): `--dataset-eval DUTS-TEST`
- [ECSSD](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html): `--dataset-eval ECSSD`.

Unsupervised object discovery

- [VOC07](http://host.robots.ox.ac.uk/pascal/VOC/): `--dataset-eval VOC07`
- [VOC12](http://host.robots.ox.ac.uk/pascal/VOC/): `--dataset-eval VOC12`
- [COCO20k](https://cocodataset.org/#home): `--dataset-eval COCO20k`

## Training

The training is performed on the dataset [DUTS-TR](http://saliencydetection.net/duts/) that should be put in the directory `datasets_local`.

Then the training can be launched using the following command.

```bash
export DATASET_DIR=datasets_local # Root directory of all datasets, both training and evaluation

python main_found_train.py --exp-name repro --dataset-dir $DATASET_DIR
```

## Eval

Once the training done, you can launch the evaluation using the scripts `evaluate_saliency.sh` and `evaluate_uod.sh` with the commands:

```bash
export MODEL="outputs/exp_name/decoder_weights_niter500.pt"

# Evaluation of saliency detection
source evaluate_saliency.sh $MODEL $DATASET_DIR single
source evaluate_saliency.sh $MODEL $DATASET_DIR multi

# Evaluation of unsupervised object discovery
source evaluate_uod.sh $MODEL $DATASET_DIR
```

See tensorboard logs by running: `tensorboard --logdir=outputs`.

## Demo

We provide here the different command lines in order to repeat all results provided in our paper.

Using the following command it is possible to apply and visualize our method on one single image.

```bash
python main_visualize.py --img-path ./datasets_local/ECSSD/images/0009.jpg
```

## Project Notes

<details><summary>Click to view</summary>
<br>

**[Nov 10, 2023]** Reproduced FOUND results.

**[Nov 10, 2023]** Added project notes section.

</details>

## Acknowledgements

This repository was built on top of https://github.com/valeoai/FOUND.