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

## Usage of FOUND 

We provide here the different command lines in order to repeat all results provided in our paper. 

### Application to one image

Using the following command it is possible to apply and visualize our method on one single image.

```bash
python main_visualize.py --img-path /datasets_local/VOC2007/JPEGImages/000030.jpg
```

### Saliency object detection

We evaluate our method *FOUND* for the saliency detection on the datasets 

- [DUT-OMRON](http://saliencydetection.net/dut-omron/): `--dataset-eval DUT-OMRON`
- [DUTS-TEST](http://saliencydetection.net/duts/): `--dataset-eval DUTS-TEST`
- [ECSSD](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html): `--dataset-eval ECSSD`.

Please download those datasets and provide the dataset directory using the argument `--dataset-dir`.
The parameter `--evaluation-mode` allow you to choose between `single` and `multi` setup and `--apply-bilateral` can be added to apply the bilateral solver. Please find here examples on the dataset `ECSSD`.

```bash
export DATASET_DIR=datasets_local/ # Put here your directory

# ECSSD, single mode, with bilateral solver post-processing
python main_found_evaluate.py --eval-type saliency --dataset-eval ECSSD --evaluation-mode single --apply-bilateral --dataset-dir $DATASET_DIR
# same but multi mode
python main_found_evaluate.py --eval-type saliency --dataset-eval ECSSD --evaluation-mode multi --apply-bilateral --dataset-dir $DATASET_DIR
```

### Unsupervised object discovery

In order to evaluate on the unsupervised object discovery task, we follow the framework used in our previous work [LOST](https://github.com/valeoai/LOST).
The task is implemented for the following datasets, please download the benckmarks and put them in the folder `data/`.

- [VOC07](http://host.robots.ox.ac.uk/pascal/VOC/): `--dataset-eval VOC07`
- [VOC12](http://host.robots.ox.ac.uk/pascal/VOC/): `--dataset-eval VOC12`
- [COCO20k](https://cocodataset.org/#home): `--dataset-eval COCO20k`

```bash
export DATASET_DIR=datasets_local/ # Put here your directory

# VOC07
python main_found_evaluate.py --eval-type uod --dataset-eval VOC07 --evaluation-mode single --dataset-dir $DATASET_DIR
# VOC12
python main_found_evaluate.py --eval-type uod --dataset-eval VOC12 --evaluation-mode single --dataset-dir $DATASET_DIR
# COCO20k
python main_found_evaluate.py --eval-type uod --dataset-eval COCO20k --evaluation-mode single --dataset-dir $DATASET_DIR
```

## Training of FOUND

In order to train a FOUND model, please start by [installing](#installation-of-found) the framework. If already installed, please run again 

```bash
# Create conda environment
conda activate found

# Install dependencies
pip install -r requirements.txt
```

The training is performed on the dataset [DUTS-TR](http://saliencydetection.net/duts/) that should be put in the directory `data`. 

Then the training can be launched using the following command. Visualizations and training curves can be observed using tensorboard.
```bash
export DATASET_DIR=datasets_local # Root directory of all datasets, both training and evaluation

python main_found_train.py --exp-name repro --dataset-dir $DATASET_DIR
```

Once the training done, you can launch the evaluation using the scripts `evaluate_saliency.sh` and `evaluate_uod.sh` with the commands:

```bash
export MODEL="outputs/repro-FOUND-DUTS-TR-vit_small8/decoder_weights_niter500.pt"

# Evaluation of saliency detection
source evaluate_saliency.sh $MODEL $DATASET_DIR single
source evaluate_saliency.sh $MODEL $DATASET_DIR multi

# Evaluation of unsupervised object discovery
source evaluate_uod.sh $MODEL $DATASET_DIR
```