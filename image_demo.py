# Code for Peekaboo
# Author: Hasib Zunair
# Modified from https://github.com/valeoai/FOUND, see license below.

# Copyright 2022 - Valeo Comfort and Driving Assistance - Oriane Siméoni @ valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Visualize model predictions"""

import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from model import PeekabooModel
from misc import load_config
from torchvision import transforms as T
from torchinfo import summary

from misc import get_bbox_from_segmentation_labels

NORMALIZE = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation of Peekaboo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--img-path",
        type=str,
        default="data/examples/octopus.jpeg",
        help="Image path.",
    )
    parser.add_argument(
        "--model-weights",
        type=str,
        default="data/weights/peekaboo_decoder_weights_niter500.pt",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/peekaboo_DUTS-TR.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
    )
    args = parser.parse_args()

    # Saving dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Configuration
    config, _ = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = PeekabooModel(
        vit_model=config.model["pre_training"],
        vit_arch=config.model["arch"],
        vit_patch_size=config.model["patch_size"],
        enc_type_feats=config.peekaboo["feats"],
    )
    # Load weights
    model.decoder_load_weights(args.model_weights)
    model.eval()
    print(f"Model {args.model_weights} loaded correctly.")

    # Print params
    summary(model, input_size=(1, 3, 224, 224))
    print(f"\n")

    # Load the image
    with open(args.img_path, "rb") as f:
        img = Image.open(f)
        img = img.convert("RGB")
        original_size = img.size  # (w, h)

        # Preprocess
        t = T.Compose([T.Resize((224, 224)), T.ToTensor(), NORMALIZE])
        img_t = t(img)[None, :, :, :]
        inputs = img_t.to(device)

    # Forward step
    with torch.no_grad():
        preds = model(inputs, for_eval=True)
        print(f"Shape of output is {preds.shape}")

    sigmoid = nn.Sigmoid()
    h, w = img_t.shape[-2:]
    orig_h, orig_w = original_size[1], original_size[0]
    preds_up = F.interpolate(
        preds, size=(orig_h, orig_w), mode="bilinear", align_corners=False
    )
    print(f"Shape of output after interpolation is {preds_up.shape}")
    preds_up = (sigmoid(preds_up.detach()) > 0.5).squeeze(0).float()

    # Get segmentation mask
    pred_bin_mask = preds_up.cpu().squeeze().numpy().astype(np.uint8)
    initial_image_size = img.size[::-1]
    scales = [
        initial_image_size[0] / pred_bin_mask.shape[0],
        initial_image_size[1] / pred_bin_mask.shape[1],
    ]

    # Get bounding box for single object discovery
    pred_bbox = get_bbox_from_segmentation_labels(
        pred_bin_mask, initial_image_size, scales
    )
    print(f"Predicted bounding box: {pred_bbox}")

    # Plot mask and box in the image
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.imshow(pred_bin_mask, "gray", interpolation="none", alpha=0.5)

    rect = patches.Rectangle(
        (pred_bbox[0], pred_bbox[1]),
        pred_bbox[2] - pred_bbox[0],
        pred_bbox[3] - pred_bbox[1],
        linewidth=2,
        edgecolor="b",
        facecolor="none",
    )
    ax.add_patch(rect)

    plt.axis("off")
    img_name = args.img_path
    img_name = img_name.split("/")[-1].split(".")[0]
    plt.savefig(
        os.path.join(args.output_dir, f"{img_name}-peekaboo.png"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()
    print(f"Saved model prediction.")
