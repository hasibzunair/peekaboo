"""Visualize outputs and save masks of both model predictions and ground truths.
"""

import os
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np

from PIL import Image
from model import FoundModel
from misc import load_config
from torchvision import transforms as T

NORMALIZE = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = 'Evaluation of FOUND',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--img-folder", type=str, default="data/examples/", help="Image folder path."
    )
    parser.add_argument(
        "--model-weights", type=str, default="data/weights/decoder_weights.pt",
    )
    parser.add_argument(
        "--config", type=str, default="configs/msl_DUTS-TR.yaml",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
    )
    args = parser.parse_args()

    # Saving dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Configuration
    config = load_config(args.config)

    # ------------------------------------
    # Load the model
    model = FoundModel(vit_model=config.model["pre_training"],
                        vit_arch=config.model["arch"],
                        vit_patch_size=config.model["patch_size"],
                        enc_type_feats=config.found["feats"],
                        bkg_type_feats=config.found["feats"],
                        bkg_th=config.found["bkg_th"])
    # Load weights
    model.decoder_load_weights(args.model_weights)
    model.eval()
    print(f"Model {args.model_weights} loaded correctly.")

    img_paths = sorted([os.path.join(args.img_folder, path) for path in os.listdir(args.img_folder)])

    dir = "./datasets_local/DUT-OMRON/pixelwiseGT-new-PNG/"
    mask_paths = sorted([os.path.join(dir, path) for path in os.listdir(dir)])

    for img_path, mask_path in zip(img_paths, mask_paths):
        # Load the image
        with open(img_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            img_np = np.array(img)

            t = T.Compose([T.ToTensor(), NORMALIZE])
            img_t = t(img)[None,:,:,:]
            inputs = img_t.to("cuda")
        
        # Load mask
        with open(mask_path, "rb") as f:
            mask = Image.open(f).convert("P")
            mask_np = np.array(mask)
            mask_np = (mask_np / np.max(mask_np) * 255).astype(np.uint8)
            mask_np_3d = np.stack([mask_np, mask_np, mask_np], axis=-1)


        # Forward step
        with torch.no_grad():
            preds, _, shape_f, att = model.forward_step(inputs, for_eval=True)

        # Apply FOUND
        sigmoid = nn.Sigmoid()
        h, w = img_t.shape[-2:]
        preds_up = F.interpolate(
            preds, scale_factor=model.vit_patch_size, mode="bilinear", align_corners=False
        )[..., :h, :w]
        preds_up = (
            (sigmoid(preds_up.detach()) > 0.5).squeeze(0).float()
        )

        preds_up_np = preds_up.cpu().squeeze().numpy()
        preds_up_np = (preds_up_np / np.max(preds_up_np) * 255).astype(np.uint8)
        preds_up_np_3d = np.stack([preds_up_np, preds_up_np, preds_up_np], axis=-1)

        #import pdb; pdb.set_trace()
        # plt.imshow(combined_image); plt.show()

        combined_image = cv2.addWeighted(img_np, 0.5, mask_np_3d, 0.5, 0)
        combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)

        save_path = os.path.join(args.output_dir, img_path.split('/')[-1])
        cv2.imwrite(save_path, combined_image)

        print(f'Saved image in {save_path} with shape {combined_image.shape}')