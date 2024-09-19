import os
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import codecs
import numpy as np
import cv2

from PIL import Image
from model import PeekabooModel
from misc import load_config
from torchvision import transforms as T

NORMALIZE = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def inference(img_path):
    # Load the image
    with open(img_path, "rb") as f:
        img = Image.open(f)
        img = img.convert("RGB")
        img_np = np.array(img)

        # Preprocess
        t = T.Compose([T.ToTensor(), NORMALIZE])
        img_t = t(img)[None, :, :, :]
        inputs = img_t.to(device)

    # Forward step
    print(f"Start Peekaboo prediction.")
    with torch.no_grad():
        preds = model(inputs, for_eval=True)
    print(f"Done Peekaboo prediction.")

    sigmoid = nn.Sigmoid()
    h, w = img_t.shape[-2:]
    preds_up = F.interpolate(
        preds,
        scale_factor=model.vit_patch_size,
        mode="bilinear",
        align_corners=False,
    )[..., :h, :w]
    preds_up = (sigmoid(preds_up.detach()) > 0.5).squeeze(0).float()
    preds_up = preds_up.cpu().squeeze().numpy()

    # Overlay predicted mask with input image
    preds_up_np = (preds_up / np.max(preds_up) * 255).astype(np.uint8)
    preds_up_np_3d = np.stack([preds_up_np, preds_up_np, preds_up_np], axis=-1)
    combined_image = cv2.addWeighted(img_np, 0.5, preds_up_np_3d, 0.5, 0)
    print(f"Output shape is {combined_image.shape}")
    return combined_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation of Peekaboo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--img-path",
        type=str,
        default="data/examples/VOC_000030.jpg",
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

    # App
    title = "PEEKABOO: Hiding Parts of an Image for Unsupervised Object Localization"
    description = codecs.open("./media/description.html", "r", "utf-8").read()
    article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2407.17628' target='_blank'>PEEKABOO: Hiding Parts of an Image for Unsupervised Object Localization</a> | <a href='https://github.com/hasibzunair/peekaboo' target='_blank'>Github</a></p>"

    gr.Interface(
        inference,
        gr.components.Image(type="filepath", label="Input Image"),
        gr.components.Image(type="numpy", label="Predicted Output"),
        examples=[
            "./data/examples/octopus.jpeg",
            "./data/examples/godzillaxkong.jpeg",
            "./data/examples/avengers.jpeg",
            "./data/examples/dinosaur.jpeg",
            "./data/examples/chitauri.jpeg",
        ],
        title=title,
        description=description,
        article=article,
        allow_flagging=False,
        analytics_enabled=False,
    ).launch()
