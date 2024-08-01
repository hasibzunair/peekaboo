import sys
import torch

sys.path.insert(0, "../")
from model import PeekabooModel


# Variables
PRE_TRAINING = "dino"
ARCH = "vit_small"
PATCH_SIZE = 8
FEATS = "k"


def test_model_pretrained():
    model = PeekabooModel(
        vit_model=PRE_TRAINING,
        vit_arch=ARCH,
        vit_patch_size=PATCH_SIZE,
        enc_type_feats=FEATS,
    )
    model.decoder_load_weights("data/weights/peekaboo_decoder_weights_niter500.pt")


def test_model_function():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PeekabooModel(
        vit_model=PRE_TRAINING,
        vit_arch=ARCH,
        vit_patch_size=PATCH_SIZE,
        enc_type_feats=FEATS,
    )
    model.decoder_load_weights("data/weights/peekaboo_decoder_weights_niter500.pt")

    img = torch.ones(1, 3, 224, 224).to(device)
    pred = model(img)
    assert torch.isnan(pred.flatten()).sum().cpu().numpy() == 0


if __name__ == "__main__":
    test_model_pretrained()
    test_model_function()
