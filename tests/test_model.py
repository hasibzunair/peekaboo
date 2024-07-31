import sys
import torch

sys.path.insert(0, "../")
from model import PeekabooModel
from misc import load_config


def test_model_pretrained():
    config, _ = load_config("configs/peekaboo_DUTS-TR.yaml")
    model = PeekabooModel(
        vit_model=config.model["pre_training"],
        vit_arch=config.model["arch"],
        vit_patch_size=config.model["patch_size"],
        enc_type_feats=config.peekaboo["feats"],
    )
    model.decoder_load_weights("data/weights/peekaboo_decoder_weights_niter500.pt")


def test_model_function():
    config, _ = load_config("configs/peekaboo_DUTS-TR.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PeekabooModel(
        vit_model=config.model["pre_training"],
        vit_arch=config.model["arch"],
        vit_patch_size=config.model["patch_size"],
        enc_type_feats=config.peekaboo["feats"],
    )
    model.decoder_load_weights("data/weights/peekaboo_decoder_weights_niter500.pt")

    img = torch.ones(1, 3, 224, 224).to(device)
    pred = model(img)
    assert torch.isnan(pred.flatten()).sum().cpu().numpy() == 0


if __name__ == "__main__":
    test_model_pretrained()
    test_model_function()
