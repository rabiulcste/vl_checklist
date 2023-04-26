import itertools
import logging

import fsspec
import torch.cuda
from open_clip import create_model_and_transforms, get_tokenizer
from PIL import Image

from vl_checklist.vlp_model import VLPModel


# TODO: this is repeated in the other file too (src/training/file_utils.py)
def pt_load(file_path, map_location=None):
    if not file_path.startswith("/"):
        logging.info("Loading remote checkpoint, which may take a bit.")
    of = fsspec.open(file_path, "rb")
    with of as f:
        out = torch.load(f, map_location=map_location)
    return out


def chunks(iterable, size):
    """
    Break an iterable into chunks of the given size.
    """
    iterator = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(iterator, size))
        if not chunk:
            return
        yield chunk


class OpenCLIP(VLPModel):
    def __init__(self, model_id):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dir = "resources"
        self.model_id = model_id
        self.model, self.preprocess = self._load_model(self.model_id)
        self.tokenizer = get_tokenizer("ViT-B-32")

    def model_name(self):
        return self.model_id

    def _load_model(self, model_id):
        if model_id is None:
            raise Exception("Model ID cannot be None.")
        print("Loading model: {}".format(model_id))

        model, _, preprocess = create_model_and_transforms(
            "ViT-B-32", "openai", device=self.device
        )

        checkpoint = pt_load(model_id, map_location="cpu")
        if "epoch" in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            sd = checkpoint["state_dict"]
            if next(iter(sd.items()))[0].startswith("module"):
                sd = {k[len("module.") :]: v for k, v in sd.items()}
            model.load_state_dict(sd)

        return model, preprocess

    def predict(self, images: list, texts: list, src_type: str = "local"):
        # text format is [["there is a cat","there is a dog"],[...,...]...]
        images = [Image.open(image) for image in images]
        images = [self.preprocess(image) for image in images]
        images = torch.stack(images).to(self.device)
        texts = self.tokenizer(texts).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(texts)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            scores_matrix = image_features @ text_features.T
            img_txt_probs = torch.diag(scores_matrix).cpu().numpy().tolist()
        return {"probs": img_txt_probs}
