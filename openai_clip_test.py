import os
from example_models.clip.engine import CLIP
from vl_checklist.evaluate import Evaluate

CHECKPOINT_DIR = "/network/projects/aishwarya_lab/checkpoints/neruips23_lezhang_compositional_vl/clip/"

MODEL_NAMES = [
    "RN50",
    "RN101",
    "RN50x4",
    "RN50x16",
    "RN50x64",
    "ViT-B/32",
    "ViT-B/16",
    "ViT-L/14",
    "ViT-L/14@336px",
]

configs = [
    "configs/our_clip_attribute.yaml",
    "configs/our_clip_object.yaml",
    "configs/our_clip_relation.yaml",
    "configs/our_clip_relation_spatial.yaml",
]
if __name__ == '__main__':
    for model_name in MODEL_NAMES:
        print(f"Evaluating checkpoint: {model_name}")
        model = CLIP(model_name)

        for config in configs:
            try:
                print(f"Evaluating config: {config}")
                eval = Evaluate(config, model=model)
                eval.start()
            except Exception as e:
                print(f"Error in evaluating {config}: {e}")

    


