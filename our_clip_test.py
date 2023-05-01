import os
from example_models.open_clip.engine import OpenCLIP
from vl_checklist.evaluate import Evaluate

CHECKPOINT_DIR = "/network/projects/aishwarya_lab/checkpoints/compositional_vl/Outputs"
CHECKPOINT_NAMES = [
    "clip_coco-1e-06-weight0.2/checkpoints/epoch_5.pt",
    "rank_coco-dis_text-hn--5e-06-weight0.2-2/checkpoints/epoch_5.pt",
    "rank-coco-mean-hn-5e-06-weight0.2-ub10/checkpoints/epoch_4.pt"
    "rank-coco-mean-5e-06-weight0.2-1/checkpoints/epoch_4.pt",
    "rank_coco-dis_text_mean-hn--5e-06-weightd0.3-weightr0.2-ub5-w_special/checkpoints/epoch_9.pt",
    "rank-coco-mean-hn-5e-06-weight0.2-ub10/checkpoints/epoch_4.pt"
    "rank_coco-dis_text_mean-hn--5e-06-weightd0.2-weightr0.2-ub5-w_special/checkpoints/epoch_5.pt",
]

if __name__ == '__main__':
    for checkpoint_name in CHECKPOINT_NAMES:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
        print(f"Evaluating checkpoint: {checkpoint_path}")
        model = OpenCLIP(checkpoint_path)
        configs = [
            "configs/our_clip_attribute.yaml",
            "configs/our_clip_object.yaml",
            "configs/our_clip_relation.yaml"
        ]
        for config in configs:
            try:
                print(f"Evaluating config: {config}")
                eval = Evaluate(config, model=model)
                eval.start()
            except Exception as e:
                print(f"Error in evaluating {config}: {e}")
                continue

    


