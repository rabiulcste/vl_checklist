import os
from example_models.xvlm.engine import XVLMWrapper
from vl_checklist.evaluate import Evaluate

CHECKPOINT_DIR = "/network/projects/aishwarya_lab/checkpoints/neruips23_lezhang_compositional_vl/xvlm"
CHECKPOINT_NAMES = [
    # "clip_coco-1e-06-weight0.2/checkpoints/epoch_5.pt",
    # "clip_coco-hn-5e-06-weight0.2/checkpoints/epoch_5.pt",
    # "rank_coco-dis_text-hn--5e-06-weight0.2-2/checkpoints/epoch_5.pt",
    # "rank-coco-mean-hn-5e-06-weight0.2-ub10/checkpoints/epoch_4.pt",
    # "rank-coco-mean-5e-06-weight0.2-1/checkpoints/epoch_4.pt",
    # "rank_coco-dis_text_mean-hn--5e-06-weightd0.3-weightr0.2-ub5-w_special/checkpoints/epoch_9.pt",
    # "rank-coco-mean-hn-5e-06-weight0.2-ub10/checkpoints/epoch_4.pt",
    # "rank_coco-dis_text_mean-hn--5e-06-weightd0.2-weightr0.2-ub5-w_special/checkpoints/epoch_5.pt",
    "all",
    "xvlm",
]

if __name__ == '__main__':
    for checkpoint_name in CHECKPOINT_NAMES:
        print(f"Evaluating checkpoint: {CHECKPOINT_DIR}")
        device = "cuda"
        model = XVLMWrapper(root_dir=CHECKPOINT_DIR, device=device, checkpoint_name=checkpoint_name)
        configs = [
            # "configs/our_clip_attribute.yaml",
            # "configs/our_clip_object.yaml",
            "configs/our_clip_relation.yaml",
            "configs/our_clip_relation_spatial.yaml",
        ]
        for config in configs:
            print(f"Evaluating config: {config}")
            eval = Evaluate(config, model=model)
            eval.start()


    


