from example_models.open_clip.engine import OpenCLIP
from vl_checklist.evaluate import Evaluate

CHECKPOINT_DIR = (
    "/network/projects/aishwarya_lab/checkpoints/neruips23_lezhang_compositional_vl/clip/"
)
CHECKPOINT_NAMES = [
    "itchn_cmr.pt",
    "itchn.pt",
    "itchn_tec.pt",
    "itc.pt",
    "all.pt",
    "negclip.pt",
]

MODEL_NAMES = [
    "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
    "hf-hub:laion/CLIP-ViT-B-16-DataComp.L-s1B-b8K",
    "hf-hub:laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K",
    "hf-hub:laion/CLIP-ViT-B-32-DataComp.M-s128M-b4K",
    "hf-hub:laion/CLIP-ViT-B-32-DataComp.S-s13M-b4K",
    "hf-hub:laion/CLIP-ViT-L-14-CommonPool.XL.clip-s13B-b90K",
    "hf-hub:laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K",
    "hf-hub:laion/CLIP-ViT-L-14-CommonPool.XL.laion-s13B-b90K",
    "hf-hub:laion/CLIP-ViT-B-16-CommonPool.L.clip-s1B-b8K",
    "hf-hub:laion/CLIP-ViT-B-32-CommonPool.S-s13M-b4K",
    "hf-hub:laion/CLIP-ViT-B-16-CommonPool.L.image-s1B-b8K",
    "hf-hub:laion/CLIP-ViT-B-16-CommonPool.L.basic-s1B-b8K",
    "hf-hub:laion/CLIP-ViT-B-32-CommonPool.S.text-s13M-b4K",
    # "hf-hub:laion/CoCa-ViT-L-14-laion2B-s13B-b90k",
    # "hf-hub:laion/CoCa-ViT-B-32-laion2B-s13B-b90k",
    "hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
    "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K",
    "hf-hub:laion/CLIP-ViT-g-14-laion2B-s34B-b88K",
    # "hf-hub:laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k",
    "hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    "hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K",
    "hf-hub:laion/CLIP-convnext_large_d.laion2B-s26B-b102K-augreg",
    "hf-hub:laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup",
    "hf-hub:laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup",
    "hf-hub:laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-rewind",
    "hf-hub:laion/CLIP-convnext_base_w_320-laion_aesthetic-s13B-b82K-augreg",
    # "hf-hub:laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k",
    "hf-hub:laion/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k",
    "hf-hub:laion/mscoco_finetuned_CoCa-ViT-B-32-laion2B-s13B-b90k",
]

VL_CHECKLIST_CONFIGS = [
    "configs/our_clip_attribute.yaml",
    "configs/our_clip_object.yaml",
    "configs/our_clip_relation.yaml",
    "configs/our_clip_relation_spatial.yaml",
]

# argparser for command line arguments
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="")
args = parser.parse_args()

if __name__ == "__main__":
    for checkpoint_name in CHECKPOINT_NAMES:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
        print(f"Evaluating checkpoint: {checkpoint_path}")
        model = OpenCLIP(checkpoint_path)

    prompt = args.prompt
    for model_name in MODEL_NAMES:
        try:
            model = OpenCLIP(model_id=model_name, src_type="hf_hub")
            for config_name in VL_CHECKLIST_CONFIGS:
                print(f"Evaluating {config_name}")
                eval = Evaluate(config_name, model=model)
                eval.start(prompt=prompt)
        except Exception as e:
            print(f"Error in evaluating {config_name}: {e}")
