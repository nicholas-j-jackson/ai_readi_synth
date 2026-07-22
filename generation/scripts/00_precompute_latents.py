import argparse
import os
import sys
from time import strftime

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from ldm import instantiate_from_config
from dataset import AI_READI_Dataset


def print_with_prefix(*messages):
    prefix = f"\033[34m[Latent-Precompute {strftime('%Y-%m-%d %H:%M:%S')}]\033[0m"
    print(f"{prefix}: {' '.join(map(str, messages))}")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Encode the raw AI-READI fundus/OCT images through a trained "
                    "AutoencoderKL and cache the resulting latents as .npy files, "
                    "at the paths AI_READI_Dataset(pre_embed=True) expects them."
    )
    parser.add_argument(
        "-b", "--base", required=True,
        help="AE config the checkpoint was trained with (e.g. configs/ae_32_fundus.yml)",
    )
    parser.add_argument("--ckpt", required=True, help="path to the trained AutoencoderKL checkpoint")
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="cap the number of images encoded per split, for a quick smoke test",
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser


if __name__ == "__main__":
    opt = get_parser().parse_args()

    config = OmegaConf.load(opt.base)
    task = config.data.params.task
    data_path = config.data.params.data_path

    device = torch.device(opt.device)

    model = instantiate_from_config(config.model)
    state_dict = torch.load(opt.ckpt, map_location=device)["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    # Same preprocessing DataModuleFromConfig uses for pre_embed=False, so the
    # cached latents match what the AE was actually trained/decoded on.
    transforms = Compose([
        Resize(256),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    src_dir = "resized_retinal_oct" if task == "oct" else "resized_retinal_photography"
    dst_dir = "latent_oct" if task == "oct" else "latent_photography"

    for split in opt.splits:
        dataset = AI_READI_Dataset(data_path, transforms, mode=split, task=task, pre_embed=False, limit=opt.limit)
        print_with_prefix(f"Encoding {len(dataset)} {task} images for split={split}")

        loader = DataLoader(
            dataset,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            shuffle=False,
        )

        offset = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc=split):
                imgs = batch["img"].to(device)
                bs = imgs.shape[0]
                z = model.encode(imgs).mode().cpu().numpy()

                for i in range(bs):
                    raw_relpath = dataset.df.iloc[offset + i]["filepath"]
                    latent_relpath = raw_relpath.replace(".jpg", ".npy").replace(src_dir, dst_dir)
                    # filepath values have a leading slash (see updated_manifest.csv), so
                    # os.path.join would discard data_path entirely; match dataset.py's
                    # own convention of plain concatenation instead.
                    out_path = data_path.rstrip("/") + latent_relpath
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    np.save(out_path, z[i])

                offset += bs

        print_with_prefix(f"Done with split={split}")
