import os
import warnings
import argparse
import sys
import math

import torch
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image, to_grayscale, to_tensor

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, ConcatDataset
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from pytorch_lightning import seed_everything
from accelerate import Accelerator
from accelerate.utils import gather_object 

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from ldm import instantiate_from_config
from dataset import AI_READI_Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging",
    )
    parser.add_argument(
        "-s",
        '--split',
        type=str,
        default='train'
    )
    return parser

from time import strftime
def print_with_prefix(*messages):
    prefix = f"\033[34m[EyeDiff-Sampling {strftime('%Y-%m-%d %H:%M:%S')}]\033[0m"
    combined_message = ' '.join(map(str, messages))
    print(f"{prefix}: {combined_message}")

if __name__ == "__main__":
    sys.path.append(os.getcwd())

    # Parse args
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    split = opt.split

    # Create configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    # DDP & accelerator
    torch.backends.cuda.matmul.allow_tf32 = True  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    accelerator = Accelerator()
    seed = 23 * accelerator.num_processes + accelerator.process_index
    torch.manual_seed(seed)

    # torch.cuda.set_device(device)
    print_with_prefix(f"Starting rank={accelerator.local_process_index}, seed={seed}, world_size={accelerator.num_processes}.")
    rank = accelerator.local_process_index

    # Load model
    device = accelerator.device
    OmegaConf.update(config, "model.params.device", str(device))
    print(config)

    model = instantiate_from_config(config.model)
    model = accelerator.prepare(model)  # Prepare model for distributed inference
    seed_everything(seed)


    # Data (for labels)
    batch_size = config.data.batch_size
    dataset = AI_READI_Dataset(config.data.data_path, transforms=None, mode=split, task=config.data.task, pre_embed=False)


    # Create save directories
    if rank == 0:
        os.makedirs(os.path.join(config.data.save_path, split), exist_ok=True)

        if accelerator.process_index == 0:
            print_with_prefix(f"Saving .png samples at {config.data.save_path}/{split}")
    accelerator.wait_for_everyone()


    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = batch_size
    global_batch_size = n * accelerator.num_processes

    # Optional cap on the number of samples generated (e.g. for a quick smoke test);
    # defaults to the full dataset size, matching prior behavior.
    target_count = len(dataset) if config.data.get("limit", None) is None else min(config.data.limit, len(dataset))

    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    num_samples = len([name for name in os.listdir(config.data.save_path+'/'+split) if (os.path.isfile(os.path.join(config.data.save_path, split, name)) and ".png" in name)])
    total_samples = int(math.ceil(target_count / global_batch_size) * global_batch_size)


    if rank == 0:
        if accelerator.process_index == 0:
            print_with_prefix(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % accelerator.num_processes == 0, "total_samples must be divisible by world_size"
    
    samples_needed_this_gpu = int(total_samples // accelerator.num_processes)
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    
    
    iterations = int(samples_needed_this_gpu // n)
    done_iterations = int( int(num_samples // accelerator.num_processes) // n)
    pbar = tqdm(range(iterations))
    total = 0

    # Main Loop
    data = []
    for i in pbar:
        label = torch.stack([dataset[x]['class_label'] for x in np.random.randint(0, len(dataset.df), n)]).to(device)#batch['class_label']
        label[label==777] = 0 # replace missing
        
        # Move label to the correct device using accelerator
        label = accelerator.prepare(label)

        imgs = model.sample(
            # Number of images to synthesize
            batch_size=batch_size,
            # Number of DDIM sampling steps
            sampling_steps=100,
            # eta in DDIM sampling
            eta=1.0,
            # Use the AE decoder to translate from latent space.
            decode=True,
            conditioning={'class_label': label} #torch.tensor(label.tolist(), dtype=torch.float32).to(device)},
        )

        # The image are still in [-1, 1] so they need to be rescaled.
        imgs.clamp_(-1, 1)
        imgs = (imgs + 1) / 2

        filenames = []
        #if accelerator.is_main_process:  # Save images only in the main process

        for j, img in enumerate(imgs):
            index = j * accelerator.num_processes + accelerator.process_index + total
            #print_with_prefix(index, accelerator.num_processes, accelerator.process_index, total)
            img = (img.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            Image.fromarray(img).save(f"{config.data.save_path}/{split}/{index:06d}.png")                    
            data.append(torch.cat([label[j], torch.tensor([index]).to(device)])) #torch.tensor(f"{config.data.save_path}/{index:06d}.png")])
            #print(data)
        
        total += global_batch_size
        accelerator.wait_for_everyone()
    

    # Ensure all processes are synchronized
    accelerator.wait_for_everyone()
    all_data = gather_object(data)
    all_data = torch.cat([v.cpu().reshape(1,-1) for v in all_data])
    
    if accelerator.is_main_process:
        # Save dataframe
        all_data = pd.DataFrame(all_data, columns=dataset.cols + ['index']).astype(int) #pd.DataFrame({k: v.cpu() for k,v in all_data.items()}, columns=dataset.cols + ['filepath'])
        all_data['filepath'] = all_data['index'].apply(lambda x: f"{config.data.save_path}/{split}/{x:06d}.png")
        all_data.set_index('index').to_csv(f"{config.data.save_path}/{split}.csv")#.sort_index()
        print_with_prefix(f"Saved data to {config.data.save_path}/{split}.csv")
