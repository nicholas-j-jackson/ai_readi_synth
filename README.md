
## Synthetic AI-READI Dataset

This is the code used to create the synthetic AI-READI dataset from the paper: XXXXX

### Prerequisites & Preparation

1. The packages used for this project are available in `environment.yml`:
```sh
conda env create -f environment.yml
```
This creates a conda environment named `synth_aireadi`. You may still need to install `taming-transformers` and `clip` manually within `generation` if they aren't picked up automatically.

2. Additionally, we used a simple script to reduce the size of the original AI-READI images to 512x512 resolution before performing our analyses:
```sh
python downsizing.py
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Image Generation (Fundus and OCT images)

All commands below are run from `generation/`, with the `synth_aireadi` conda environment active.

The pipeline has three stages: train an autoencoder (VAE), precompute latents for every image using that autoencoder, then train the diffusion model on those latents. Generation (sampling) loads both the diffusion checkpoint and the autoencoder checkpoint together (via `ldm.inference.EyeDiff`) to produce images.

1. **Train the autoencoder (VAE)**, once per modality:
```sh
python scripts/01_train_ldm.py -b configs/ae_32_fundus.yml
python scripts/01_train_ldm.py -b configs/ae_32_oct.yml
```
This trains a GAN-style `AutoencoderKL` (reconstruction + LPIPS + discriminator loss) that compresses 256x256 images down to 32x32x3 latents. Checkpoints land in `generation/logs/<timestamp>_<config-name>/checkpoints/`.

2. **Precompute latents** for every image, using the trained autoencoder:
```sh
python scripts/00_precompute_latents.py \
    -b configs/ae_32_fundus.yml \
    --ckpt logs/<timestamp>_ae_32_fundus/checkpoints/<epoch>.ckpt \
    --splits train val test
```
This encodes every raw image through the AE's encoder (deterministically, via `.mode()`, not `.sample()`) and caches the result as a `.npy` file under `latent_photography/` (fundus) or `latent_oct/` (OCT), mirroring the directory structure of `resized_retinal_photography/`/`resized_retinal_oct/` with `.jpg` swapped for `.npy`. This is what `diff-fundus.yml`/`diff-oct.yml`'s `pre_embed: true` reads from — the diffusion model is trained on these cached latents, not on raw images.

Add `--limit N` to only encode the first `N` images per split (useful for a quick, disk-cheap smoke test rather than encoding the full dataset).

3. **Train the diffusion model**, once per modality (requires step 2 to have been run for the same data):
```sh
python scripts/01_train_ldm.py -b configs/diff-fundus.yml
python scripts/01_train_ldm.py -b configs/diff-oct.yml
```
This is a class-conditional latent diffusion model (`ldm.models.diffusion.ddpm.LatentDiffusion`), conditioned via cross-attention on a learned embedding (`MultiClassEmbedder`) of each image's device/anatomy/laterality/disease attributes.

**Quick smoke tests**: `01_train_ldm.py` accepts `--max_epochs`, `--devices`, `--limit_train_batches`, and `--limit_val_batches` to run a fast, small-scale pass instead of a full training run. To test with a small, real subset of data (rather than the full dataset), pass a matching `data.params.limit=N` to *both* the precompute script's dataset construction and the training run — `AI_READI_Dataset` truncates deterministically to the first `N` rows, so the same `N` on both sides selects the same images. For example:
```sh
python scripts/00_precompute_latents.py -b configs/ae_32_fundus.yml --ckpt <ckpt> --splits train val --limit 64
python scripts/01_train_ldm.py -b configs/diff-fundus.yml --max_epochs 1 --devices 1 \
    --limit_train_batches 2 --limit_val_batches 1 --no-test true \
    data.params.limit=64 data.params.batch_size=8
```

4. **Generate synthetic images**, after both the autoencoder and diffusion model have finished training. Update `configs/synth-fundus.yml`/`configs/synth-oct.yml` with the trained `model_path` (diffusion checkpoint) and `ae_path` (autoencoder checkpoint), then run:
```sh
./scripts/run_inference.sh
```
This uses `accelerate` for multi-GPU sampling, but `02_generate_synthetic_dataset.py` can also be run directly (single process) without it. This saves images to the directory specified by `data.save_path` in the config, plus a CSV of per-image attributes/labels.

`02_generate_synthetic_dataset.py` refuses to write into the real, already-generated datasets (`/data/7TB/nick/ai_readi_v3/synth_fundus_32`, `synth_oct_32`) — it hard-blocks any `data.save_path` that resolves inside those directories, raising an error rather than overwriting. Pass `data.force=true` on the command line to override this only if you are certain.

**To quickly and safely test that generation works** without touching real data or requiring a fully-trained model, use:
```sh
./scripts/test_generation.sh <fundus|oct> <model_path> <ae_path> [n_classes] [num_samples]
```
This always writes to `generation/test_generation_output/<task>/` (gitignored, unrelated to any real dataset directory) and defaults to generating just 8 images, regardless of what checkpoints you point it at.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Tabular Data Generation

1. Navigate to preprocessing directory
```sh
cd tabular_generation/preprocessing_AIREADI
```
2. Specify the DATA_PATH and SAVE_PATH for your data in 'tabular_generation/preprocessing_AIREADI/run_preprocessing_all.sh' and run
```sh
./run_preprocessing_all.sh
```
3. Navigate to the RL model and run the training/eval script (after updating path variables in the script)
```sh
cd ../RL
./train_eval.sh
```

This process will save the synthetic images in the files titled synthetic.csv and synthetic_rescaled.csv in the specified save directory. These files differ in that synthetic.csv has been normalized and synthetic_rescaled matches the range of the original data. Evaluation metrics will be computed and saved to tabular_generation/results.csv (note that the membership risk computed here is not correct and is re-calculated in Analysis/tabular_privacy_eval.ipynb, see below)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Structuring in the format of AI-READI

The following jupyter notebooks within Structure_Dataset can be used to format the synthetic dataset in the same structure as the original AI-READI dataset:
```sh
cd ../Structure_Dataset
structure_synth_fundus_dataset.ipynb
structure_synth_oct_dataset.ipynb
structure_synth_tabular_dataset.ipynb
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Image Classification 

The image classification analysis can be run using a series of scripts that operate directly on top of the AI-READI directory or the synthetic directory created from the previous step.

1. Run RETFound over the real fundus and OCT images:
```sh
cd ../classification/RETFound_MAE
./finetune_RETFound_CFP.sh
./finetune_RETFound_OCT.sh
```
2. To repeat this model training and evaluation over the synthetic images:
```sh
./finetune_RETFound_CFP_synth.sh
./finetune_RETFound_OCT_synth.sh
```
3. Lastly, we use the RETFound embeddings to perform the privacy analysis. The following scripts extract these embeddings and save them to .csv files:
```sh
./extract_fundus.sh
./extract_oct.sh
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Analysis
1. Run the following jupyter notebooks to create the measurements needed for the privacy evaluation: 
```sh
Analysis/fundus_privacy_eval.ipynb
Analysis/oct_privacy_eval.ipynb
Analysis/tabular_privacy_eval.ipynb
```
2. After these have finished 'figures.ipynb can be run to generate the figures used for the paper
