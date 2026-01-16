
## Synthetic AI-READI Dataset

This is the code used to create the synthetic AI-READI dataset from the paper: XXXXX

### Prerequisites & Preparation

1. The packages used for this project are available in environment.yml
    ```sh
    conda env create -f environment.yml
    ```
    Note: you may need to install taming-transformers and clip within 'generation'
 
2. Additionally, we usedd a simple script to reduce the size of the original AI-READI images to 512x512 resolution before performing our analyses
   ```sh
   python downsizing.py
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Image Generation (Fundus and OCT images)

1. Navigate to generation
   ```sh
   cd generation
   ```
2. Update config files according to your implementation and run 
   ```sh
   python scripts/01_train_ldm.py -b configs/<your_vae_config_file>.yml
   python scripts/01_train_ldm.py -b configs/<your_diffusion_config_file>.yml
   ```
3. After both have finished training run  
   ```sh
   ./scripts/run_inference.sh
   ```
   This script uses accelerate, but the 02_generate_synthetic_dataset.py can be run directly without accelerate.

This process will save the synthetic images in the directory specified by data.save_path in the config file. 

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


