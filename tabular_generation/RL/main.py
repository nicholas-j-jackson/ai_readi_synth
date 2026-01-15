import time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os 
from pathlib import Path
import numpy as np 
import argparse 
from evaluation import (log_result_RL, predict, plot_rl_losses, mem_risk)
from hyperparams import HyperParams_Best
from models import build_models
import random
from tqdm import tqdm

def set_seed(seed: int = 42) -> None:
    random.seed(seed)                      
    np.random.seed(seed)          
    torch.manual_seed(seed)        
    torch.cuda.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed) 
 
def gen_save(H, G):
    #generate and save  
    z = torch.randn(H.NUM_SAMPLES, H.NOISE_DIM, device=H.DEVICE)
    synthetic, _, _, _ = G.sample(z)
    cols = H.NUM_COLS + H.CAT_COLS
    if H.DEVICE == "cuda": 
        df_syn = pd.DataFrame(synthetic.cpu().detach().numpy(), columns=cols)
    else: 
        df_syn = pd.DataFrame(synthetic.detach().numpy(), columns=cols)
    df_syn.to_csv(f"{H.OUT_DIR}/synthetic.csv")
    #rescale 
    feature_range = np.load(H.NPY_PATH, allow_pickle=True).item()
    for col in H.NUM_COLS:
        xmin, xmax = feature_range[col]
        df_syn[col] = (1.0 - df_syn[col]) * xmin + df_syn[col] * xmax
    df_syn.to_csv(f"{H.OUT_DIR}/synthetic_rescaled.csv")
    return df_syn

def train(df_train, real, loader, H): 
    start_time = time.time() 
    #instantiate
    G, D = build_models(H)
    opt_G = torch.optim.Adam(G.parameters(), lr=H.G_LR)
    opt_D = torch.optim.Adam(D.parameters(), lr=H.D_LR)
    os.makedirs(f"{H.OUT_DIR}/losses", exist_ok=True)
    with open(f"{H.OUT_DIR}/losses/output.txt", "w") as f:
        f.write(f"Logging\n")
    with open(f"{H.OUT_DIR}/losses/G_loss.txt", "w") as f:
        f.write(f"Logging\n")
    #ppo training loop
    real_iter = iter(loader)
    for it in tqdm(range(H.ITERS)):
        #rollout policy 
        z = torch.randn(H.BATCH, H.NOISE_DIM, device=H.DEVICE)
        rows, logp_old, _, v_old = G.sample(z)
        #detach for grad
        rows = rows.detach()      
        logp_old = logp_old.detach()
        v_old = v_old.detach()
        with torch.no_grad():
            rewards = torch.sigmoid(D(rows)).squeeze()
        adv = rewards - v_old 
        #normalize to smooth advantage 
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        #mean penalty 
        with torch.no_grad():
            target_mean = real.mean(0, keepdim=True).to(H.DEVICE)
        num_fake = rows[:, :len(H.NUM_COLS)]
        mean_pen = (num_fake.mean(0) - target_mean[0, :len(H.NUM_COLS)]).pow(2).mean()
        #PPO update 
        for _ in range(H.PPO_EPOCHS):
            logp, v = G.eval_action(z, rows)
            ratio = (logp - logp_old).exp()
            surr1 = ratio * adv_n
            surr2 = torch.clamp(ratio, 1-H.CLIP_EPS, 1+H.CLIP_EPS) * adv_n 
            loss_pi = -(torch.min(surr1, surr2)).mean()
            loss_v = F.mse_loss(v, rewards)
            entropy = -logp.mean()
            loss_G = loss_pi + H.VF_COEF * loss_v - H.ENT_BETA * entropy
            loss_G += H.MEAN_PENALTY_SCALE * mean_pen 
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()
            with open(f"{H.OUT_DIR}/losses/G_loss.txt", "a") as f:
                f.write(f"ITERATION {it:.4f} | LOSS PI {loss_pi:.4f} | LOSS V {H.VF_COEF*loss_v:.4f} | ENTROPY {(H.ENT_BETA*entropy):.4f} | MEAN PEN (*.2) {(H.MEAN_PENALTY_SCALE * mean_pen):.4f} | TOTAL G LOSS {loss_G:.4f}\n")
        #discriminator update 
        for d_it in range(H.DISC_STEPS):
            try:
                real_batch, = next(real_iter)
            except StopIteration:
                real_iter = iter(loader)
                real_batch, = next(real_iter)
            real_batch = real_batch.to(H.DEVICE)
            #fresh fake batch
            fake_batch, _, _, _ = G.sample(torch.randn(H.BATCH, H.NOISE_DIM, device=H.DEVICE))
            fake_batch = fake_batch.detach()
            #R1 gradient penalty on real data --> not quite WGAN!!! Could try with wasserstein distance 
            real_batch.requires_grad_(True)
            real_logits = D(real_batch)
            grad_real = torch.autograd.grad(real_logits.sum(), real_batch, create_graph=True)[0]
            gp = H.GRADIENT_PENALTY * 0.5 * grad_real.pow(2).view(real_batch.size(0), -1).sum(1).mean()
            loss_D = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_batch[:, :1])) + F.binary_cross_entropy_with_logits(D(fake_batch), torch.zeros_like(fake_batch[:, :1])) + gp
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()
        if it % 50 == 0:
            #print(f"{it} complete")
            with open(f"{H.OUT_DIR}/losses/output.txt", "a") as f:
                f.write(f"iteration {it} | D LOSS = {loss_D.item():.4f} | G LOSS = {loss_G.item():.4f} | AVG R ={rewards.mean():.4f} | mean_pen: {(mean_pen.item()*H.MEAN_PENALTY_SCALE):.4f} |  \n")
    
    elapsed_time = (time.time() - start_time) / 60 # minutes 
    df_syn = gen_save(H, G)
    return df_syn, elapsed_time



def evaluate_model_aireadi(df_hold_norm, df_train_norm, df_syn_norm, H, elapsed_time): 
    #plot losses
    plot_rl_losses(f"{H.OUT_DIR}")

    #-----------------UTILITY-----------------#
    #get hold out (keep same size no matter the train - test split)
    #ten_percent = 490
    #if len(df_hold_norm) > ten_percent: 
    #    hold_fraction = ten_percent / len(df_hold_norm)   
    #    df_hold_norm = df_hold_norm.sample(frac=hold_fraction, random_state=H.SEED).reset_index(drop=True)

    #synthetic to hold out 
    s2h_auc = {}
    r2r_auc = {}
    for col in ['Obesity', 'Type 2 Diabetes', 'Heart attack', 'Kidney problems', 'High blood cholesterol', 'High blood pressure']:
        s2h_auc[col] = predict(df_syn_norm,  df_hold_norm, f"{H.OUT_DIR}/synth_to_hold", H.SEED, label_col=col)[0]
        r2r_auc[col] = predict(df_train_norm,  df_hold_norm, f"{H.OUT_DIR}/real_to_hold", H.SEED, label_col=col)[0]

    #-----------------PRIVACY---------------------#
    mem_aucs = mem_risk(df_train_norm, df_hold_norm, df_syn_norm, H.CAT_COLS, H.NUM_COLS, f"{H.OUT_DIR}/mem_risk", H.SEED) 
    #-----------------LOG RESULTS-----------------#
    log_result_RL(H.RESULT_CSV, H.RUN_NAME, H.ITERS, H.DATA_SIZE, H.SEED, r2r_auc, s2h_auc, mem_aucs["real_roc_auc"], mem_aucs["synth_roc_auc"], elapsed_time)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--OUT_DIR", type=str, required=True )
    p.add_argument("--RESULT_CSV", type=str, required=True)
    p.add_argument("--RUN_NAME", type=str, required=True)
    p.add_argument("--SEED", type=int, required=True)
    p.add_argument("--ITERS", type=int, required=True)
    p.add_argument("--DATA_PATH", type=str, required=True)
    p.add_argument("--TRAIN", action='store_true')
    return vars(p.parse_args())
   
    
def main(): 


    #hyperparameter set up
    global H 
    H = HyperParams_Best().override(**parse_args())
    H = H.override(
        NPY_PATH = f"{H.DATA_PATH}/min_max_log.npy", 
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    )
    set_seed(H.SEED)

    
    os.makedirs(H.OUT_DIR, exist_ok=True)

    if H.TRAIN:

        df_train = pd.read_csv(f"{H.DATA_PATH}/normalized_training_data.csv")[H.NUM_COLS + H.CAT_COLS]
        real = torch.tensor(df_train.values, dtype=torch.float32)
        loader = DataLoader(TensorDataset(real), batch_size=H.BATCH, shuffle=True, num_workers=0) 
        df_syn, elapsed_time = train(df_train, real, loader, H)
    else:
        elapsed_time = 0
        
    #get raw to use for cwc, value stat analysis, histograms etc. 
    #df_train = pd.read_csv(f"{H.DATA_PATH}/original_training_data.csv")[H.NUM_COLS+ H.CAT_COLS]
    #df_hold = pd.read_csv(f"{H.DATA_PATH}/original_testing_data.csv")[H.NUM_COLS+ H.CAT_COLS]
    #df_real = pd.read_csv(f"{H.DATA_PATH}/original_data_with_patients.csv").drop(columns=['patient_id'])[H.NUM_COLS+H.CAT_COLS]
    #df_syn = pd.read_csv(f"{H.OUT_DIR}/synthetic_rescaled.csv")[H.NUM_COLS+H.CAT_COLS]
    #get normalized data to use for classifications (need patients to split real data without leakage)
    df_hold_norm = pd.read_csv(f"{H.DATA_PATH}/normalized_testing_data.csv")[H.NUM_COLS+ H.CAT_COLS]
    #df_real_with_patients_norm =  pd.read_csv(f"{H.DATA_PATH}/preprocessed_data_with_patients.csv")[H.NUM_COLS + H.CAT_COLS +["patient_id"]]
    df_syn_norm = pd.read_csv(f"{H.OUT_DIR}/synthetic.csv")[H.NUM_COLS+H.CAT_COLS].sample(8950)
    df_train_norm = pd.read_csv(f"{H.DATA_PATH}/normalized_training_data.csv")[H.NUM_COLS+ H.CAT_COLS]

    #evaluate 
    evaluate_model_aireadi(df_hold_norm, df_train_norm, df_syn_norm, H, elapsed_time) 

if __name__ == '__main__':
    
   

    main() 

