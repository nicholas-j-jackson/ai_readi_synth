import torch, torch.nn as nn, torch.nn.functional as F
import os 
import optuna
import gc 
from optuna.visualization import (plot_optimization_history, plot_parallel_coordinate, plot_pareto_front, plot_param_importances)
from evaluation.evaluate_aireadi import (log_result_RL_search, compute_value_stats,  train_on_synth_test_on_hold, get_column_wise_correlations, train_on_real_test_on_synth,)
from RL.hyperparams import HyperParams_Search 
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tabular_generation.RL.main import train, set_seed
import os 
from pathlib import Path
import numpy as np 
import random

def evaluate_model(df_real, df_hold, df_syn, df_hold_norm, df_syn_norm, df_real_with_patients_norm, H): 
    #-----------------UTILITY-----------------#
    #get hold out (keep same size no matter the train - test split)
    ten_percent = 490
    if len(df_hold_norm) > ten_percent: 
        hold_fraction = ten_percent / len(df_hold_norm)   
        df_hold_norm = df_hold_norm.sample(frac=hold_fraction, random_state=H.SEED).reset_index(drop=True)
    #synthetic to hold out 
    s2h_auc, s2h_acc = train_on_synth_test_on_hold(df_syn_norm,  df_hold_norm, f"{H.OUT_DIR}/synth_to_hold", H.SEED)
    #real to synthetic 
    r2s_auc, r2s_acc = train_on_real_test_on_synth(df_real_with_patients_norm, df_syn_norm, f"{H.OUT_DIR}/real_to_synth", H.SEED)
    #-----------------LOG RESULTS-----------------#
    with open(f"{H.OUT_DIR}/eval.txt", "a") as f:
        f.write(f"r2s_auc: {r2s_auc}\nr2s_acc: {r2s_acc}\ns2h_auc: {s2h_auc}\ns2h_acc: {s2h_acc}\n")
        f.write(f"BATCH: {H.BATCH}, NOISE_DIM: {H.NOISE_DIM}, N_CRITIC: {H.DISC_STEPS}, GP_COEFF: {H.GRADIENT_PENALTY}, DISC_LR: {H.D_LR}, GEN LR: {H.G_LR}, G_HIDDEN_DIM: {H.G_H}, D_HIDDEN_DIM: {H.D_H}")
    log_result_RL_search(H.RESULT_CSV, H.RUN_NAME, H.ITERS, H.DATA_SIZE, H.SEED, s2h_auc, s2h_acc, r2s_auc, r2s_acc)
    return s2h_auc, r2s_auc 

def objective(trial:optuna.Trial): 
    #define parameters 
    BATCH = trial.suggest_int("batch", 64, 512, step=64) #128 
    NOISE_DIM = trial.suggest_int("noise_dim", 16, 128, step=16) # 16 
    PPO_EPOCHS = trial.suggest_int("ppo_epochs", 1, 5) # 4 
    DISC_STEPS = trial.suggest_int("disc_steps", 1, 5) # 1 
    MEAN_PENALTY_SCALE = trial.suggest_float("mean_penalty", 0, 0.2, step=0.1) # 0 or 0.2 
    GRADIENT_PENALTY = trial.suggest_int("gradient_penalty", 5, 10, step = 5) # 10 
    USE_TANH = trial.suggest_categorical("use_tanh", [True, False])
    G_LR = trial.suggest_categorical("g_lr", [5e-5, 1e-4, 2e-4]) #1e-4 
    D_LR = trial.suggest_categorical("d_lr", [1e-5, 3e-5, 5e-5, 5e-4]) # 3e-5 
    G_H = trial.suggest_int("g_h", 64, 128, step=64) # 64 
    D_H = trial.suggest_int("d_h", 64, 128, step=64) # 64 
    #define save paths 
    
    H = HyperParams_Search()
    H = H.override(
        OUT_DIR = f"RL_optuna_full_sun/trial_{trial.number}", 
        DATA_PATH = f"AI-READI-FULL/preprocessed_data0.9_seed0", 
        RESULT_CSV = f"RL_optuna_full_sun/results.csv", 
        RUN_NAME = f"trial_{trial.number}", 
        DATASET = "AI-READI-FULL", 
        SEED = 0, 
        ITERS = 30000, 
        DATA_SIZE = 0.9, 
        NPY_PATH = f"AI-READI-FULL/preprocessed_data0.9_seed0/min_max_log.npy", 
        CAT_COLS = ['Age-related macular degeneration', 'Arthritis', 'Cancer', 'Cataracts (1+ eyes)', 'Chronic pullmonary problems', 'Circulation problems', 'Diabetic retinopathy (1+)', 'Digestive problems', 'Marijuana user', 'Dry eye (1+)', 'Glaucoma (1+)', 'Hearing impairment',
            'Heart attack', 'High blood cholesterol', 'High blood pressure', 'Kidney problems', 'Low blood pressure', 'Mild cognitive impairmen', 'Multiple sclerosis', 'Obesity', 'Osteoporosis','Other heart issues (pacemaker)', 'Other neurological conditions',
            "Parkinson's disease", 'Retinal vascular occlusion', 'Stroke', 'Type 2 Diabetes', 'Urinary problems', 
            'ABNORMAL', 'BORDERLINE', 'NORMAL', 'OTHERWISE NORMAL', 'cube_visuospatial_executive_value', 'fluency_language_value', 'lettera_value', 'trails_visuospatial_executive_value'
        ], 

        #same wearables, NEW measurements and NEW ekgs 
        NUM_COLS = [
            #wearable (same as ai-readi-og)
            'heart_rate_mean', 'blood_glucose_mean', 'resp_rate_mean', 'stress_mean', 'blood_glucose_std', 'total_steps',  'resting_heart_rate',
            'total_kcal', 'sleep_light_hrs', 'sleep_deep_hrs', 'sleep_rem_hrs', 'sleep_awake_hrs', 'act_generic_hrs', 'act_running_hrs','act_walking_hrs', 'act_sedentary_hrs', 
            #measurements: avoid a1c and the following columns with not enough data: [NT-proBNP (pg/mL)_value (pg/mL), Troponin-T (ng/L)_value (ng/L) ]
            'A/G Ratio_value', 'ALT (IU/L)_value (IU/L)', 'AST (IU/L)_value (IU/L)', 'Albumin (g/dL)_value (g/dL)', 'Alkaline Phosphatase (IU/L)_value (IU/L)', 'BUN (mg/dL)_value (mg/dL)', 'BUN/Creatinine ratio_value', "Bilirubin Total (mg/dL)_value (mg/dL)", 'C-Peptide (ng/mL)_value (ng/mL)', 
            'CRP - HS (mg/L)_value (mg/L)', 'Calcium (mg/dL)_value (mg/dL)', 'Carbon Dioxide, Total (mEq/L)_value (mEq/L)', 'Chloride (mEq/L)_value (mEq/L)', 'Creatinine (mg/dL)_value (md/dL)', 'Globulin, Total (g/dL)_value (g/dL)', 'Glucose (mg/dL)_value (mg/dL)', 'HDL Cholesterol (mg/dL)_value (mg/dL)',
            'INSULIN (ng/mL)_value (ng/mL)', 'LDL Cholesterol Calculation (mg/dL)_value (mg/dL)',  'Potassium (mEq/L)_value (mEq/L)', 'Protein, Total (g/dL)_value (g/dL)', 'Sodium (mEq/L)_value (mEq/L)', 'Total Cholesterol (mg/dL)_value (mg/dL)', 
            'Triglycerides (mg/dL)_value (mg/dL)',  'Urine Albumin (mg/dL)_value (mg/DL)', 'Urine Creatinine (mg/dL)_value (mg/DL)', 
            'clock_visuospatial_executive_time_value', 'delayed_recall_with_no_clue_time_value', 'digitspan_time_value', 'memory_trial1_time_value', 'memory_trial2_time_value', 'moca_abstraction_time_value', 'moca_orientation_time_value', 'moca_total_score_value', 'naming_time_value', 'repetition_time_value', 'subtraction_time_value',  'cube_visuospatial_executive_time_value', 'lettera_time_value', 'trails_visuospatial_executive_time_value',
            #multicategorical - try as numeric 
            'moca_combined_mis_score_value', #0 to 15
            'memory_trial1_value', #0 to 5 
            'memory_trial2_value', #2 to 5 
            'moca_abstraction_value', # 0 1 2 
            'moca_orientation_value', #3 to 6 
            'naming_value', # 0 to 3 
            'repetition_value', # 0 1 2 
            'subtraction_value', # 0 to 3 
            #ekgs 
            'Rate', 'PR', 'QRSD', 'QT', 'QTc', 'P', 'QRS', 'T',     
        ], 
        CAT_DIM = 36, 
        LABEL = "Type 2 Diabetes", 
        BATCH = BATCH, 
        NOISE_DIM = NOISE_DIM, 
        GRADIENT_PENALTY = GRADIENT_PENALTY, 
        G_LR = G_LR, 
        G_H = G_H, 
        D_LR = D_LR, 
        D_H = D_H, 
        DISC_STEPS = DISC_STEPS, 
        NUM_SAMPLES = 5000, 
        USE_TANH = USE_TANH, 
        PPO_EPOCHS = PPO_EPOCHS, 
        MEAN_PENALTY_SCALE = MEAN_PENALTY_SCALE, 
        VF_COEF = 0.5, 
        CLIP_EPS = 0.1, 
        ENT_BETA = 1e-3, 
        EPS  = 1e-6, 
    ) 
    os.makedirs(H.OUT_DIR, exist_ok=True)

    df_train = pd.read_csv(f"{H.DATA_PATH}/normalized_training_data.csv")[H.NUM_COLS + H.CAT_COLS]
    real = torch.tensor(df_train.values, dtype=torch.float32)
    loader = DataLoader(TensorDataset(real), batch_size=H.BATCH, shuffle=True, num_workers=0) 
    df_syn, elapsed_time = train(df_train, real, loader, H)

    #get raw to use for cwc, value stat analysis, histograms etc. 
    df_hold = pd.read_csv(f"{H.DATA_PATH}/original_testing_data.csv")[H.NUM_COLS+ H.CAT_COLS]
    df_real = pd.read_csv(f"{H.DATA_PATH}/original_data_with_patients.csv").drop(columns=['patient_id'])[H.NUM_COLS+H.CAT_COLS]
    df_syn = pd.read_csv(f"{H.OUT_DIR}/synthetic_rescaled.csv")[H.NUM_COLS+H.CAT_COLS]
    #get normalized data to use for classifications (need patients to split real data without leakage)
    df_hold_norm = pd.read_csv(f"{H.DATA_PATH}/normalized_testing_data.csv")[H.NUM_COLS+ H.CAT_COLS]
    df_real_with_patients_norm =  pd.read_csv(f"{H.DATA_PATH}/preprocessed_data_with_patients.csv")[H.NUM_COLS + H.CAT_COLS +["patient_id"]]
    df_syn_norm = pd.read_csv(f"{H.OUT_DIR}/synthetic.csv")[H.NUM_COLS+H.CAT_COLS]
    try:
        s2h_auc, r2s_auc = evaluate_model(df_real, df_hold, df_syn, df_hold_norm, df_syn_norm, df_real_with_patients_norm, H)          
        return s2h_auc, r2s_auc
    except Exception as e:
        raise optuna.TrialPruned(f"Pruned due to evaluation failure")
    


    #torch.cuda.empty_cache() 
    #torch.cuda.ipc_collect() 
    



def main(): 
    set_seed(0)
    base_dir = "RL_optuna_full_sun"
    os.makedirs(base_dir, exist_ok=True) 
    study = optuna.create_study(
        study_name="rl_optuna_full_sun",
        storage=f"sqlite:///rl_optuna_full_sun.db",       
        load_if_exists=True,
        directions= ("maximize", "maximize")
    )
    objectives = ["r2s_acc", "s2r_acc"]
    study.optimize(lambda trial: objective(trial), n_trials = 100)

    #best trial info 
    best_save = f"{base_dir}/best_trial_summary.txt"
    with open(best_save, "w") as f:
        pareto = study.best_trials
        f.write(f"Pareto front has {len(pareto)} trial(s)\n")
        for t in pareto: 
            f.write(f"{t.number},{t.values}, {t.params}\n")

    #PLOTS
    #pareto front: 
    fig = plot_pareto_front(study, target_names=objectives) 
    fig.write_html(f"{base_dir}/pareto_front.html")
    # 1D optimization history 
    for i, name in enumerate(objectives):
        fig = plot_optimization_history(study, target = lambda t, idx=i: t.values[idx], target_name = name)
        fig.write_html(f"{base_dir}/opt_history_{name}.html")
    #parameter importance plots per objective
    for i, name in enumerate(objectives):
        fig = plot_param_importances(study, target = lambda t, idx=i: t.values[idx], target_name = name)
        fig.write_html(f"{base_dir}/param_importance_{name}.html")
    

if __name__ == '__main__':
    main() 
