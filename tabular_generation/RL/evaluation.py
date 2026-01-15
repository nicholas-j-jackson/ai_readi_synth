
import pandas as pd
import matplotlib.pyplot as plt
import os 
from pathlib import Path
import numpy as np 
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold
from numpy.linalg import norm
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score)
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
import seaborn as sns
import re
import math 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import metrics
import csv 

#------------------------------------------------------------RL MISCELLANEOUS--------------------------------------------------------------# 
#plot losses
def plot_rl_losses(out_dir):
    out_dir = Path(out_dir)
    log_pat = re.compile(r"iteration\s+(\d+)\s*\|\s*D LOSS\s*=\s*([-\d.]+)\s*\|\s*G LOSS\s*=\s*([-\d.]+)\s*\|\s*AVG R\s*=?\s*([-\d.]+)")
    rows = []
    with open(out_dir/"losses/output.txt") as fh:
        for ln in fh:
            m = log_pat.search(ln)
            if m: rows.append([int(m[1]), *map(float, m.groups()[1:])])
    df_log = pd.DataFrame(rows, columns=["iter","d_loss","g_loss","avg_reward"]).sort_values("iter")
    if df_log.empty: 
        df_log = pd.DataFrame(columns=["iter","d_loss","g_loss","avg_reward"])
    g_pat = re.compile(r"ITERATION\s+([0-9]+(?:\.[0-9]+)?)\s*\|\s*LOSS PI\s+([-\d.]+)\s*\|\s*LOSS V\s+([-\d.]+)\s*\|\s*ENTROPY\s+([-\d.]+)\s*\|\s*MEAN PEN.*?\s+([-\d.]+)\s*\|\s*TOTAL G LOSS\s+([-\d.]+)")
    g_rows = []
    with open(out_dir/"losses/G_loss.txt") as fh:
        for ln in fh:
            m = g_pat.search(ln)
            if m: g_rows.append([float(x) for x in m.groups()])
    df_g = pd.DataFrame(g_rows, columns=["iter","loss_pi","loss_v","entropy","mean_pen","g_total"])
    if not df_g.empty:
        df_g["iter"] = df_g["iter"].round().astype(int)
        df_g["entropy"] = df_g["entropy"]
        df_g = df_g.groupby("iter", as_index=False).mean()
        df_g["bin25"] = (df_g["iter"]//25)*25
        df_g25 = df_g.groupby("bin25", as_index=False).mean()
    else:
        df_g25 = pd.DataFrame(columns=["bin25","loss_pi","loss_v","entropy","mean_pen","g_total"])
    df50 = df_log.copy()
    if not df50.empty:
        df50["bin50"] = (df50["iter"]//50)*50
        df50 = df50.groupby("bin50", as_index=False).mean()
    fig, axs = plt.subplots(2,2, figsize=(10,7))
    if not df50.empty:
        axs[0,0].plot(df50["bin50"], df50["d_loss"])
        axs[0,0].set_title("D loss")
        axs[0,1].plot(df50["bin50"], df50["g_loss"])
        axs[0,1].set_title("G loss")
        axs[1,0].plot(df50["bin50"], df50["d_loss"], label="D")
        axs[1,0].plot(df50["bin50"], df50["g_loss"], label="G")
        axs[1,0].legend()
        axs[1,0].set_title("D vs G")
    if not df_log.empty:
        axs[1,1].plot(df_log["iter"], df_log["avg_reward"])
        axs[1,1].set_title("Avg reward")
    for ax in axs.ravel(): ax.set_xlabel("iteration")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir/"losses/losses.png", dpi=300)
    plt.close(fig)
    fig, axs = plt.subplots(3,2, figsize=(12,8), sharex=True)
    comps = ["loss_pi","loss_v","entropy","mean_pen","g_total"]
    for ax, c in zip(axs.ravel(), comps):
        if not df_g25.empty: ax.plot(df_g25["bin25"], df_g25[c])
        ax.set_title(c)
        ax.grid(True, linewidth=0.3, alpha=0.5)
    axs.ravel()[-1].axis("off")
    axs[-1,0].set_xlabel("iteration")
    axs[-1,1].set_xlabel("iteration")
    fig.suptitle("Generator components (avg every 25 iters)", y=0.98)
    fig.tight_layout()
    fig.savefig(out_dir/"losses/G_loss.png", dpi=300)
    plt.close(fig)

def log_result_RL(RESULTS_CSV, tag, iters, data_size, seed, r2r_auc, s2h_auc, real_mem_auc, synth_mem_auc, elapsed_time):
    p = Path(RESULTS_CSV)
    p.parent.mkdir(parents=True, exist_ok=True)
    r2r_dict = {f"r2r_auc_{x}" : r2r_auc[x] for x in r2r_auc.keys()}
    s2r_dict = {f"s2r_auc_{x}" : s2h_auc[x] for x in s2h_auc.keys()}
    row = {"tag": tag, "iters": iters, "data_size" : data_size, "seed" : seed, "real_mem_auc" : real_mem_auc,  "synth_mem_auc" : synth_mem_auc, "elapsed_time" : elapsed_time}
    row = {**row, **r2r_dict}
    row = {**row, **s2r_dict}
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def log_result_RL_search(RESULTS_CSV, tag, iters, data_size, seed, s2h_auc, s2h_acc, r2s_auc, r2s_acc):
    p = Path(RESULTS_CSV)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"tag": tag, "iters": iters, "data_size" : data_size, "seed" : seed, "s2h_auc" : s2h_auc, "s2h_acc" : s2h_acc, "r2s_auc" : r2s_auc, "r2s_acc" : r2s_acc}
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


#------------------------------------------------------------UTILITY ANALYSIS------------------------------------------------------------# 
#helpers 
def _ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
def _svm(rs):
    return SVC(kernel="rbf", probability=True, C=1.0, gamma="scale", random_state=rs)
def _interp(mean_fpr, fpr, tpr):
    x = np.interp(mean_fpr, fpr, tpr)
    x[0] = 0.0
    return x
def _append(fp, s):
    with Path(fp).open("a") as f: f.write(s)
def _save_roc(tprs, mean_fpr, aucs, title, path):
    plt.figure(figsize=(6,5))
    for t in tprs: plt.plot(mean_fpr, t, color="grey", alpha=0.3)
    mt, st = np.mean(tprs,0), np.std(tprs,0)
    plt.plot(mean_fpr, mt, label=f"Mean ROC (AUC = {np.mean(aucs):.3f})")
    plt.fill_between(mean_fpr, np.maximum(mt-st,0), np.minimum(mt+st,1), color="blue", alpha=0.2, label="±1 SD")
    plt.plot([0,1],[0,1],"k--",lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels, title, out_path):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(5,4))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def predict(df_syn, df_hold, out_dir, random_state, label_col="Type 2 Diabetes", n_splits=5):
    out_dir = _ensure_dir(out_dir)
    Xtr, ytr = df_syn.drop(columns=[label_col]), (df_syn[label_col] >= 0.5).astype(int)
    Xv, yv = df_hold.drop(columns=[label_col]), df_hold[label_col].astype(int)
    mean_fpr = np.linspace(0,1,100)
    tprs, aucs, accs = [], [], []
    y_true_all, y_pred_all = [], []
    (out_dir/"results.txt").write_text(f"--------5-FOLD: TRAIN ON SYNTHETIC / TEST ON HOLD--------\nTarget distribution (hold):\n{yv.value_counts()}\n\n")
        
    clf = _svm(random_state).fit(Xtr, ytr)
    proba = clf.predict_proba(Xv)[:,1]
    pred = (proba >= 0.5).astype(int)
    y_true_all.extend(yv)
    y_pred_all.extend(pred)
    auc = roc_auc_score(yv, proba)
    acc = accuracy_score(yv, pred)
    prec = precision_score(yv, pred)
    rec = recall_score(yv, pred)
    f1 = f1_score(yv, pred)
    
    aucs.append(auc)
    accs.append(acc)

    _append(out_dir/"results.txt", f"AUC={auc:.4f}  Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}\n")
    fpr, tpr, _ = roc_curve(yv, proba)
    tprs.append(_interp(mean_fpr, fpr, tpr))
    
    _append(out_dir/"results.txt", "\nCross-validation summary\nMean AUC : %.4f ± %.4f\nMean Acc : %.3f ± %.3f\n" % (np.mean(aucs), np.std(aucs), np.mean(accs), np.std(accs)))
    _save_roc(tprs, mean_fpr, aucs, "ROC – Train Synth, Test Hold", out_dir/"roc_synth2hold.png")
    plot_confusion_matrix(y_true_all, y_pred_all, labels=["No Diabetes", "Type 2 Diabetes"], title="Confusion Matrix – Synth Train / Hold Test", out_path=out_dir / "confusion_matrix.png")
    return np.mean(aucs), np.mean(accs)

#------------------------------------------------------------PRIVACY ANALYSIS------------------------------------------------------------# 

#helpers 
def find_replicant(real, fake):
    #a = (square every elemnt in synthetic matrix and sum across features into col vector) + (square and sum real into row vector )
    #results in (n_fake x n_real) matrix where every (i, j) holds fake_i^2 + real_j^2
    a = np.sum(fake ** 2, axis=1).reshape(fake.shape[0], 1) + np.sum(real.T ** 2, axis=0)
    #each entry (i, j) is 2 (fake I, realIj)
    b = np.dot(fake, real.T) * 2
    #| x - y | ^2 = |x|^2 | y|^2 (full squared euclidean distance matrix between every fake real pair)
    distance_matrix = a - b
    #for every real row j, find the closest fake sample (the min over i), take the sqrt to get euclidean distance --> 1D (n_real)
    #the function returns, for every real record, the distance to its nearest synthetic replicant 
    return np.sqrt(np.min(distance_matrix, axis=0))

def each_group(model, batchsize, n_train, n_test, n_cont_col, model_id, train, test, fake, theta):
    distance_train = np.zeros(n_train)
    distance_test = np.zeros(n_test)
    #for the synethic: for each batch slice of data, compute nearest-distance from that real batch to all fake records
    if model_id != 'real':
        steps = np.ceil(n_train / batchsize)
        for i in range(int(steps)):
            distance_train[i * batchsize:(i + 1) * batchsize] = find_replicant(train[i * batchsize:(i + 1) * batchsize], fake)
    #do the same for test 
    steps = np.ceil(n_test / batchsize)
    for i in range(int(steps)):
        distance_test[i * batchsize:(i + 1) * batchsize] = find_replicant(test[i * batchsize:(i + 1) * batchsize], fake)
    #true positives: real-train rows whose nearest synthetic neighbour is within the radius theta 
    n_tp = np.sum(distance_train <= theta) 
    #false negatives: real-train rows not captured inside theta 
    n_fn = n_train - n_tp
    #false positive counts: hold out rows that also fall within theta (attacker thought they were in training)
    n_fp = np.sum(distance_test <= theta) 
    #F1 score
    f1 = n_tp / (n_tp + (n_fp + n_fn) / 2)  
    return f1, n_tp, n_fn, n_fp 

def _append(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f: f.write(text)

def _batched_min_dists(X, fake, batchsize):
    d = np.empty(len(X))
    for i in range(0, len(X), batchsize):
        d[i:i+batchsize] = find_replicant(X[i:i+batchsize], fake)
    return d

def _plot_rates(theta, tpr, fpr, title, path):
    plt.figure(figsize=(5, 4))
    plt.plot(theta, tpr, label="TPR (recall)")
    plt.plot(theta, fpr, label="FPR")
    plt.xlabel("θ (distance threshold)"); plt.ylabel("Rate"); plt.title(title)
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()

def _plot_roc(fpr, tpr, title_tpl, label, path):
    order = np.argsort(fpr); auc_val = metrics.auc(fpr[order], tpr[order])
    plt.figure(figsize=(4.5, 4.5))
    plt.plot(fpr, tpr, label=f"{label} (AUC {auc_val:.3f})")
    plt.plot([0,1],[0,1],"--", lw=0.8, label="random guess")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title_tpl.format(auc=auc_val))
    plt.legend(loc="lower right"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    return auc_val

def mem_risk(train_df, test_df, synth_df, CAT_COLS, NUM_COLS, OUT_DIR, GLOBAL_SEED): 
    OUT_DIR = Path(OUT_DIR); OUT_DIR.mkdir(parents=True, exist_ok=True)
    BAL_DIR = OUT_DIR / "balanced"; BAL_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "auc.txt").write_text("Mem risk AUCS\n")
    (OUT_DIR / "mem_risk.txt").write_text("Mem risk results\n")
    #just double check all binary cols 0 or 1 
    for df in (train_df, test_df, synth_df):
        df[CAT_COLS] = (df[CAT_COLS] >= 0.5).astype(float)
    ret = {}
    for model_id in ["real", "synth"]:
        _append(OUT_DIR / "mem_risk.txt", f"Model {model_id} results\n")
        thetas = np.round(np.linspace(0.05, 8, 200), 2)
        tpr_raw, fpr_raw, tpr_bal, fpr_bal, theta_list = [], [], [], [], []
        train, test = train_df.values, test_df.values
        fake = train.copy() if model_id == "real" else synth_df.values
        n_train, n_test = len(train), len(test)
        n_cont_col, batchsize = len(NUM_COLS), 1000
        for theta in thetas:
            risk, n_tp, n_fn, n_fp = each_group(model_id.lower(), batchsize, n_train, n_test, n_cont_col, model_id, train, test, fake, float(theta))
            adv = (n_tp / n_train) - (n_fp / n_test)
            _append(OUT_DIR / "mem_risk.txt", f"model_id={model_id}   θ={theta:.2f}   Membership-risk F1 = {risk:.3f} TP: {n_tp} FN: {n_fn} FP {n_fp} Advantage: {adv:.3f} \n")
            tpr_raw.append(n_tp / n_train)
            fpr_raw.append(n_fp / n_test)
            theta_list.append(theta)
            n_bal = min(n_train, n_test)
            #balance 
            train_bal = (train_df.sample(n=n_bal, random_state=GLOBAL_SEED) if n_train > n_bal else train_df).values
            test_bal = (test_df .sample(n=n_bal, random_state=GLOBAL_SEED) if n_test  > n_bal else test_df ).values
            d_train = _batched_min_dists(train_bal, fake, batchsize)
            d_test = _batched_min_dists(test_bal,  fake, batchsize)
            tpb = (d_train <= theta).sum()
            fnb = n_bal - tpb
            fpb = (d_test <= theta).sum()
            tpr_bal.append(tpb / n_bal); fpr_bal.append(fpb / n_bal)
            f1b = tpb / (tpb + (fpb + fnb) / 2)
            _append(BAL_DIR / "mem_risk_balanced.txt", f"model_id={model_id} θ={theta:.2f} F1: {f1b:.3f} TP_bal:{tpb} FN_bal:{fnb} FP_bal:{fpb}\n")
        #raw metrics 
        tpr_raw, fpr_raw, th = np.asarray(tpr_raw), np.asarray(fpr_raw), np.asarray(theta_list)
        _plot_rates(th, tpr_raw, fpr_raw, f"TPR / FPR vs θ – {model_id}", OUT_DIR / f"tpr_fpr_vs_theta_{model_id}.png")
        auc_raw = _plot_roc(fpr_raw, tpr_raw, "Membership-Inference ROC (AUC = {auc:.4f})", model_id, OUT_DIR / f"roc_{model_id}.png")
        print(f"[{model_id}] ROC-AUC = {auc_raw:.4f}\n"); _append(OUT_DIR / "auc.txt", f"[{model_id}] ROC-AUC = {auc_raw:.4f}\n")
        ret[f"{model_id}_roc_auc"] = auc_raw
        #balanced metrix 
        tpr_bal, fpr_bal = np.asarray(tpr_bal), np.asarray(fpr_bal)
        _plot_rates(th, tpr_bal, fpr_bal, f"TPR / FPR vs θ BAL – {model_id}", BAL_DIR / f"tpr_fpr_vs_theta_{model_id}.png")
        auc_bal = _plot_roc(fpr_bal, tpr_bal, "Membership-Inference ROC BAL (AUC = {auc:.4f})", model_id, BAL_DIR / f"roc_{model_id}.png")
        print(f"[{model_id}] ROC-AUC BAL = {auc_bal:.4f}"); _append(OUT_DIR / "auc.txt", f"[{model_id}] ROC-AUC BAL = {auc_bal:.4f}\n")
        ret[f"{model_id}_roc_auc_bal"] = auc_bal
        np.savez(BAL_DIR / f"tpr_fpr_vs_theta_{model_id}.npz", theta=th, TPR=tpr_bal, FPR=fpr_bal)
        #alpha  
        ADV_DIR = OUT_DIR / "advantages_at_alpha"; ADV_DIR.mkdir(exist_ok=True)
        for a in range(1, 101):
            alpha = a * 0.01
            idx = np.where(fpr_raw <= alpha)[0]
            if idx.size:
                best = idx[np.argmax(th[idx])]
                adv = tpr_raw[best] - fpr_raw[best]
                ppv = tpr_raw[best] / (tpr_raw[best] + fpr_raw[best])
                _append(ADV_DIR / f"advantage_{model_id}.txt", f"α={alpha:.2%}, θ*={th[best]:.2f}, TPR={tpr_raw[best]:.4f}, FPR={fpr_raw[best]:.4f}, PPV={ppv:.4f}, Advantage={adv:.4f}\n")
            else:
                _append(ADV_DIR / f"advantage_{model_id}.txt", f"No θ achieves FPR ≤ {alpha:.2%}\n")
    return ret
