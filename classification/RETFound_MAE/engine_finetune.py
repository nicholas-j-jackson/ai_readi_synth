import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Optional
from timm.data import Mixup
from timm.utils import accuracy
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, average_precision_score,
    hamming_loss, jaccard_score, recall_score, precision_score, cohen_kappa_score
)
from pycm import ConfusionMatrix
import util.misc as misc
import util.lr_sched as lr_sched
from tqdm import tqdm

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    ema,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None
    ):
    """Train the model for one epoch."""
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print_freq, accum_iter = 20, args.accum_iter
    optimizer.zero_grad()
    
    if log_writer:
        print(f'log_dir: {log_writer.log_dir}')
    
    # Iterate over dataloader
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, f'Epoch: [{epoch}]')):
        # Adjust LR at first step of each epoch
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # Unilateral model
        if 'img' in batch.keys():        
            samples = batch['img']
            targets = batch['label']
            samples, targets = samples.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            if mixup_fn:
                samples, targets = mixup_fn(samples, targets)
            
            with torch.cuda.amp.autocast():
                outputs = model(samples)
                loss = criterion(outputs, targets)

        # Bilateral model
        else:
            left, right = batch['img_l'].to(device, non_blocking=True), batch['img_r'].to(device, non_blocking=True)
            targets = batch['label'].to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                outputs = model(left, right)
                loss = criterion(outputs, targets)

        # Update & log
        loss_value = loss.item()
        loss /= accum_iter
        
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            ema.update(model.parameters())
            optimizer.zero_grad()
        
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss/train', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
    
    # Synchronize between processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args, epoch, mode, num_class, log_writer):
    if args.modality == 'fundus':
        classes = ['AMD', 'DR', 'GL']
    else:
        classes = ['AMD', 'DR', 'GL'] #['DR']

    """Evaluate the model."""
    criterion = nn.BCEWithLogitsLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    os.makedirs(os.path.join(args.output_dir, str(args.task) + '_' + str(args.seed)), exist_ok=True)
    
    model.eval()
    true_onehot, pred_onehot, true_labels, pred_labels, pred_softmax = [], [], [], [], []
    
    # Iterate over dataset
    for batch in metric_logger.log_every(data_loader, 10, f'{mode}:'):

        # Unilateral model
        if 'img' in batch.keys():
            images, target = batch['img'].to(device, non_blocking=True), batch['label'].to(device, non_blocking=True)        
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)

        # Bilateral Model
        else:
            left, right = batch['img_l'].to(device, non_blocking=True), batch['img_r'].to(device, non_blocking=True)
            target = batch['label'].to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                output = model(left, right).reshape(-1,num_class)
                loss = criterion(output, target)


        # Apply sigmoid to get probabilities per class
        output_sigmoid = torch.sigmoid(output) 

        # Logging and metric prep
        metric_logger.update(loss=loss.item())
        true_onehot.extend(target.cpu().numpy())
        pred_softmax.extend(output_sigmoid.detach().cpu().numpy())

    true_onehot = np.array(true_onehot)
    pred_softmax = np.array(pred_softmax)

    # multi-label AUC
    roc_auc = {f'roc_auc_{clss}': roc_auc_score(true_onehot[:, i], pred_softmax[:, i]) for i, clss in enumerate(classes)}
    
    # Log results
    if log_writer:
        for metric_name, value in zip(roc_auc.keys(), roc_auc.values()):
            log_writer.add_scalar(f'perf/{metric_name}', value, epoch)
    
    print(f'val loss: {metric_logger.meters["loss"].global_avg}')
    for i, clss in enumerate(classes):
        k = f'roc_auc_{clss}'
        print(f'{k}: {roc_auc[k]:.4f}')
    
    metric_logger.synchronize_between_processes()
    
    # Save results to csv
    results_path = os.path.join(args.output_dir, str(args.task) + '_' + str(args.seed), f'metrics_{mode}.csv')
    file_exists = os.path.isfile(results_path)
    with open(results_path, 'a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        if args.modality == 'fundus' or args.modality == 'oct':
            if not file_exists:
                wf.writerow(['val_loss', 'roc_auc_AMD', 'roc_auc_DR', 'roc_auc_GL'])
            wf.writerow([metric_logger.meters["loss"].global_avg, roc_auc['roc_auc_AMD'], roc_auc['roc_auc_DR'], roc_auc['roc_auc_GL']])
        else:
            if not file_exists:
                wf.writerow(['val_loss', 'roc_auc_DR'])
            wf.writerow([metric_logger.meters["loss"].global_avg, roc_auc['roc_auc_DR']])

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, np.mean(list(roc_auc.values()))


@torch.no_grad()
def extract(data_loader, model, device, args, mode='train', synth=False, fid=False):
    """Extract latent """
    model.eval()
    out = []

    # Iterate over dataset
    for batch in tqdm(data_loader):
        images = batch['img'].to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            #output = model(images)
            output = model.forward_features(images)

        out.append(output)
    
    out = torch.concat(out, axis=0)
    
    # Save these extracted features
    import pandas as pd
    out = pd.DataFrame(out.cpu().squeeze().numpy(), columns=['feat_' + str(x) for x in range(out.shape[1])])
    df = data_loader.dataset.df

    print(out.shape, df.shape)
    out.index = df.index

    df = pd.concat([df, out], axis=1)

    print(df.shape)
    suffix = '_fid' if fid else '_latents'

    if args.modality == 'fundus':
        if synth:
            df.to_csv('/data/7TB/nick/synth_ai_readi_fundus/retinal_photography/{}{}.csv'.format(mode, suffix))
        else:
            df.to_csv('/data/7TB/nick/ai_readi_v3/resized_retinal_photography/{}{}.csv'.format(mode, suffix))
    else:
        if synth:
            df.to_csv('/data/7TB/nick/synth_ai_readi_oct/retinal_oct/{}{}.csv'.format(mode, suffix))
        else:
            df.to_csv('/data/7TB/nick/ai_readi_v3/resized_retinal_oct/{}{}.csv'.format(mode, suffix))

