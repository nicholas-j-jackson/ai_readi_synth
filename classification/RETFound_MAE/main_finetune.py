import datetime
import json

import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from timm.data.mixup import Mixup

import util.lr_decay as lrd
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine_finetune import train_one_epoch, evaluate, extract
from models_vit import ExponentialMovingAverage

import warnings
import faulthandler

faulthandler.enable()
warnings.simplefilter(action='ignore', category=FutureWarning)


def main(args, criterion):

    os.environ["CUDA_VISIBLE_DEVICES"] = "0" #str(args.seed)
    import util.misc as misc

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Class labels for classifying T2D status
    if args.modality == 'fundus':
        classes = ['AMD', 'DR', 'GL']
    else:
        classes = ['AMD', 'DR', 'GL'] #['DR']

    from util.datasets import RetinalImageDataset
    dataset_train = RetinalImageDataset(args.data_path_train, args, classes, 'train', synth=args.synth_train)
    dataset_val = RetinalImageDataset(args.data_path_test, args, classes, 'val', synth=args.synth_test)
    dataset_test = RetinalImageDataset(args.data_path_test, args, classes, 'test', synth=args.synth_test)

    
    # Create logger
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.task))
    else:
        log_writer = None

    print(args)

    sampler_train, sampler_val, sampler_test = misc.get_distributed_samplers(args, dataset_train, dataset_val, dataset_test)

    # Define dataloaders
    # Use args.eval to determine whether train/val dataloaders are sequential
    data_loader_train = torch.utils.data.DataLoader( dataset_train, sampler=sampler_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
    data_loader_val = torch.utils.data.DataLoader( dataset_val, sampler=sampler_val, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, sampler=sampler_test, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)

    # Use mixup if applicable
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)


    # Load RETfound
    #
    #
    #model = misc.load_RETFound(args)
    #
    #
    #
    #

    import models_vit as models
    from timm.models.layers import trunc_normal_
    from util.pos_embed import interpolate_pos_embed
    import util.misc as misc
    if args.model=='RETFound_mae' or args.model == 'Bilateral_RETFound_mae':
        model = models.__dict__[args.model](
        img_size=args.input_size,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )
    else:
        raise NotImplementedError("Only RETFound or Bilateral RETFound are acceptable choices")
    
    # Load in RETFound checkpoint    
    if args.modality == 'fundus':
        checkpoint = torch.load("pretrained_weights/RETFound_mae_natureCFP.pth", map_location='cpu', weights_only=False)
    else: 
        checkpoint = torch.load("pretrained_weights/RETFound_mae_natureOCT.pth", map_location='cpu', weights_only=False)

    print("Load pre-trained checkpoint from: %s" % args.finetune)
    checkpoint_model = checkpoint['model']

    checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("mlp.w12.", "mlp.fc1."): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("mlp.w3.", "mlp.fc2."): v for k, v in checkpoint_model.items()}
    
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)

    #
    if args.finetune and not args.eval:
        trunc_normal_(model.head.weight, std=2e-5)

    # If performing evaluation from a checkpoint
    if args.ckpt and args.eval and not 'latent' in args.task:
        checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        print("Load checkpoint from: %s" % args.ckpt)
        model.load_state_dict(checkpoint['model'])


    if 'latent' in args.task:
        raise NotImplementedError("Fix this")

    #
    model.to(device)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of model params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # Distribute data
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module


    # Weight decay & optimizer
    no_weight_decay = model_without_ddp.no_weight_decay() if hasattr(model_without_ddp, 'no_weight_decay') else []
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                        no_weight_decay_list=no_weight_decay,
                                        layer_decay=args.layer_decay
                                        )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    # EMA
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)


    print("criterion = %s" % str(criterion))
    
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # Training
    #
    #
    #
    if not args.eval:
        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        max_score = 0.0
        best_epoch = 0

        # Iterate over epochs
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)

            # Train over one epoch
            train_stats = train_one_epoch( model, criterion, data_loader_train, optimizer, device, epoch, loss_scaler, ema, args.clip_grad, mixup_fn, log_writer=log_writer, args=args)

            # Compute
            ema.copy_to(model.parameters())  # use EMA weights
            val_stats, val_score = evaluate(data_loader_val, model, device, args, epoch, mode='val', num_class=args.nb_classes, log_writer=log_writer)
            if max_score < val_score:
                max_score = val_score
                best_epoch = epoch
                if args.output_dir and args.savemodel:
                    misc.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, mode='best')
            print("Best epoch = %d, Best score = %.4f" % (best_epoch, max_score))

            # On the last epoch, load best model checkpoint and evaluate
            if epoch == (args.epochs - 1):
                checkpoint = torch.load(os.path.join(args.output_dir, str(args.task) + '_' + str(args.seed), 'checkpoint-best.pth'), map_location='cpu', weights_only=False)
                model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
                model.to(device)
                print("Test with the best model, epoch = %d:" % checkpoint['epoch'])
                test_stats, auc_roc = evaluate(data_loader_test, model, device, args, -1, mode='test', num_class=args.nb_classes, log_writer=None)

            # Log results
            if log_writer is not None:
                log_writer.add_scalar('loss/val', val_stats['loss'], epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch,
                        'n_parameters': n_parameters}

            if args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, str(args.task) + '_' + str(args.seed), "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

        # Time
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    # Evaluation
    #
    #
    #
    else:
        # Extract latent representations & save
        if args.extract:
            extract(data_loader_train, model, device, args, mode='train', synth=args.synth_train)
            extract(data_loader_val, model, device, args, mode='val', synth=args.synth_test)
            extract(data_loader_test, model, device, args, mode='test', synth=args.synth_test)

        # Test best epoch
        elif 'epoch' in checkpoint:
            print("Test with the best model at epoch = %d" % checkpoint['epoch'])
            test_stats, auc_roc = evaluate(data_loader_test, model, device, args, epoch=checkpoint['epoch'], mode='test',
                                                num_class=args.nb_classes, log_writer=log_writer)
        
        # Evaluate on test set
        else: 
            test_stats, auc_roc = evaluate(data_loader_test, model, device, args, epoch=0, mode='test', num_class=args.nb_classes, log_writer=log_writer)

    

if __name__ == '__main__':
    args = misc.get_args_parser()

    criterion = torch.nn.BCEWithLogitsLoss()

    if args.output_dir:
        Path(os.path.join(args.output_dir, str(args.task) + '_' + str(args.seed))).mkdir(parents=True, exist_ok=True)
    main(args, criterion)


