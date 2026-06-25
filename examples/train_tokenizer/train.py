import gc
import os
import sys
import time
import glob
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from examples.train_tokenizer.trainer import Trainer
from models import build_wintok, build_discriminator
from models.modules import LPIPS
from data.dataset import build_clip_transforms, build_vae_transforms, load_data
from utils import config, dist
from utils.optimizer import build_optimizer
from utils.visualizer import setup_visualizer
from utils.scheduler import LRScheduler
from utils.logger import SmoothedValue, MetricLogger, ProfileLogger, setup_tensorboard_logger, tensorboard_log

from transformers import AutoTokenizer


def maybe_auto_resume(args: config.Args, pattern='ckpt*.pth'):
    if len(args.resume_from):
        resume = args.resume_from
        print(f'[auto_resume] load from args.resume @ {resume} ...')
    else:
        all_ckpt = glob.glob(os.path.join(args.output_dir, pattern), recursive=False)
        all_ckpt = sorted(all_ckpt, key=os.path.getmtime, reverse=True)
        if len(all_ckpt) == 0:
            resume = None
            print(f'[auto_resume] no ckpt found @ {pattern}')
            print(f'[auto_resume quit]')
        else:
            resume = all_ckpt[0]
            print(f'[auto_resume] auto load from @ {resume} ...')

    if resume is not None:
        try:
            ckpt = torch.load(resume, map_location='cpu')
            dist.barrier()
            resume_epoch = ckpt['epoch']
            resume_iter = ckpt['iter']
            if resume_epoch == args.epoch:
                print(f'[auto_resume] Training finished, skipping ...\n\n')
                exit()
            else:
                print(f'[auto_resume success] resume from ep{resume_epoch}, it{resume_iter}')
                return ckpt
        except Exception as e:
            print(f'[auto_resume] failed, {e} @ {resume}')
            return {}
    else:
        return {}

def train_one_ep(
    args,
    data,
    epoch,
    trainer,
    start_iter,
    unitok_scheduler,
    disc_scheduler,
    visualizer,
    logging_config
):
    dataloader = data['train'].dataloader
    num_iters = data['train'].num_batches

    metric_logger = MetricLogger(cur_epoch=epoch, total_epoch=args.epoch, delimiter='  ')
    [metric_logger.add_meter(x, SmoothedValue(window_size=1, fmt='{value:.2g}')) for x in ('glr', 'dlr')]
    [metric_logger.add_meter(x, SmoothedValue(window_size=1, fmt='{median:.2f}')) for x in ('gnm', 'dnm')]
    [metric_logger.add_meter(x, SmoothedValue(fmt='{median:.3f}')) for x in ('L1', 'Lnll', 'Ld', 'Lc', 'Wg')]

    disc_start_iter = args.disc_start_ep * num_iters
    disc_warmup_iter = args.disc_warmup_ep * num_iters
    
    eval_interval = int(num_iters // args.eval_per_epoch)
    profile_logger = ProfileLogger(args, eval_interval)

    for cur_iter, sample in metric_logger.monitor_enumerate(dataloader, start_iter, num_iters, print_freq=args.print_freq):
        profile_logger.log(cur_iter)

        imgs, texts = sample
        imgs = imgs.to(args.device, non_blocking=True)
        texts = texts.to(args.device, non_blocking=True)

        global_iter = epoch * num_iters + cur_iter
        disc_global_iter = global_iter - disc_start_iter

        unitok_lr_stats = unitok_scheduler.step(global_iter)
        disc_lr_stats = disc_scheduler.step(disc_global_iter) if disc_global_iter >= 0 else [0]
        unitok_lr_stats = list(set(unitok_lr_stats))
        disc_lr_stats = list(set(disc_lr_stats))

        stepping = (global_iter + 1) % args.grad_accu == 0
        warmup_disc_schedule = 0 if disc_global_iter < 0 else min(1.0, disc_global_iter / disc_warmup_iter)
        fade_blur_schedule = 0 if disc_global_iter < 0 else min(1.0, disc_global_iter / (disc_warmup_iter * 2))
        fade_blur_schedule = 1 - fade_blur_schedule

        trainer.train_step(
            img=imgs,
            text=texts,
            global_iter=global_iter,
            stepping=stepping,
            metric_logger=metric_logger,
            warmup_disc_schedule=warmup_disc_schedule,
            fade_blur_schedule=fade_blur_schedule,
            logging_config=logging_config
        )

        metric_logger.update(glr=max(unitok_lr_stats))
        metric_logger.update(dlr=max(disc_lr_stats))

        if logging_config['use_tensorboard']:
            for i, lr in enumerate(unitok_lr_stats):
                name = 'Param_unitok_group_{}_lr'.format(i)
                tensorboard_log(tb_logger=logging_config['tb_logger'], data={name: lr}, step=global_iter, log_freq=200)
            for i, lr in enumerate(disc_lr_stats):
                name = 'Param_disc_group_{}_lr'.format(i)
                tensorboard_log(tb_logger=logging_config['tb_logger'], data={name: lr}, step=global_iter, log_freq=200)

        if (cur_iter + 1) % eval_interval == 0:
            if dist.is_master():
                vis_path = os.path.join(args.output_dir, f'img_{global_iter}.png')
                visualizer.vis(cur_iter, png_path=vis_path, tb_logger=logging_config['tb_logger'])

            if dist.is_master():
                ckpt_path = os.path.join(args.output_dir, f'ckpt-ep{epoch}-iter{cur_iter}.pth')
                torch.save({
                    'args': args.state_dict(),
                    'epoch': epoch, 'iter': cur_iter,
                    'trainer': trainer.state_dict(),
                }, ckpt_path)

            dist.barrier()

    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats


def main():
    args = config.init_dist_and_get_args()
    print(f'[args] initial args:\n{str(args)}')

    # resume ckpt
    ckpt = maybe_auto_resume(args, 'ckpt*.pth')
    start_iter = ckpt.get('iter', 0)
    start_epoch = ckpt.get('epoch', 0)
    trainer_state = ckpt.get('trainer', {})

    # load data
    print(f'[data] load data...\n')
    aug_cfg = {'scale': [0.64, 1.0]} if args.use_aug else None
    preprocess_fns = build_clip_transforms(args, aug_cfg=aug_cfg)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data = load_data(args, preprocess_fns, epoch=start_epoch, iters=start_iter, tokenizer=tokenizer)

    # build models
    unitok = build_wintok(args)
    disc = build_discriminator(args)

    print(f'[model] UniTok #paras {sum(p.numel() for p in unitok.parameters()) / 1e6:.2f}')
    print(f'[model] Disc #paras {sum(p.numel() for p in disc.parameters()) / 1e6:.2f}')

    # build optimizers & scheduler
    unitok_optim = build_optimizer(args, 'weunitok', unitok)
    disc_optim = build_optimizer(args, 'disc', disc)

    max_iter = args.epoch * data['train'].num_batches
    warmup_iter = args.warmup_ep * data['train'].num_batches
    disc_max_iter = max_iter - args.disc_start_ep * data['train'].num_batches
    disc_warmup_iter = args.disc_warmup_ep * data['train'].num_batches

    unitok_schedule = {
        'lr': args.lr,
        'type': args.schedule,
        'start_factor': args.lr_start_ratio,
        'end_factor': args.lr_end_ratio,
        'warmup_iter': warmup_iter,
        'max_iter': max_iter,
    }
    disc_schedule = {
        'lr': args.disc_lr,
        'type': args.schedule,
        'start_factor': args.lr_start_ratio,
        'end_factor': args.disc_lr_end_ratio,
        'warmup_iter': disc_warmup_iter,
        'max_iter': disc_max_iter,
    }
    unitok_scheduler = LRScheduler(unitok_optim.optimizer, unitok_schedule)
    disc_scheduler = LRScheduler(disc_optim.optimizer, disc_schedule)

    # build loss

    lpips_loss: LPIPS = LPIPS(args.lpips_path).to(args.device)

    # torch compile model
    if args.compile_model:
        unitok = torch.compile(unitok, backend='inductor')
        disc = torch.compile(disc, backend='inductor')
        lpips_loss = torch.compile(lpips_loss, backend='inductor')

    # distributed wrapper
    unitok = DDP(unitok, device_ids=[dist.get_local_rank()], static_graph=args.ddp_static)
    disc = DDP(disc, device_ids=[dist.get_local_rank()], static_graph=args.ddp_static)

    # build trainer
    trainer = Trainer(
        args=args,
        unitok=unitok,
        disc=disc,
        unitok_optim=unitok_optim,
        disc_optim=disc_optim,
        lpips_loss=lpips_loss,
    )
    if trainer_state:
        trainer.load_state_dict(trainer_state, strict=True)

    # setup visualizer
    vis_transform = build_vae_transforms(args)[1]
    visualizer = setup_visualizer(args, trainer, vis_transform)

    # setup tensorboard
    tb_logger = setup_tensorboard_logger(args)
    if args.use_tensorboard and dist.is_master():
        print(f'[tensorboard] logging to {args.tensorboard_log_dir}/{args.exp_name}')
        
    # setup logging info for trainer
    logging_config = {
        'use_tensorboard': args.use_tensorboard,
        'tb_logger': tb_logger
    }

    # train
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    print(f'[train] exp output directory: {args.output_dir}')
    print(f'[train] start exp at epoch {start_epoch} iter {start_iter}')

    for epoch in range(start_epoch, args.epoch):
        print(f'[dataloader] set_epoch({epoch})]')
        data['train'].set_epoch(epoch)

        start_iter = start_iter if epoch == start_epoch else 0

        stats = train_one_ep(
            args=args,
            data=data,
            epoch=epoch,
            trainer=trainer,
            start_iter=start_iter,
            unitok_scheduler=unitok_scheduler,
            disc_scheduler=disc_scheduler,
            visualizer=visualizer,
            logging_config=logging_config
        )

    if dist.is_master():
        ckpt_path = os.path.join(args.output_dir, 'ckpt-last.pth')
        torch.save({
            'args': args.state_dict(),
            'epoch': args.epoch, 'iter': 0,
            'trainer': trainer.state_dict(),
        }, ckpt_path)
    dist.barrier()

    total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    print(f"[train] Total Training Time: {total_time},\t Lg: {stats['Lnll']:.3f},\t Ld: {stats['Ld']:.3f}")

    if isinstance(sys.stdout, dist.BackupStreamToFile) and isinstance(sys.stderr, dist.BackupStreamToFile):
        sys.stdout.close(), sys.stderr.close()


if __name__ == '__main__':
    main()
