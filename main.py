import torch
from torch import nn
from torch.optim import AdamW
from torch.nn import functional as F
from tqdm import tqdm
import timm
from timm.models import create_model
from timm.scheduler.cosine_lr import CosineLRScheduler
from argparse import ArgumentParser
from vtab import *
from utils import save, load, load_config, set_seed, QLinear, AverageMeter, log
import adaptformer
import lora
import fpet
import evidence
from torch.utils.tensorboard import SummaryWriter
import time
import datetime
import json


def consistency_loss(args, model, x, y):
    """
    Merge-consistency objective: CE on the merged branch, CE on the full-token branch and
    a KL term distilling the (detached) full-token prediction into the merged branch.
    """
    out, out_full, feat, feat_full = model.forward_two_branch(x)
    ce = F.cross_entropy(out, y)
    ce_full = F.cross_entropy(out_full, y)
    T = args.cons_temp
    kl = F.kl_div(F.log_softmax(out / T, dim=-1), F.softmax(out_full.detach() / T, dim=-1),
                  reduction='batchmean') * T * T
    loss = ce + args.full_ce_weight * ce_full + args.cons_weight * kl
    parts = {'ce': ce.item(), 'ce_full': ce_full.item(), 'kl': kl.item()}
    if args.feat_weight > 0:
        feat_loss = (1 - F.cosine_similarity(feat, feat_full.detach(), dim=-1)).mean()
        loss = loss + args.feat_weight * feat_loss
        parts['feat'] = feat_loss.item()
    return loss, parts


def run_name(model_path):
    # model_path is <root>/<run> or <root>/<run>/seed<k>
    parts = os.path.normpath(model_path).split(os.sep)
    return parts[-2] if parts[-1].startswith('seed') and len(parts) > 1 else parts[-1]


def train(args, model, dl, opt, scheduler, epoch, log_writer):
    model.train()
    model = model.cuda()
    torch.cuda.reset_peak_memory_stats()
    pbar = tqdm(range(epoch))
    start_train = time.time()
    step = 0
    for ep in pbar:
        model.train()
        model = model.cuda()
        meters = {}
        for i, batch in enumerate(dl):
            x, y = batch[0].cuda(), batch[1].cuda()
            if args.consistency and step % args.full_every == 0:
                loss, parts = consistency_loss(args, model, x, y)
            else:
                out = model(x)
                loss = F.cross_entropy(out, y)
                parts = {'ce': loss.item()}
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            for k, v in parts.items():
                meters.setdefault(k, []).append(v)
        if scheduler is not None:
            scheduler.step(ep)
        if ep % args.eval_every == args.eval_every - 1:
            end_train = time.time()
            start_test = time.time()
            acc = test(model, test_dl)
            end_test = time.time()
            acc_full = test_full(model, test_dl) if args.full_eval else None
            train_time = (end_train-start_train)/(len(dl)*args.eval_every)
            test_time = (end_test-start_test)/len(test_dl)
            log(args, acc, train_time, test_time, loss.item(), ep)
            log_writer.add_scalar('acc', acc, ep)
            log_writer.add_scalar('train_time', train_time, ep)
            log_writer.add_scalar('test_time', test_time, ep)
            log_writer.add_scalar('loss', loss.item(), ep)
            for k, v in meters.items():
                log_writer.add_scalar(f'loss_{k}', sum(v) / len(v), ep)
            if acc_full is not None:
                log_writer.add_scalar('acc_full', acc_full, ep)
            log_writer.flush()
            args.history.append({'epoch': ep, 'acc': acc, 'acc_full': acc_full,
                                 'train_time_per_it': train_time, 'test_time_per_it': test_time,
                                 **{f'loss_{k}': sum(v) / len(v) for k, v in meters.items()}})
            if acc > args.best_acc:
                args.best_acc = acc
                args.best_ep = ep
                args.best_acc_full = acc_full
                save(args, model)
            pbar.set_description('best_acc ' + str(args.best_acc))
            model = model.cuda()
            start_train = time.time()

    args.peak_train_mem_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    model = model.cpu()
    return model


@torch.no_grad()
def test(model, dl):
    model.eval()
    acc = AverageMeter()
    model = model.cuda()
    for batch in tqdm(dl):
        x, y = batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        acc.update(out, y)
    return acc.result().item()


@torch.no_grad()
def test_full(model, dl):
    """Accuracy of the same weights without token merging (full-token branch)."""
    model.eval()
    acc = AverageMeter()
    model = model.cuda()
    for batch in tqdm(dl):
        x, y = batch[0].cuda(), batch[1].cuda()
        out = model.forward_two_branch(x)[1].data
        acc.update(out, y)
    return acc.result().item()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bit', type=int, default=1, choices=[1, 2, 4, 8, 32])
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--model', type=str, default='vit_base_patch16_224_in21k')
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--method', type=str, default='adaptformer',
                        choices=['adaptformer', 'adaptformer-bihead', 'lora', 'lora-bihead'])
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--config_path', type=str, default='.')
    parser.add_argument('--model_path', type=str, default='.')
    parser.add_argument('--load_config', action='store_true', default=False)
    parser.add_argument('--log_path', type=str, default='.')
    parser.add_argument('--r_layer', default=None, type=int)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=10)
    # merge-consistency training
    parser.add_argument('--consistency', action='store_true', default=False)
    parser.add_argument('--cons_weight', type=float, default=1.0, help='weight of KL(merged || full)')
    parser.add_argument('--cons_temp', type=float, default=1.0, help='distillation temperature')
    parser.add_argument('--full_ce_weight', type=float, default=1.0, help='weight of CE on the full branch')
    parser.add_argument('--feat_weight', type=float, default=0.0, help='weight of CLS feature alignment')
    parser.add_argument('--full_every', type=int, default=1, help='run the full branch every k steps')
    parser.add_argument('--no_full_eval', action='store_true', default=False,
                        help='skip evaluating the full-token branch at test time')
    parser.add_argument('--no_latency', action='store_true', default=False)
    args = parser.parse_args()
    print(args)
    if args.consistency and args.r_layer is None:
        parser.error('--consistency requires --r_layer')
    args.full_eval = args.r_layer is not None and not args.no_full_eval
    if args.eval or args.load_config:
        load_config(args)
    set_seed(args.seed)
    args.best_acc = 0
    args.best_ep = 0
    args.best_acc_full = None
    args.history = []
    vit = create_model(args.model, checkpoint_path='ViT-B_16.npz', drop_path_rate=0.1)
    train_dl, test_dl = get_data(args.dataset, normalize=False)

    save_folder = os.path.join(args.model_path, args.dataset)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    log_writer = SummaryWriter(log_dir=os.path.join(args.model_path, args.dataset))

    if args.method == 'adaptformer':
        adaptformer.set_adapter(vit, dim=args.dim, s=args.scale, bit=args.bit)
        vit.reset_classifier(get_classes_num(args.dataset))
    elif args.method == 'lora':
        lora.set_adapter(vit, dim=args.dim, s=args.scale, bit=args.bit)
        vit.reset_classifier(get_classes_num(args.dataset))

    if args.r_layer is not None:
        fpet.apply_fpet.apply_patch(vit, method = args.method, r_layer = args.r_layer)
        print(f'FPET applied!')

    if not args.eval:
        import shutil
        shutil.copytree('./configs', os.path.join(save_folder, 'codes/configs'), dirs_exist_ok=True)
        shutil.copytree('./fpet', os.path.join(save_folder, 'codes/fpet'), dirs_exist_ok=True)
        for f in ['adaptformer.py', 'lora.py', 'main.py', 'utils.py', 'vtab.py', 'evidence.py']:
            shutil.copy(f, os.path.join(save_folder, 'codes', f))

        import sys
        with open(os.path.join(save_folder, 'params.sh'), 'w+') as out:
            sys.argv[0] = os.path.join(os.getcwd(), sys.argv[0])
            out.write('#!/bin/bash\n')
            out.write('python3 ')
            out.write(' '.join(sys.argv))
            out.write('\n')

        start_total = time.time()

    run_info = {'args': {k: v for k, v in vars(args).items() if k != 'history'},
                'env': evidence.env_info(), 'num_train': len(train_dl.dataset), 'num_test': len(test_dl.dataset)}
    evidence.write_json(os.path.join(save_folder, 'run_info.json' if not args.eval else 'eval_info.json'), run_info)

    if not args.eval:
        trainable = []
        trainable_n = []
        for n, p in vit.named_parameters():
            if ('adapter' in n or 'head' in n) and p.requires_grad:
                trainable.append(p)
                trainable_n.append(n)
            else:
                p.requires_grad = False
        print(f'{trainable_n=}')
        opt = AdamW(trainable, lr=args.lr, weight_decay=args.wd)
        scheduler = CosineLRScheduler(opt, t_initial=args.epochs,
                                      warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6, decay_rate=0.1)
        vit = train(args, vit, train_dl, opt, scheduler, epoch=args.epochs, log_writer=log_writer)

        end_total = time.time()
        total_time = end_total - start_total
        log_stats = {"best_epoch": args.best_ep, "best_acc": args.best_acc, "total_train_time": str(datetime.timedelta(seconds=int(total_time)))}
        log(args, 0, 0, 0, 0, 0, log_stats=log_stats)

    else:
        load(args, vit)
        args.best_acc = test(vit, test_dl)
        if args.full_eval:
            args.best_acc_full = test_full(vit, test_dl)

    # evidence: accuracy, efficiency and training cost of this run
    vit = vit.cuda()
    summary = {
        'dataset': args.dataset,
        'run': run_name(args.model_path),
        'method': args.method,
        'r_layer': args.r_layer,
        'consistency': args.consistency,
        'seed': args.seed,
        'best_acc': args.best_acc * 100,
        'best_epoch': args.best_ep,
        'best_acc_full_branch': args.best_acc_full * 100 if args.best_acc_full is not None else None,
        'gflops': evidence.count_flops(vit),
        'latency_ms_per_img': None if args.no_latency else evidence.measure_latency(vit),
        **evidence.count_params(vit),
    }
    if not args.eval:
        summary['peak_train_mem_gb'] = args.peak_train_mem_gb
        summary['total_train_time_s'] = total_time
        summary['history'] = args.history
    evidence.write_json(os.path.join(save_folder, 'summary.json' if not args.eval else 'eval_summary.json'), summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'history'}, indent=2))

    print('best_acc:', args.best_acc)
