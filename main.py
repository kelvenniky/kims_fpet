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
import random


def consistency_loss(args, model, x, y):
    """
    Merge-consistency objective: CE on the merged branch, CE on the full-token branch and
    a KL term distilling the (detached) full-token prediction into the merged branch.
    """
    out, out_full, feat, feat_full = model.forward_two_branch(
        x, tie_rng=args.tie_masks, full_grad=not args.full_no_grad)
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


def distillation_loss(args, model, teacher, x, y):
    """
    Distillation baseline: CE on the merged model plus the same KL term as consistency_loss,
    but the target comes from a separately trained, frozen unmerged PET model (no shared weights).
    """
    out = model(x)
    with torch.no_grad():
        out_t = teacher(x)
    ce = F.cross_entropy(out, y)
    T = args.cons_temp
    kl = F.kl_div(F.log_softmax(out / T, dim=-1), F.softmax(out_t / T, dim=-1),
                  reduction='batchmean') * T * T
    loss = ce + args.cons_weight * kl
    return loss, {'ce': ce.item(), 'kl': kl.item()}


def build_teacher(args):
    """Frozen unmerged PET model (same method/dim/scale) loaded from --teacher_ckpt."""
    # keep the CPU RNG stream identical to a plain FPET run with the same seed
    rng = torch.get_rng_state()
    teacher = create_model(args.model, checkpoint_path='ViT-B_16.npz', drop_path_rate=0.1)
    if args.method == 'adaptformer':
        adaptformer.set_adapter(teacher, dim=args.dim, s=args.scale, bit=args.bit)
    else:
        lora.set_adapter(teacher, dim=args.dim, s=args.scale, bit=args.bit)
    teacher.reset_classifier(get_classes_num(args.dataset))
    torch.set_rng_state(rng)

    state = torch.load(args.teacher_ckpt, map_location='cpu')
    own = teacher.state_dict()
    unknown = [k for k in state if k not in own]
    missing = [k for k, p in teacher.named_parameters() if ('adapter' in k or 'head' in k) and k not in state]
    assert not unknown and not missing, f'teacher checkpoint mismatch: unknown={unknown[:3]} missing={missing[:3]}'
    teacher.load_state_dict(state, strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher.cuda()


def run_name(model_path):
    # model_path is <root>/<run> or <root>/<run>/seed<k>
    parts = os.path.normpath(model_path).split(os.sep)
    return parts[-2] if parts[-1].startswith('seed') and len(parts) > 1 else parts[-1]


def train(args, model, dl, opt, scheduler, epoch, log_writer, teacher=None):
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
            elif teacher is not None:
                loss, parts = distillation_loss(args, model, teacher, x, y)
            elif args.merge_prob < 1:
                # stochastic merging: one forward per step, merged with probability merge_prob
                merged = merge_rng.random() < args.merge_prob
                out = model(x) if merged else model.forward_full(x)
                loss = F.cross_entropy(out, y)
                parts = {'ce': loss.item(), 'merged': float(merged)}
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
            val_acc = test(model, val_dl) if val_dl is not None else None
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
            args.history.append({'epoch': ep, 'acc': acc, 'acc_full': acc_full, 'val_acc': val_acc,
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
        out = model.forward_full(x).data
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
    # revision experiments (reviewer M1-M3, M7)
    parser.add_argument('--tie_masks', action='store_true', default=False,
                        help='full-token path replays the merged path RNG: tied drop-path masks, '
                             'RNG stream identical to FPET')
    parser.add_argument('--full_no_grad', action='store_true', default=False,
                        help='run the full-token path without gradients (no-grad teacher; needs --full_ce_weight 0)')
    parser.add_argument('--merge_prob', type=float, default=1.0,
                        help='<1: stochastic merging, one forward per step, merged with this probability')
    parser.add_argument('--drop_path', type=float, default=0.1, help='stochastic depth rate of the backbone')
    parser.add_argument('--adapter_dropout', type=float, default=0.1, help='dropout inside every adapter')
    parser.add_argument('--val_select', action='store_true', default=False,
                        help='train on train800 only, evaluate val200 too, report test acc at the best val epoch')
    # distillation baseline: KL to a frozen unmerged PET model (uses --cons_weight / --cons_temp)
    parser.add_argument('--teacher_ckpt', type=str, default=None,
                        help='<dataset>.pth of a trained unmerged run (e.g. adaptformer/seed0/cifar.pth)')
    args = parser.parse_args()
    print(args)
    if args.consistency and args.r_layer is None:
        parser.error('--consistency requires --r_layer')
    if args.teacher_ckpt and (args.consistency or args.r_layer is None or args.eval):
        parser.error('--teacher_ckpt requires --r_layer and cannot be combined with --consistency or --eval')
    if args.full_no_grad and args.full_ce_weight > 0:
        parser.error('--full_no_grad needs --full_ce_weight 0 (the full path gets no gradient)')
    if args.merge_prob < 1 and (args.r_layer is None or args.consistency or args.teacher_ckpt):
        parser.error('--merge_prob < 1 needs --r_layer and no --consistency / --teacher_ckpt')
    args.full_eval = args.r_layer is not None and not args.no_full_eval
    if args.eval or args.load_config:
        load_config(args)
    set_seed(args.seed)
    args.best_acc = 0
    args.best_ep = 0
    args.best_acc_full = None
    args.history = []
    vit = create_model(args.model, checkpoint_path='ViT-B_16.npz', drop_path_rate=args.drop_path)
    if args.val_select:
        train_dl, val_dl = get_data(args.dataset, normalize=False, evaluate=False)
        test_dl = get_data(args.dataset, normalize=False)[1]
    else:
        train_dl, test_dl = get_data(args.dataset, normalize=False)
        val_dl = None
    # separate stream so stochastic merging does not change data order or dropout masks
    merge_rng = random.Random(1000 + args.seed)

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
    for m in vit.modules():
        if isinstance(m, (adaptformer.Adapter, lora.Adapter)) and hasattr(m, 'dropout'):
            m.dropout.p = args.adapter_dropout

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
        teacher = build_teacher(args) if args.teacher_ckpt else None
        vit = train(args, vit, train_dl, opt, scheduler, epoch=args.epochs, log_writer=log_writer, teacher=teacher)
        del teacher

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
        'teacher_ckpt': args.teacher_ckpt,
        'variant': {k: getattr(args, k) for k in
                    ['cons_weight', 'cons_temp', 'full_ce_weight', 'feat_weight', 'full_every', 'tie_masks',
                     'full_no_grad', 'merge_prob', 'drop_path', 'adapter_dropout', 'val_select', 'epochs',
                     'eval_every']},
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
        # reviewer M3: last evaluation (no checkpoint selection) and validation-based selection
        last = args.history[-1]
        summary['final_acc'] = last['acc'] * 100
        summary['final_acc_full_branch'] = last['acc_full'] * 100 if last['acc_full'] is not None else None
        summary['n_evals'] = len(args.history)
        if args.val_select:
            sel = max(args.history, key=lambda h: (h['val_acc'], h['epoch']))
            summary['val_selected_epoch'] = sel['epoch']
            summary['val_selected_val_acc'] = sel['val_acc'] * 100
            summary['val_selected_test_acc'] = sel['acc'] * 100
    evidence.write_json(os.path.join(save_folder, 'summary.json' if not args.eval else 'eval_summary.json'), summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'history'}, indent=2))

    print('best_acc:', args.best_acc)
