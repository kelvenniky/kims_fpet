import os
import json
import time
import socket
import platform
import subprocess

import torch
from torch import nn
import timm
from timm.models.vision_transformer import Attention


def git_info():
    def run(cmd):
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    return {
        'commit': run(['git', 'rev-parse', 'HEAD']),
        'dirty': bool(run(['git', 'status', '--porcelain', '--untracked-files=no'])),
    }


def env_info():
    info = {
        'host': socket.gethostname(),
        'platform': platform.platform(),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'timm': timm.__version__,
        'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    info['git'] = git_info()
    return info


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'params_total': total, 'params_trainable': trainable}


@torch.no_grad()
def count_flops(model, img_size=224):
    """
    Inference GFLOPs per image counted as multiply-accumulates (the fvcore convention used
    in the FPET paper, where ViT-B/16 = 17.6G). Counts every nn.Linear (incl. adapters,
    LoRA and the key-refinement adapter), the patch-embedding conv and the two attention
    matmuls, using the actual token count seen by each layer. The bipartite similarity
    matmul of the merging module is not counted (it is < 0.001G).
    """
    macs = [0]

    def linear_hook(m, inp, out):
        macs[0] += inp[0].numel() * m.out_features

    def conv_hook(m, inp, out):
        macs[0] += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def attn_hook(m, inp, out):
        B, N, C = inp[0].shape
        macs[0] += 2 * B * N * N * C

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, Attention):
            hooks.append(m.register_forward_hook(attn_hook))

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    model(torch.randn(1, 3, img_size, img_size, device=device))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return macs[0] / 1e9


@torch.no_grad()
def measure_latency(model, batch_size=256, warmup=10, iters=30, img_size=224):
    """Inference time per image in ms (batch size 256, as in the FPET paper)."""
    model.eval()
    model = model.cuda()
    x = torch.randn(batch_size, 3, img_size, img_size, device='cuda')
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        model(x)
    torch.cuda.synchronize()
    return (time.time() - start) / (iters * batch_size) * 1000


def write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
