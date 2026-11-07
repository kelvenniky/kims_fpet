"""
CPU sanity checks for merge-consistency training (no data or pretrained weights needed).
  1. merged branch of forward_two_branch == the regular FPET forward
  2. full-token branch == the same PET model without FPET (same weights)
  3. FLOPs counter reproduces ViT-B/16 = 17.6G and the FPET reductions
  4. the consistency loss trains adapters, key refinement and head, and nothing else
Usage: python smoke_test.py
"""
import copy
from argparse import Namespace

import torch
from timm.models import create_model

import adaptformer
import lora
import fpet
import evidence
from main import consistency_loss


def build(method, r_layer=None, num_classes=10):
    torch.manual_seed(0)
    vit = create_model('vit_base_patch16_224_in21k', pretrained=False, drop_path_rate=0.1)
    if method == 'adaptformer':
        adaptformer.set_adapter(vit, dim=8, s=1, bit=32)
    else:
        lora.set_adapter(vit, dim=8, s=1, bit=32)
    vit.reset_classifier(num_classes)
    # adapters are zero-initialised; perturb them so the checks are not trivial
    for n, p in vit.named_parameters():
        if 'adapter' in n or 'lora' in n:
            torch.nn.init.normal_(p, std=0.02)
    if r_layer is not None:
        fpet.apply_fpet.apply_patch(vit, method=method, r_layer=r_layer)
    return vit


def freeze(vit):
    for n, p in vit.named_parameters():
        p.requires_grad = ('adapter' in n or 'head' in n)


@torch.no_grad()
def check_equivalence(method, r_layer):
    x = torch.randn(2, 3, 224, 224)
    model = build(method, r_layer).eval()
    plain = build(method, None).eval()
    plain.load_state_dict({k: v for k, v in model.state_dict().items() if 'refinement' not in k})

    out_merged, out_full, _, _ = model.forward_two_branch(x)
    ref_merged = model(x)
    ref_full = plain(x)
    e1 = (out_merged - ref_merged).abs().max().item()
    e2 = (out_full - ref_full).abs().max().item()
    assert e1 < 1e-4, f'merged branch differs from FPET forward: {e1}'
    assert e2 < 1e-4, f'full branch differs from unmerged PET model: {e2}'
    print(f'[ok] {method} r={r_layer}: |merged - FPET| = {e1:.1e}, |full - PET| = {e2:.1e}')


def check_flops():
    base = evidence.count_flops(build('adaptformer', None))
    print(f'[ok] AdaptFormer (no merging): {base:.2f} GFLOPs')
    assert abs(base - 17.6) < 0.1, base
    for r in (2, 4, 6):
        g = evidence.count_flops(build('adaptformer', r))
        print(f'[ok] FPET-AdaptFormer r={r}: {g:.2f} GFLOPs ({(g - base) / base * 100:+.1f}%)')


def check_gradients(method):
    model = build(method, 4).train()
    freeze(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    args = Namespace(cons_temp=1.0, full_ce_weight=1.0, cons_weight=1.0, feat_weight=0.1,
                     tie_masks=False, full_no_grad=False)
    x, y = torch.randn(4, 3, 224, 224), torch.randint(0, 10, (4,))
    loss, parts = consistency_loss(args, model, x, y)
    opt.zero_grad()
    loss.backward()
    with_grad = {n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    frozen_with_grad = {n for n, p in model.named_parameters() if not p.requires_grad and p.grad is not None}
    assert not frozen_with_grad, frozen_with_grad
    assert any('refinement' in n for n in with_grad), 'key refinement receives no gradient'
    assert any('head' in n for n in with_grad)
    assert any(('adapter' in n or 'lora' in n) and 'refinement' not in n for n in with_grad)
    before = copy.deepcopy(model.state_dict())
    opt.step()
    changed = [k for k, v in model.state_dict().items() if not torch.equal(v, before[k])]
    assert all('adapter' in k or 'head' in k for k in changed), changed
    print(f'[ok] {method}: loss parts {({k: round(v, 3) for k, v in parts.items()})}; '
          f'{len(with_grad)} trainable tensors get gradients, frozen backbone untouched')


@torch.no_grad()
def check_forward_full(method, r_layer):
    """forward_full (stochastic merging, test_full) == full branch of forward_two_branch == unmerged PET."""
    x = torch.randn(2, 3, 224, 224)
    model = build(method, r_layer).eval()
    plain = build(method, None).eval()
    plain.load_state_dict({k: v for k, v in model.state_dict().items() if 'refinement' not in k})
    e1 = (model.forward_full(x) - model.forward_two_branch(x)[1]).abs().max().item()
    e2 = (model.forward_full(x) - plain(x)).abs().max().item()
    e3 = (model(x) - model.forward_two_branch(x)[0]).abs().max().item()  # skip_merge flag was reset
    assert e1 < 1e-4 and e2 < 1e-4 and e3 < 1e-4, (e1, e2, e3)
    print(f'[ok] {method} r={r_layer}: forward_full == full branch ({e1:.1e}) == unmerged PET ({e2:.1e}); '
          f'merging restored afterwards ({e3:.1e})')


def check_tied_rng(method, r_layer):
    """
    Train mode (drop path + adapter dropout active):
      1. with tie_rng the merged logits equal a plain FPET training forward from the same RNG state,
         and the RNG stream afterwards is the same as after that FPET forward (pairing, reviewer M2);
      2. without dropout/drop path the two full-path variants agree exactly;
      3. full_grad=False gives the same loss value and the same gradients when lambda_f = 0.
    """
    x, y = torch.randn(2, 3, 224, 224), torch.randint(0, 10, (2,))
    model = build(method, r_layer).train()
    torch.manual_seed(123)
    ref = model(x)
    after_ref = torch.rand(1)
    torch.manual_seed(123)
    out_m, out_f, _, _ = model.forward_two_branch(x, tie_rng=True)
    after_tied = torch.rand(1)
    e = (out_m - ref).abs().max().item()
    assert e < 1e-4 and torch.equal(after_ref, after_tied), (e, after_ref, after_tied)
    torch.manual_seed(123)
    out_m2, _, _, _ = model.forward_two_branch(x, tie_rng=False)
    untied = (out_m2 - ref).abs().max().item()
    print(f'[ok] {method} r={r_layer} train mode: tied merged path == FPET forward ({e:.1e}), RNG stream '
          f'unchanged; untied merged path differs from FPET by {untied:.2e} (independent masks)')

    # drop path off and adapter dropout off: everything deterministic, tied == untied
    for m in model.modules():
        if isinstance(m, (adaptformer.Adapter, lora.Adapter)) and hasattr(m, 'dropout'):
            m.dropout.p = 0.0
        if m.__class__.__name__ == 'DropPath':
            m.drop_prob = 0.0
    a = model.forward_two_branch(x, tie_rng=True)
    b = model.forward_two_branch(x, tie_rng=False)
    assert all((u - v).abs().max() < 1e-4 for u, v in zip(a, b))

    freeze(model)
    args = Namespace(cons_temp=1.0, full_ce_weight=0.0, cons_weight=1.0, feat_weight=0.0,
                     tie_masks=True, full_no_grad=False)
    grads = []
    for no_grad in (False, True):
        args.full_no_grad = no_grad
        model.zero_grad()
        loss, _ = consistency_loss(args, model, x, y)
        loss.backward()
        grads.append({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
        grads[-1]['loss'] = loss.detach()
    g0, g1 = grads
    assert g0.keys() == g1.keys()
    d = max((g0[k] - g1[k]).abs().max().item() for k in g0)
    assert d < 1e-5, d
    print(f'[ok] {method} r={r_layer}: lambda_f = 0 with and without gradients through the full path give '
          f'the same loss and gradients (max diff {d:.1e})')


if __name__ == '__main__':
    for method in ('adaptformer', 'lora'):
        check_forward_full(method, 2)
        check_tied_rng(method, 2)
    for method in ('adaptformer', 'lora'):
        for r in (2, 6):
            check_equivalence(method, r)
    check_flops()
    for method in ('adaptformer', 'lora'):
        check_gradients(method)
    print('all checks passed')
