# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------


from typing import Tuple

import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer

from fpet.merge_fpet import bipartite_diff_matching, merge_source, merge_wavg
from adaptformer import Adapter

def forward_block(self, x):

    attn_size = self._fpet_info["size"] if self._fpet_info["prop_attn"] else None
    x_attn, metric = self.attn(self.norm1(x), attn_size)
    x = x + self.drop_path(x_attn)

    if hasattr(self, 'refinement'):
        metric = metric.detach()
        metric = metric + 1 * self.refinement(metric)
        # Apply FPET here
        merge = bipartite_diff_matching(
            metric,
            self._fpet_info["class_token"],
            self._fpet_info["distill_token"],
        )
        if self._fpet_info["trace_source"]:
            self._fpet_info["source"] = merge_source(
                merge, x, self._fpet_info["source"]
            )
        x, self._fpet_info["size"] = merge_wavg(merge, x, self._fpet_info["size"])
    x = x + self.drop_path(self.mlp(self.norm2(x))) + self.adapter_mlp(x) * self.s
    return x

def _drop_path1(block, x):
    return block.drop_path1(x) if hasattr(block, "drop_path1") else block.drop_path(x)


def _drop_path2(block, x):
    return block.drop_path2(x) if hasattr(block, "drop_path2") else block.drop_path(x)


def mlp_residual(block, x):
    # AdaptFormer blocks carry a parallel adapter next to the frozen MLP
    out = x + _drop_path2(block, block.mlp(block.norm2(x)))
    if hasattr(block, "adapter_mlp"):
        out = out + block.adapter_mlp(x) * block.s
    return out


def forward_merge_branch(block, x):
    """
    Forward of the merging block for consistency training.
    The attention is computed once on all tokens and shared; the block then returns
    both the merged output (as in FPET) and the full-token output (no merging).
    """
    info = block._fpet_info
    attn_size = info["size"] if info["prop_attn"] else None
    x_attn, metric = block.attn(block.norm1(x), attn_size)
    x = x + _drop_path1(block, x_attn)

    x_full = mlp_residual(block, x)

    metric = metric.detach()
    metric = metric + 1 * block.refinement(metric)
    merge = bipartite_diff_matching(
        metric,
        info["class_token"],
        info["distill_token"],
    )
    x_merged, info["size"] = merge_wavg(merge, x, info["size"])
    return mlp_residual(block, x_merged), x_full


def forward_attn(self, x, size):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    delta_q = self.lora_q(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) * self.s
    delta_v = self.lora_v(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) * self.s
    q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
    q, v = q + delta_q[0], v + delta_v[0]
    attn = (q @ k.transpose(-2, -1)) * self.scale

    if size is not None:
        attn = attn + size.log()[:, None, None, :, 0]

    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x, k.mean(1)

class FPETBlock(Block):
    """
    Modifications:
     - Apply FPET between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f'x.shape is {x.shape}')
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        attn_size = self._fpet_info["size"] if self._fpet_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + self._drop_path1(x_attn)

        if hasattr(self, 'refinement'):
            metric = metric.detach()
            metric = metric + 1 * self.refinement(metric)
            # Apply FPET here
            merge = bipartite_diff_matching(
                metric,
                self._fpet_info["class_token"],
                self._fpet_info["distill_token"],
            )
            if self._fpet_info["trace_source"]:
                self._fpet_info["source"] = merge_source(
                    merge, x, self._fpet_info["source"]
                )
            x, self._fpet_info["size"] = merge_wavg(merge, x, self._fpet_info["size"])

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x


class FPETAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k as well here
        return x, k.mean(1)


def make_fpet_class(transformer_class):
    class FPETVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._fpet_info["size"] = None
            self._fpet_info["source"] = None

            return super().forward(*args, **kwdargs)

        def forward_two_branch(self, x: torch.Tensor):
            """
            Shared-prefix two-branch forward used for merge-consistency training.
            Layers before the merging block (and its attention) run once; afterwards the
            merged branch and the full-token branch run through the remaining blocks.

            Returns (logits_merged, logits_full, feat_merged, feat_full).
            The merged branch is numerically identical to the regular forward.
            """
            assert self.dist_token is None, "two-branch forward assumes no distillation token"
            self._fpet_info["size"] = None
            self._fpet_info["source"] = None

            x = self.patch_embed(x)
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = self.pos_drop(x + self.pos_embed)

            x_full = None
            for blk in self.blocks:
                if x_full is None:
                    if hasattr(blk, "refinement"):
                        x, x_full = forward_merge_branch(blk, x)
                    else:
                        x = blk(x)
                else:
                    x = blk(x)
                    # the full branch has unit token sizes, so no proportional attention
                    size = self._fpet_info["size"]
                    self._fpet_info["size"] = None
                    x_full = blk(x_full)
                    self._fpet_info["size"] = size
            assert x_full is not None, "no merging block found; set r_layer"

            feat = self.pre_logits(self.norm(x)[:, 0])
            feat_full = self.pre_logits(self.norm(x_full)[:, 0])
            return self.head(feat), self.head(feat_full), feat, feat_full

    return FPETVisionTransformer


def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True, method = '', r_layer: int = 6, dim_key = 64
):
    """
    Applies FPET to this transformer.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._fpet_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    FPETVisionTransformer = make_fpet_class(model.__class__)

    model.__class__ = FPETVisionTransformer
    model._fpet_info = {
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._fpet_info["distill_token"] = True

    idx = 0
    for module in model.modules():
        if isinstance(module, Block):
            module._fpet_info = model._fpet_info
            if r_layer == idx:
                module.refinement = Adapter(dim=32, bit=1, in_dim=dim_key)
                
            if 'adaptformer' not in method:
                module.__class__ = FPETBlock
            else:
                bound_method = forward_block.__get__(module, module.__class__)
                setattr(module, 'forward', bound_method)
            
            idx += 1
                
        elif isinstance(module, Attention):
            if 'lora' not in method:
                module.__class__ = FPETAttention
            else:
                bound_method = forward_attn.__get__(module, module.__class__)
                setattr(module, 'forward', bound_method)
                
