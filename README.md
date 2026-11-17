# MC-FPET: merge-consistency training for faster parameter-efficient tuning

This branch (`merge-consistency`) extends the official FPET code below with **merge-consistency training (MC-FPET)**.
During training, a second, full-token path runs through the same frozen ViT-B/16, adapters and classifier as
the token-merged path. Both paths are trained with cross-entropy, and the merged prediction is also distilled
towards the detached full-token prediction:

    loss = CE(merged) + λ_f · CE(full) + λ_kl · T² · KL(stop-grad(full) ‖ merged)

The full-token path is used only during training. The deployed model has the same architecture, parameters
and FLOPs as FPET. On four VTAB-1k tasks with merging at layer 2, MC-FPET improves on FPET by +1.39 pp
(12 of 12 task–seed runs) and recovers about half of the accuracy lost to merging.

## Reproduce the paper's numbers (no GPU needed)

The per-run results of all 334 training runs are in the repository: the 60 original runs in `*/pilot/` and
the 274 revision runs in `results/` (see `results/README.md`). Each run has a `summary.json` (accuracy at
every evaluation, FLOPs, latency, memory, training time) and, where it was saved, a `run_info.json` (all
arguments, software environment and Git commit).

```
pip install numpy scipy pandas statsmodels matplotlib
python revision_analysis.py           # every statistic in the paper -> REVISION_ANALYSIS.md, revision_analysis.json
python paper/scripts/make_figures.py  # the figures
```

## Train a run

Set up the environment, data and pretrained ViT-B/16 as described in the FPET instructions below. The data
directory can be set with `VTAB_ROOT=<path to vtab-1k>`. For example, on CIFAR-100 with seed 0:

```
python main.py --dataset cifar --method adaptformer --dim 8 --bit 32 --load_config --seed 0 --model_path result/adaptformer/seed0                            # AdaptFormer (no merging)
python main.py --dataset cifar --method adaptformer --dim 8 --bit 32 --load_config --seed 0 --model_path result/fpet-r2/seed0  --r_layer 2               # FPET, merge at layer 2
python main.py --dataset cifar --method adaptformer --dim 8 --bit 32 --load_config --seed 0 --model_path result/mcpet-r2/seed0 --r_layer 2 --consistency # MC-FPET
```

The loss weights are set with `--full_ce_weight`, `--cons_weight` and `--cons_temp` (all 1 by default).
`jobs.py` holds the exact command of every run in the paper (`python jobs.py list`), including the ablations,
LoRA (`--method lora`) and the other merge layers. `python smoke_test.py` runs the unit tests of the method.

---

# [CVPR 2025] FasterPET (FPET)

The official pytorch implementation of FasterPET (FPET) from our paper [Faster Parameter-Efficient Tuning with Token Redundancy Reduction](https://arxiv.org/abs/2503.20282)

Kwonyoung Kim<sup>1</sup> Jungin Park<sup>1*</sup> Jin Kim<sup>1</sup> Hyeongjun Kwon<sup>1</sup> Kwanghoon Sohn<sup>1,2*</sup>

<sup>1</sup>Yonsei University <sup>2</sup>Korea Institute of Science and Technology (KIST) <sup>*</sup>Corresponding authors


## Installation
```
conda create -n fpet python=3.10.13
conda install -r requirements.txt
```

## Data Preparation
To download the datasets, please refer to https://github.com/ZhangYuanhan-AI/NOAH/#data-preparation. Set the directory of the dataset as `<YOUR PATH>/fpet/data/`.

## Pretrained Model
Download the [pretrained ViT-B/16](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) to `<YOUR PATH>/fpet/ViT-B_16.npz`.

## Train & Evaluation
Train and evaluation on VTAB-1K.
```
sh train.sh
```

## Citation
```
@inproceedings{kim2025faster,
  title={Faster Parameter-Efficient Tuning with Token Redundancy Reduction},
  author={Kim, Kwonyoung and Park, Jungin and Kim, Jin and Kwon, Hyeongjun and Sohn, Kwanghoon},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

## Acknowledgements
Our implementation is based on [NOAH](https://github.com/ZhangYuanhan-AI/NOAH), [timm](https://github.com/rwightman/pytorch-image-models), [Binary Adapter](https://github.com/JieShibo/PETL-ViT/tree/main/binary_adapter) and [ToMe](https://github.com/facebookresearch/tome). Thanks for their awesome works.
