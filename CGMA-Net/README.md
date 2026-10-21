# CGMA-Net

This folder contains the code set required for **training and inference**.
The trained model weights have been uploaded to Hugging Face and are publicly available at: https://huggingface.co/zjuzjz/CGMA-Net.

## Files

| File | Purpose |
|---|---|
| `models.py` | Model architecture: RETFound CFP encoder, VF patch transformer encoder, cross-modal gated fusion, classifier head and uncertainty head |
| `datasets.py` | Scans the `Test/` directory and builds multimodal samples (`cfp` / `vf` / `pair`), producing images and modality masks |
| `utils.py` | Image preprocessing transforms, device selection, MC Dropout enabling, random seed |
| `losses.py` | Weighted BCE, heteroscedastic uncertainty loss, gate entropy regularization, and the combined multimodal training loss |
| `metrics.py` | Binary classification metrics (accuracy, sensitivity, specificity, precision, F1, AUC, AUPRC, Brier, ECE) and subgroup metrics |
| `summarize_test_metrics.py` | Inference entry point: loads a checkpoint, runs MC Dropout inference, and outputs per-sample predictions with 95% CIs |
| `train_single.py` | Single-modality pretraining entry point (CFP-only or VF-only classifier) |
| `train_multimodal.py` | Multimodal two-branch gated fusion training entry point |
| `README.md` | This file |

## Requirements

```bash
pip install torch torchvision scikit-learn numpy pillow
```

Python 3.10+ and PyTorch 2.x are recommended.

## Data Layout

`--data-root` must contain a `Test/` split with the following structure:

```text
Test/
├── 0/            # 0 = normal
│   ├── cfp/      # CFP-only images
│   ├── vf/       # VF-only images 
│   └── pair/
│       └── <patient_id>/
│           ├── xxx.jpg        # CFP (any image not starting with vf_)
│           └── vf_xxx.jpg     # VF (files starting with vf_ are treated as VF)
└── 1/            # 1 = glaucoma
```

## Inference

```bash
python summarize_test_metrics.py \
  --checkpoint outputs/multimodal/best_model.pt \
  --data-root ../DataSet \
  --output-dir outputs/multimodal/statistics_out \
  --source checkpoint \
  --device auto \
  --threshold 0.5 \
  --bootstrap 2000 \
  --seed 1997 \
  --mc-samples 20
```

### Arguments

- `--checkpoint`: the `best_model.pt` saved during training. It must contain `model_state_dict` and `args` (architecture hyperparameters are restored from `args`).
- `--data-root`: root directory containing the `Test/` split.
- `--output-dir`: directory for the generated CSV files.
- `--device`: `auto`, `cuda`, or `cpu` (default `auto`).
- `--mc-samples`: number of Monte Carlo Dropout forward passes (default 20).
- `--threshold`: classification threshold (default 0.5).
- `--bootstrap`: number of bootstrap iterations for the 95% confidence intervals (default 2000).
- `--seed`: random seed.

## Outputs

The `--output-dir` will contain:

- `test_predictions_used_for_statistics.csv`: per-sample predictions (`patient_id`, `label`, `sample_type`, `cfp_path`, `vf_path`, `prob_mean`, `uncertainty_mc_var`, `uncertainty_head_mean`, `mask_cfp`, `mask_vf`).
- `test_metrics_with_95ci_long.csv`: per-metric rows grouped by `overall` / `cfp` / `vf` / `pair`, each with its 95% confidence interval (long format).
- `test_metrics_with_95ci_wide.csv`: the same metrics in wide format.

## Training

### Training data layout

`--data-root` must contain three splits: `TrainAug/`, `Valid/`, and `Test/`. Each split uses the same nested `0/` and `1/` structure as the `Test/` layout above, with `cfp/`, `vf/`, and `pair/` subdirectories.

### Two-stage training

The model is trained in two stages:

1. **Single-modality pretraining.** Train a CFP-only classifier and a VF-only classifier with `train_single.py`.
2. **Multimodal gated-fusion training.** Initialize both encoders from the pretrained single-modality checkpoints and train the two-branch gated fusion model with `train_multimodal.py`.

The [RETFound](https://www.nature.com/articles/s41586-023-06555-x) pretrained backbone (`RETFound_mae_natureCFP.pth`) is required for the CFP branch. Download it and point `--retfound-checkpoint` to it.

### Stage 1: single-modality pretraining

```bash
python train_single.py \
  --data-root ../DataSet \
  --modality cfp \
  --output-dir outputs \
  --retfound-checkpoint ../RETFound/RETFound_mae_natureCFP.pth \
  --epochs 50 \
  --batch-size 8
```

```bash
python train_single.py \
  --data-root ../DataSet \
  --modality vf \
  --output-dir outputs \
  --vf-patch-size 16 \
  --vf-embed-dim 256 \
  --vf-transformer-depth 6 \
  --vf-transformer-heads 8 \
  --epochs 50 \
  --batch-size 8
```

Each run creates `outputs/single_<modality>/` with `best_model.pt`, `epoch_metrics.csv`, `test_metrics.csv`, and `summary.json`.

### Stage 2: multimodal gated-fusion training

```bash
python train_multimodal.py \
  --data-root ../DataSet \
  --output-dir outputs/multimodal \
  --cfp-checkpoint outputs/single_cfp/best_model.pt \
  --vf-checkpoint outputs/single_vf/best_model.pt \
  --retfound-checkpoint ../RETFound/RETFound_mae_natureCFP.pth \
  --vf-patch-size 16 \
  --vf-embed-dim 256 \
  --vf-transformer-depth 6 \
  --vf-transformer-heads 8 \
  --gate-entropy-weight 0.01 \
  --uncertainty-loss-weight 0.05 \
  --epochs 100 \
  --batch-size 8 \
  --modality-dropout-prob 0.5 \
  --mc-samples 20 \
  --use-group-balanced-sampler
```

Key training options:

- `--modality-dropout-prob`: probability of randomly dropping one modality of paired samples during training, for missing-modality robustness.
- `--gate-entropy-weight`: weight of the gate entropy regularization.
- `--uncertainty-loss-weight`: weight of the heteroscedastic uncertainty loss.
- `--use-group-balanced-sampler`: optional sampler balancing `sample_type x class` groups.

Training writes to `outputs/multimodal/`:

- `best_model.pt`: best validation-AUC checkpoint, used for inference.
- `epoch_metrics.csv`: train/valid metrics per epoch.
- `mc_dropout_metrics.csv`: final valid/test MC Dropout metrics.
- `valid_predictions_mc_dropout.csv` / `test_predictions_mc_dropout.csv`: per-sample predictions with uncertainty.

After training, run `summarize_test_metrics.py` (see above) to compute grouped metrics with 95% CIs; its default `--predictions` path (`outputs/multimodal/test_predictions_mc_dropout.csv`) points to the file produced here.

## Notes

- Trained model weights are not included in this repository. Download them from [Hugging face](https://huggingface.co/zjuzjz/CGMA-Net).
- `summarize_test_metrics.py` performs batch inference over the `Test/` directory. For online inference on an arbitrary CFP + VF image pair, wrap the `MultiModalGatedClassifier` from `models.py` in your own script.
- When applying the model to your own data, please preprocess the VF and CFP images following the procedures described in the paper. Raw fundus photographs and raw visual field data should not be input directly to the model.
