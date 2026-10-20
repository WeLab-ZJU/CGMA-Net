# CGMA-Net

This folder contains the minimal code set required for **inference only**.
It does not include training code or model weights. The trained model weights have been uploaded to Hugging Face and are publicly available at: https://huggingface.co/zjuzjz/CGMA-Net.

## Files

| File | Purpose |
|---|---|
| `models.py` | Model architecture: RETFound CFP encoder, VF patch transformer encoder, cross-modal gated fusion, classifier head and uncertainty head |
| `datasets.py` | Scans the `Test/` directory and builds multimodal samples (`cfp` / `vf` / `pair`), producing images and modality masks |
| `utils.py` | Image preprocessing transforms, device selection, MC Dropout enabling, random seed |
| `summarize_test_metrics.py` | Inference entry point: loads a checkpoint, runs MC Dropout inference, and outputs per-sample predictions with 95% CIs |
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

## Notes

- Model weights are not included in this repository. Download them from [Hugging face](https://huggingface.co/zjuzjz/CGMA-Net).
- `summarize_test_metrics.py` performs batch inference over the `Test/` directory. For online inference on an arbitrary CFP + VF image pair, wrap the `MultiModalGatedClassifier` from `models.py` in your own script.
- The RETFound pretrained backbone is only needed for training. At inference time the encoder is built from scratch and the complete trained weights are loaded from the checkpoint.
- When applying the model to your own data, please preprocess the VF and CFP images following the procedures described in the paper. Raw fundus photographs and raw visual field data should not be input directly to the model.
