from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from datasets import MultiModalDataset, scan_multimodal_records
from models import EncoderHead, MultiModalGatedClassifier, RETFoundCFPEncoder, build_vf_encoder_head
from utils import build_cfp_transforms, build_vf_transforms, collect_device, enable_mc_dropout, set_seed


GROUP_ORDER = ["overall", "cfp", "vf", "pair"]
MetricFn = Callable[[np.ndarray, np.ndarray], float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RETFound+CrossAtten test metrics with 95% CIs.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/multimodal/best_model.pt"))
    parser.add_argument("--predictions", type=Path, default=Path("outputs/multimodal/test_predictions_mc_dropout.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/multimodal/statistics_csv"))
    parser.add_argument("--data-root", type=str, default=None, help="Override checkpoint data_root if needed.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mc-samples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=120456)
    parser.add_argument(
        "--source",
        choices=["auto", "checkpoint", "predictions"],
        default="auto",
        help="auto uses checkpoint if present, otherwise the predictions CSV.",
    )
    return parser.parse_args()


def checkpoint_args(checkpoint: Dict, cli_args: argparse.Namespace) -> SimpleNamespace:
    saved = dict(checkpoint.get("args", {}))
    saved["data_root"] = cli_args.data_root or saved.get("data_root", "../DataSet")
    saved["batch_size"] = cli_args.batch_size or int(saved.get("batch_size", 8))
    saved["num_workers"] = cli_args.num_workers
    saved["mc_samples"] = cli_args.mc_samples or int(saved.get("mc_samples", 20))
    saved["device"] = cli_args.device
    saved.setdefault("feature_dim", 256)
    saved.setdefault("dropout", 0.3)
    saved.setdefault("vf_image_size", 224)
    saved.setdefault("vf_patch_size", 16)
    saved.setdefault("vf_embed_dim", 256)
    saved.setdefault("vf_transformer_depth", 6)
    saved.setdefault("vf_transformer_heads", 8)
    saved.setdefault("vf_transformer_mlp_ratio", 4.0)
    saved.setdefault("vf_encoder_dropout", 0.1)
    return SimpleNamespace(**saved)


def build_model_from_checkpoint_args(args: SimpleNamespace) -> MultiModalGatedClassifier:
    cfp_head = EncoderHead(
        encoder=RETFoundCFPEncoder(),
        feature_dim=args.feature_dim,
        dropout=args.dropout,
    )
    vf_head = build_vf_encoder_head(
        feature_dim=args.feature_dim,
        dropout=args.dropout,
        image_size=args.vf_image_size,
        patch_size=args.vf_patch_size,
        embed_dim=args.vf_embed_dim,
        depth=args.vf_transformer_depth,
        num_heads=args.vf_transformer_heads,
        mlp_ratio=args.vf_transformer_mlp_ratio,
        encoder_dropout=args.vf_encoder_dropout,
    )
    return MultiModalGatedClassifier(
        cfp_encoder_head=cfp_head,
        vf_encoder_head=vf_head,
        feature_dim=args.feature_dim,
        dropout=args.dropout,
    )


def build_test_loader(args: SimpleNamespace) -> DataLoader:
    records = scan_multimodal_records(args.data_root, "Test")
    dataset = MultiModalDataset(
        records,
        cfp_transform=build_cfp_transforms(train=False),
        vf_transform=build_vf_transforms(train=False),
        modality_dropout_prob=0.0,
        apply_modality_dropout=False,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


@torch.no_grad()
def generate_predictions_from_checkpoint(cli_args: argparse.Namespace) -> List[Dict[str, object]]:
    checkpoint = torch.load(cli_args.checkpoint, map_location="cpu")
    args = checkpoint_args(checkpoint, cli_args)
    set_seed(cli_args.seed)
    device = collect_device(args.device)

    model = build_model_from_checkpoint_args(args)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    enable_mc_dropout(model)

    loader = build_test_loader(args)
    rows: List[Dict[str, object]] = []
    for batch in loader:
        cfp = batch["cfp"].to(device, non_blocking=True)
        vf = batch["vf"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        label = batch["label"].float().to(device, non_blocking=True)

        prob_samples = []
        uncertainty_samples = []
        for _ in range(args.mc_samples):
            output = model(cfp, vf, mask)
            prob_samples.append(torch.sigmoid(output["logit"]).unsqueeze(0))
            uncertainty_samples.append(output["uncertainty"].unsqueeze(0))

        prob_stack = torch.cat(prob_samples, dim=0)
        uncertainty_stack = torch.cat(uncertainty_samples, dim=0)
        mean_prob = prob_stack.mean(dim=0)
        mc_var = prob_stack.var(dim=0, unbiased=False)
        mean_head_uncertainty = uncertainty_stack.mean(dim=0)

        batch_true = label.detach().cpu().numpy().astype(int).tolist()
        batch_prob = mean_prob.detach().cpu().numpy().tolist()
        batch_var = mc_var.detach().cpu().numpy().tolist()

        for idx in range(len(batch_true)):
            rows.append(
                {
                    "patient_id": batch["patient_id"][idx],
                    "label": batch_true[idx],
                    "sample_type": batch["sample_type"][idx],
                    "cfp_path": batch["cfp_path"][idx],
                    "vf_path": batch["vf_path"][idx],
                    "prob_mean": batch_prob[idx],
                    "uncertainty_mc_var": batch_var[idx],
                    "uncertainty_head_mean": float(mean_head_uncertainty[idx].item()),
                    "mask_cfp": float(mask[idx, 0].item()),
                    "mask_vf": float(mask[idx, 1].item()),
                }
            )
    return rows


def read_predictions(path: Path) -> List[Dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as file_obj:
        return [dict(row) for row in csv.DictReader(file_obj)]


def load_or_generate_predictions(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], str]:
    use_checkpoint = args.source == "checkpoint" or (args.source == "auto" and args.checkpoint.exists())
    if use_checkpoint:
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        rows = generate_predictions_from_checkpoint(args)
        return rows, "checkpoint"
    if not args.predictions.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {args.predictions}")
    return read_predictions(args.predictions), "predictions"


def write_prediction_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = [
        "patient_id",
        "label",
        "sample_type",
        "cfp_path",
        "vf_path",
        "prob_mean",
        "uncertainty_mc_var",
        "uncertainty_head_mean",
        "mask_cfp",
        "mask_vf",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= start) & ((y_prob < end) if end < 1.0 else (y_prob <= end))
        if mask.sum() == 0:
            continue
        ece += abs(y_prob[mask].mean() - y_true[mask].mean()) * (mask.sum() / len(y_true))
    return float(ece)


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")


def safe_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": safe_auc(y_true, y_prob),
        "auprc": safe_auprc(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: MetricFn,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    values: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        value = metric_fn(y_true[idx], y_prob[idx])
        if np.isfinite(value):
            values.append(float(value))
    if not values:
        return float("nan"), float("nan")
    lower, upper = np.percentile(values, [2.5, 97.5])
    return float(lower), float(upper)


def rows_for_group(rows: List[Dict[str, object]], group: str) -> List[Dict[str, object]]:
    if group == "overall":
        return rows
    return [row for row in rows if str(row["sample_type"]) == group]


def summarize_group(
    rows: List[Dict[str, object]],
    group: str,
    threshold: float,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    y_true = np.array([int(row["label"]) for row in rows], dtype=int)
    y_prob = np.array([float(row["prob_mean"]) for row in rows], dtype=float)
    mc_unc = np.array([float(row["uncertainty_mc_var"]) for row in rows], dtype=float)
    head_unc = np.array([float(row["uncertainty_head_mean"]) for row in rows], dtype=float)
    metrics = binary_metrics(y_true, y_prob, threshold)
    metric_fns: Dict[str, MetricFn] = {
        "accuracy": lambda yt, yp: binary_metrics(yt, yp, threshold)["accuracy"],
        "sensitivity": lambda yt, yp: binary_metrics(yt, yp, threshold)["sensitivity"],
        "specificity": lambda yt, yp: binary_metrics(yt, yp, threshold)["specificity"],
        "precision": lambda yt, yp: binary_metrics(yt, yp, threshold)["precision"],
        "f1": lambda yt, yp: binary_metrics(yt, yp, threshold)["f1"],
        "auc": safe_auc,
        "auprc": safe_auprc,
        "brier": lambda yt, yp: float(brier_score_loss(yt, yp)),
        "ece": expected_calibration_error,
    }
    long_rows: List[Dict[str, object]] = []
    for metric, value in metrics.items():
        if metric in {"tp", "tn", "fp", "fn"}:
            ci_lower, ci_upper, display = "", "", f"{value:.0f}"
        else:
            ci_lower, ci_upper = bootstrap_ci(y_true, y_prob, metric_fns[metric], n_bootstrap, rng)
            display = f"{value:.4f} ({ci_lower:.4f}-{ci_upper:.4f})"
        long_rows.append(
            {
                "group": group,
                "metric": metric,
                "value": value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "value_with_95ci": display,
                "n": len(rows),
                "threshold": threshold,
                "bootstrap_samples": n_bootstrap,
            }
        )
    extra = {
        "prob_mean_mean": float(y_prob.mean()),
        "uncertainty_mc_var_mean": float(mc_unc.mean()),
        "uncertainty_head_mean_mean": float(head_unc.mean()),
    }
    for metric, value in extra.items():
        long_rows.append(
            {
                "group": group,
                "metric": metric,
                "value": value,
                "ci_lower": "",
                "ci_upper": "",
                "value_with_95ci": f"{value:.6f}",
                "n": len(rows),
                "threshold": threshold,
                "bootstrap_samples": n_bootstrap,
            }
        )
    wide = {"group": group, "n": len(rows)}
    for row in long_rows:
        wide[str(row["metric"])] = row["value_with_95ci"]
    return long_rows, wide


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    rows, source = load_or_generate_predictions(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_prediction_csv(args.output_dir / "test_predictions_used_for_statistics.csv", rows)

    rng = np.random.default_rng(args.seed)
    long_rows: List[Dict[str, object]] = []
    wide_rows: List[Dict[str, object]] = []
    for group in GROUP_ORDER:
        group_rows = rows_for_group(rows, group)
        if not group_rows:
            continue
        group_long, group_wide = summarize_group(group_rows, group, args.threshold, args.bootstrap, rng)
        long_rows.extend(group_long)
        wide_rows.append(group_wide)

    long_fields = ["group", "metric", "value", "ci_lower", "ci_upper", "value_with_95ci", "n", "threshold", "bootstrap_samples"]
    write_csv(args.output_dir / "test_metrics_with_95ci_long.csv", long_rows, long_fields)
    wide_fields = [
        "group",
        "n",
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "auc",
        "auprc",
        "brier",
        "ece",
        "tp",
        "tn",
        "fp",
        "fn",
        "prob_mean_mean",
        "uncertainty_mc_var_mean",
        "uncertainty_head_mean_mean",
    ]
    write_csv(args.output_dir / "test_metrics_with_95ci_wide.csv", wide_rows, wide_fields)
    print(f"Source: {source}")
    print(f"Saved statistics CSV files to {args.output_dir}")


if __name__ == "__main__":
    main()
