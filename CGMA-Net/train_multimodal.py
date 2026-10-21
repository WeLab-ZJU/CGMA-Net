from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import MultiModalDataset, build_group_balanced_sampler, compute_class_weights, scan_multimodal_records
from losses import build_weighted_bce_from_class_weights, multimodal_total_loss
from metrics import binary_classification_metrics, subgroup_metrics
from models import (
    EncoderHead,
    MultiModalGatedClassifier,
    build_cfp_encoder_head,
    build_vf_encoder_head,
    load_cfp_encoder_head_checkpoint,
    load_vf_encoder_head_checkpoint,
)
from utils import (
    AverageMeter,
    CSVLogger,
    build_cfp_transforms,
    build_vf_transforms,
    collect_device,
    enable_mc_dropout,
    ensure_dir,
    save_checkpoint,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multimodal gated glaucoma classifier.")
    parser.add_argument("--data-root", type=str, default="../DataSet")
    parser.add_argument("--output-dir", type=str, default="outputs/multimodal")
    parser.add_argument("--cfp-checkpoint", type=str, default="")
    parser.add_argument("--vf-checkpoint", type=str, default="")
    parser.add_argument("--retfound-checkpoint", type=str, default="../RETFound/RETFound_mae_natureCFP.pth")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--modality-dropout-prob", type=float, default=0.5)
    parser.add_argument("--mc-samples", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use-group-balanced-sampler", action="store_true")
    parser.add_argument("--vf-image-size", type=int, default=224)
    parser.add_argument("--vf-patch-size", type=int, default=16)
    parser.add_argument("--vf-embed-dim", type=int, default=256)
    parser.add_argument("--vf-transformer-depth", type=int, default=6)
    parser.add_argument("--vf-transformer-heads", type=int, default=8)
    parser.add_argument("--vf-transformer-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--vf-encoder-dropout", type=float, default=0.1)
    parser.add_argument("--gate-entropy-weight", type=float, default=0.01)
    parser.add_argument("--uncertainty-loss-weight", type=float, default=0.05)
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> MultiModalGatedClassifier:
    if args.cfp_checkpoint:
        cfp_head: EncoderHead = load_cfp_encoder_head_checkpoint(
            checkpoint_path=args.cfp_checkpoint,
            retfound_checkpoint_path=args.retfound_checkpoint,
            feature_dim=args.feature_dim,
            dropout=args.dropout,
        )
    else:
        cfp_head = build_cfp_encoder_head(
            retfound_checkpoint_path=args.retfound_checkpoint,
            feature_dim=args.feature_dim,
            dropout=args.dropout,
        )

    if args.vf_checkpoint:
        vf_head = load_vf_encoder_head_checkpoint(
            checkpoint_path=args.vf_checkpoint,
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
    else:
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


def build_dataloaders(args: argparse.Namespace) -> Dict[str, DataLoader]:
    train_records = scan_multimodal_records(args.data_root, "TrainAug")
    valid_records = scan_multimodal_records(args.data_root, "Valid")
    test_records = scan_multimodal_records(args.data_root, "Test")

    datasets = {
        "train": MultiModalDataset(
            train_records,
            cfp_transform=build_cfp_transforms(train=True),
            vf_transform=build_vf_transforms(train=True),
            modality_dropout_prob=args.modality_dropout_prob,
            apply_modality_dropout=True,
        ),
        "valid": MultiModalDataset(
            valid_records,
            cfp_transform=build_cfp_transforms(train=False),
            vf_transform=build_vf_transforms(train=False),
            modality_dropout_prob=0.0,
            apply_modality_dropout=False,
        ),
        "test": MultiModalDataset(
            test_records,
            cfp_transform=build_cfp_transforms(train=False),
            vf_transform=build_vf_transforms(train=False),
            modality_dropout_prob=0.0,
            apply_modality_dropout=False,
        ),
    }

    sampler = build_group_balanced_sampler(train_records) if args.use_group_balanced_sampler else None
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        ),
        "valid": DataLoader(datasets["valid"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True),
    }
    return loaders


def run_epoch(
    model: MultiModalGatedClassifier,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    gate_entropy_weight: float,
    uncertainty_loss_weight: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    losses = AverageMeter()
    cls_losses = AverageMeter()
    uncertainty_losses = AverageMeter()
    gate_regs = AverageMeter()
    y_true: List[int] = []
    y_prob: List[float] = []
    sample_types: List[str] = []
    gate_c: List[float] = []
    gate_v: List[float] = []

    for batch in loader:
        cfp = batch["cfp"].to(device, non_blocking=True)
        vf = batch["vf"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        label = batch["label"].float().to(device, non_blocking=True)

        output = model(cfp, vf, mask)
        loss_dict = multimodal_total_loss(
            logits=output["logit"],
            targets=label,
            uncertainty=output["uncertainty"],
            gate_weights=output["gate_weights"],
            paired_mask=output["paired_mask"],
            criterion=criterion,
            gate_entropy_weight=gate_entropy_weight if is_train else 0.0,
            uncertainty_loss_weight=uncertainty_loss_weight if is_train else 0.0,
        )
        loss = loss_dict["total_loss"]

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        prob = torch.sigmoid(output["logit"])
        losses.update(loss.item(), cfp.size(0))
        cls_losses.update(float(loss_dict["cls_loss"].item()), cfp.size(0))
        uncertainty_losses.update(float(loss_dict["uncertainty_loss"].item()), cfp.size(0))
        gate_regs.update(float(loss_dict["gate_reg"].item()), cfp.size(0))

        y_true.extend(label.detach().cpu().numpy().astype(int).tolist())
        y_prob.extend(prob.detach().cpu().numpy().tolist())
        sample_types.extend(batch["sample_type"])
        gate_c.extend(output["gate_weights"][:, 0].detach().cpu().numpy().tolist())
        gate_v.extend(output["gate_weights"][:, 1].detach().cpu().numpy().tolist())

    metrics = binary_classification_metrics(y_true, y_prob)
    metrics["loss"] = losses.avg
    metrics["cls_loss"] = cls_losses.avg
    metrics["uncertainty_loss"] = uncertainty_losses.avg
    metrics["gate_reg"] = gate_regs.avg
    metrics["gate_c_mean"] = float(np.mean(gate_c)) if gate_c else float("nan")
    metrics["gate_v_mean"] = float(np.mean(gate_v)) if gate_v else float("nan")

    subgroup = subgroup_metrics(y_true, y_prob, sample_types)
    for group_name, group_metrics in subgroup.items():
        metrics[f"{group_name}_auc"] = group_metrics.get("auc", float("nan"))
        metrics[f"{group_name}_auprc"] = group_metrics.get("auprc", float("nan"))
        metrics[f"{group_name}_acc"] = group_metrics.get("accuracy", float("nan"))
        metrics[f"{group_name}_count"] = group_metrics.get("count", float("nan"))
    return metrics


@torch.no_grad()
def evaluate_with_mc_dropout(
    model: MultiModalGatedClassifier,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    mc_samples: int,
    prediction_path: Optional[Path] = None,
) -> Dict[str, float]:
    model.eval()
    enable_mc_dropout(model)

    losses = AverageMeter()
    cls_losses = AverageMeter()
    uncertainty_losses = AverageMeter()
    gate_regs = AverageMeter()
    y_true: List[int] = []
    y_prob: List[float] = []
    y_uncertainty: List[float] = []
    sample_types: List[str] = []
    rows: List[Dict] = []

    for batch in loader:
        cfp = batch["cfp"].to(device, non_blocking=True)
        vf = batch["vf"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        label = batch["label"].float().to(device, non_blocking=True)

        prob_samples = []
        uncertainty_head_samples = []
        for _ in range(mc_samples):
            output = model(cfp, vf, mask)
            prob_samples.append(torch.sigmoid(output["logit"]).unsqueeze(0))
            uncertainty_head_samples.append(output["uncertainty"].unsqueeze(0))

        prob_stack = torch.cat(prob_samples, dim=0)
        uncertainty_stack = torch.cat(uncertainty_head_samples, dim=0)
        mean_prob = prob_stack.mean(dim=0)
        epistemic_uncertainty = prob_stack.var(dim=0, unbiased=False)
        mean_head_uncertainty = uncertainty_stack.mean(dim=0)

        deterministic_output = model(cfp, vf, mask)
        loss_dict = multimodal_total_loss(
            logits=deterministic_output["logit"],
            targets=label,
            uncertainty=deterministic_output["uncertainty"],
            gate_weights=deterministic_output["gate_weights"],
            paired_mask=deterministic_output["paired_mask"],
            criterion=criterion,
            gate_entropy_weight=0.0,
            uncertainty_loss_weight=0.0,
        )
        losses.update(float(loss_dict["total_loss"].item()), cfp.size(0))
        cls_losses.update(float(loss_dict["cls_loss"].item()), cfp.size(0))
        uncertainty_losses.update(float(loss_dict["uncertainty_loss"].item()), cfp.size(0))
        gate_regs.update(float(loss_dict["gate_reg"].item()), cfp.size(0))

        batch_true = label.detach().cpu().numpy().astype(int).tolist()
        batch_prob = mean_prob.detach().cpu().numpy().tolist()
        batch_unc = epistemic_uncertainty.detach().cpu().numpy().tolist()
        y_true.extend(batch_true)
        y_prob.extend(batch_prob)
        y_uncertainty.extend(batch_unc)
        sample_types.extend(batch["sample_type"])

        for idx in range(len(batch_true)):
            rows.append(
                {
                    "patient_id": batch["patient_id"][idx],
                    "label": batch_true[idx],
                    "sample_type": batch["sample_type"][idx],
                    "cfp_path": batch["cfp_path"][idx],
                    "vf_path": batch["vf_path"][idx],
                    "prob_mean": batch_prob[idx],
                    "uncertainty_mc_var": batch_unc[idx],
                    "uncertainty_head_mean": float(mean_head_uncertainty[idx].item()),
                    "mask_cfp": float(mask[idx, 0].item()),
                    "mask_vf": float(mask[idx, 1].item()),
                }
            )

    metrics = binary_classification_metrics(y_true, y_prob)
    metrics["loss"] = losses.avg
    metrics["cls_loss"] = cls_losses.avg
    metrics["uncertainty_loss"] = uncertainty_losses.avg
    metrics["gate_reg"] = gate_regs.avg
    metrics["uncertainty_mean"] = float(np.mean(y_uncertainty)) if y_uncertainty else float("nan")

    subgroup = subgroup_metrics(y_true, y_prob, sample_types)
    for group_name, group_metrics in subgroup.items():
        metrics[f"{group_name}_auc"] = group_metrics.get("auc", float("nan"))
        metrics[f"{group_name}_auprc"] = group_metrics.get("auprc", float("nan"))
        metrics[f"{group_name}_acc"] = group_metrics.get("accuracy", float("nan"))
        metrics[f"{group_name}_count"] = group_metrics.get("count", float("nan"))

    if prediction_path is not None:
        logger = CSVLogger(
            prediction_path,
            fieldnames=[
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
            ],
        )
        for row in rows:
            logger.log(row)
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = collect_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    loaders = build_dataloaders(args)
    train_records = scan_multimodal_records(args.data_root, "TrainAug")
    class_weights = compute_class_weights(train_records)

    model = build_model(args)
    model.to(device)

    criterion = build_weighted_bce_from_class_weights(class_weights, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    metric_fields = [
        "epoch",
        "split",
        "loss",
        "cls_loss",
        "uncertainty_loss",
        "gate_reg",
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "auc",
        "auprc",
        "brier",
        "ece",
        "gate_c_mean",
        "gate_v_mean",
        "cfp_auc",
        "vf_auc",
        "pair_auc",
        "cfp_auprc",
        "vf_auprc",
        "pair_auprc",
        "cfp_acc",
        "vf_acc",
        "pair_acc",
        "cfp_count",
        "vf_count",
        "pair_count",
    ]
    epoch_logger = CSVLogger(output_dir / "epoch_metrics.csv", fieldnames=metric_fields)
    best_auc = -np.inf
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            gate_entropy_weight=args.gate_entropy_weight,
            uncertainty_loss_weight=args.uncertainty_loss_weight,
            optimizer=optimizer,
        )
        valid_metrics = run_epoch(
            model,
            loaders["valid"],
            criterion,
            device,
            gate_entropy_weight=0.0,
            uncertainty_loss_weight=0.0,
            optimizer=None,
        )
        scheduler.step(valid_metrics["auc"])

        for split_name, metrics in [("train", train_metrics), ("valid", valid_metrics)]:
            row = {"epoch": epoch, "split": split_name}
            row.update({key: metrics.get(key, float("nan")) for key in metric_fields if key not in {"epoch", "split"}})
            epoch_logger.log(row)

        if valid_metrics["auc"] > best_auc:
            best_auc = valid_metrics["auc"]
            save_checkpoint(
                best_path,
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                    "best_valid_auc": best_auc,
                },
            )

    checkpoint = torch.load(best_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    valid_mc_metrics = evaluate_with_mc_dropout(
        model,
        loaders["valid"],
        criterion,
        device,
        mc_samples=args.mc_samples,
        prediction_path=output_dir / "valid_predictions_mc_dropout.csv",
    )
    test_mc_metrics = evaluate_with_mc_dropout(
        model,
        loaders["test"],
        criterion,
        device,
        mc_samples=args.mc_samples,
        prediction_path=output_dir / "test_predictions_mc_dropout.csv",
    )

    summary_logger = CSVLogger(
        output_dir / "mc_dropout_metrics.csv",
        fieldnames=[
            "split",
            "loss",
            "cls_loss",
            "uncertainty_loss",
            "gate_reg",
            "accuracy",
            "sensitivity",
            "specificity",
            "precision",
            "f1",
            "auc",
            "auprc",
            "brier",
            "ece",
            "uncertainty_mean",
            "cfp_auc",
            "vf_auc",
            "pair_auc",
            "cfp_auprc",
            "vf_auprc",
            "pair_auprc",
            "cfp_acc",
            "vf_acc",
            "pair_acc",
            "cfp_count",
            "vf_count",
            "pair_count",
        ],
    )
    summary_logger.log({"split": "valid_mc", **valid_mc_metrics})
    summary_logger.log({"split": "test_mc", **test_mc_metrics})

    save_json(
        output_dir / "summary.json",
        {
            "args": vars(args),
            "class_weights": class_weights,
            "best_valid_auc": best_auc,
            "valid_mc_metrics": valid_mc_metrics,
            "test_mc_metrics": test_mc_metrics,
        },
    )


if __name__ == "__main__":
    main()
