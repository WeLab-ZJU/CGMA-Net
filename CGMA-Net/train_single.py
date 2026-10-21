from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import SingleModalityDataset, compute_class_weights, scan_single_modality_records
from losses import build_weighted_bce_from_class_weights
from metrics import binary_classification_metrics
from models import EncoderHead, SingleModalityClassifier, build_cfp_encoder_head, build_vf_encoder_head
from utils import (
    AverageMeter,
    CSVLogger,
    build_cfp_transforms,
    build_vf_transforms,
    collect_device,
    ensure_dir,
    save_checkpoint,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a single-modality glaucoma classifier.")
    parser.add_argument("--data-root", type=str, default="../DataSet", help="Project root containing TrainAug/Valid/Test.")
    parser.add_argument("--modality", type=str, choices=["cfp", "vf"], required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=120457)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--retfound-checkpoint", type=str, default="../RETFound/RETFound_mae_natureCFP.pth")
    parser.add_argument("--vf-image-size", type=int, default=224)
    parser.add_argument("--vf-patch-size", type=int, default=16)
    parser.add_argument("--vf-embed-dim", type=int, default=256)
    parser.add_argument("--vf-transformer-depth", type=int, default=6)
    parser.add_argument("--vf-transformer-heads", type=int, default=8)
    parser.add_argument("--vf-transformer-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--vf-encoder-dropout", type=float, default=0.1)
    return parser.parse_args()


def build_dataloaders(args: argparse.Namespace) -> Dict[str, DataLoader]:
    transform_fn = build_cfp_transforms if args.modality == "cfp" else build_vf_transforms
    train_records = scan_single_modality_records(args.data_root, "TrainAug", args.modality)
    valid_records = scan_single_modality_records(args.data_root, "Valid", args.modality)
    test_records = scan_single_modality_records(args.data_root, "Test", args.modality)

    datasets = {
        "train": SingleModalityDataset(train_records, transform=transform_fn(train=True)),
        "valid": SingleModalityDataset(valid_records, transform=transform_fn(train=False)),
        "test": SingleModalityDataset(test_records, transform=transform_fn(train=False)),
    }

    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            pin_memory=True,
        )
        for split, dataset in datasets.items()
    }
    return loaders


def build_single_model(args: argparse.Namespace) -> SingleModalityClassifier:
    if args.modality == "cfp":
        encoder_head: EncoderHead = build_cfp_encoder_head(
            retfound_checkpoint_path=args.retfound_checkpoint,
            feature_dim=args.feature_dim,
            dropout=args.dropout,
        )
    else:
        encoder_head = build_vf_encoder_head(
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
    return SingleModalityClassifier(encoder_head=encoder_head, feature_dim=args.feature_dim, dropout=args.dropout)


def run_epoch(
    model: SingleModalityClassifier,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    losses = AverageMeter()
    y_true: List[int] = []
    y_prob: List[float] = []

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].float().to(device, non_blocking=True)
        output = model(image)
        logit = output["logit"]
        loss = criterion(logit, label).mean()

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        prob = torch.sigmoid(logit)
        losses.update(loss.item(), image.size(0))
        y_true.extend(label.detach().cpu().numpy().astype(int).tolist())
        y_prob.extend(prob.detach().cpu().numpy().tolist())

    metrics = binary_classification_metrics(y_true, y_prob)
    metrics["loss"] = losses.avg
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = collect_device(args.device)

    output_dir = ensure_dir(Path(args.output_dir) / f"single_{args.modality}")
    loaders = build_dataloaders(args)
    train_records = scan_single_modality_records(args.data_root, "TrainAug", args.modality)
    class_weights = compute_class_weights(train_records)

    model = build_single_model(args)
    model.to(device)

    criterion = build_weighted_bce_from_class_weights(class_weights, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    metrics_fields = [
        "epoch",
        "split",
        "loss",
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "auc",
        "auprc",
        "brier",
        "ece",
    ]
    metrics_logger = CSVLogger(output_dir / "epoch_metrics.csv", fieldnames=metrics_fields)

    best_auc = -np.inf
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        print(f'{epoch}/{args.epochs + 1}')
        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer=optimizer)
        valid_metrics = run_epoch(model, loaders["valid"], criterion, device, optimizer=None)
        scheduler.step(valid_metrics["auc"])

        for split_name, metrics in [("train", train_metrics), ("valid", valid_metrics)]:
            row = {"epoch": epoch, "split": split_name}
            row.update({key: metrics.get(key) for key in metrics_fields if key not in {"epoch", "split"}})
            metrics_logger.log(row)

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
    test_metrics = run_epoch(model, loaders["test"], criterion, device, optimizer=None)
    test_logger = CSVLogger(output_dir / "test_metrics.csv", fieldnames=["split"] + metrics_fields[2:])
    test_logger.log({"split": "test", **{key: test_metrics.get(key) for key in metrics_fields[2:]}})

    save_json(
        output_dir / "summary.json",
        {
            "args": vars(args),
            "class_weights": class_weights,
            "best_valid_auc": best_auc,
            "test_metrics": test_metrics,
        },
    )


if __name__ == "__main__":
    main()
