from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


SPLITS = ("TrainAug", "Valid", "Test")
LABELS = ("0", "1")
MODALITIES = ("cfp", "vf", "pair")


@dataclass
class SampleRecord:
    image_path: Optional[Path]
    label: int
    modality: str
    patient_id: str
    split: str


@dataclass
class MultiModalRecord:
    cfp_path: Optional[Path]
    vf_path: Optional[Path]
    label: int
    sample_type: str
    patient_id: str
    split: str


def default_patient_id(path: Path) -> str:
    return path.stem


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_pair_images(pair_dir: Path) -> Dict[str, Optional[Path]]:
    cfp_path: Optional[Path] = None
    vf_path: Optional[Path] = None
    for item in sorted(pair_dir.iterdir()):
        if not item.is_file() or not _is_image(item):
            continue
        if item.name.lower().startswith("vf_"):
            vf_path = item
        else:
            cfp_path = item
    return {"cfp": cfp_path, "vf": vf_path}


def scan_single_modality_records(root_dir: str | Path, split: str, modality: str) -> List[SampleRecord]:
    root = Path(root_dir)
    if split not in SPLITS:
        raise ValueError(f"Unsupported split: {split}")
    if modality not in ("cfp", "vf"):
        raise ValueError(f"Unsupported modality: {modality}")

    records: List[SampleRecord] = []
    split_dir = root / split
    for label_name in LABELS:
        label = int(label_name)
        label_dir = split_dir / label_name

        direct_dir = label_dir / modality
        if direct_dir.exists():
            for image_path in sorted(direct_dir.iterdir()):
                if image_path.is_file() and _is_image(image_path):
                    records.append(
                        SampleRecord(
                            image_path=image_path,
                            label=label,
                            modality=modality,
                            patient_id=default_patient_id(image_path),
                            split=split,
                        )
                    )

        pair_root = label_dir / "pair"
        if pair_root.exists():
            for patient_dir in sorted(pair_root.iterdir()):
                if not patient_dir.is_dir():
                    continue
                pair_images = find_pair_images(patient_dir)
                image_path = pair_images[modality]
                if image_path is None:
                    continue
                records.append(
                    SampleRecord(
                        image_path=image_path,
                        label=label,
                        modality=modality,
                        patient_id=patient_dir.name,
                        split=split,
                    )
                )
    return records


def scan_multimodal_records(root_dir: str | Path, split: str) -> List[MultiModalRecord]:
    root = Path(root_dir)
    if split not in SPLITS:
        raise ValueError(f"Unsupported split: {split}")

    records: List[MultiModalRecord] = []
    split_dir = root / split
    for label_name in LABELS:
        label = int(label_name)
        label_dir = split_dir / label_name

        cfp_dir = label_dir / "cfp"
        if cfp_dir.exists():
            for image_path in sorted(cfp_dir.iterdir()):
                if image_path.is_file() and _is_image(image_path):
                    records.append(
                        MultiModalRecord(
                            cfp_path=image_path,
                            vf_path=None,
                            label=label,
                            sample_type="cfp",
                            patient_id=default_patient_id(image_path),
                            split=split,
                        )
                    )

        vf_dir = label_dir / "vf"
        if vf_dir.exists():
            for image_path in sorted(vf_dir.iterdir()):
                if image_path.is_file() and _is_image(image_path):
                    records.append(
                        MultiModalRecord(
                            cfp_path=None,
                            vf_path=image_path,
                            label=label,
                            sample_type="vf",
                            patient_id=default_patient_id(image_path),
                            split=split,
                        )
                    )

        pair_root = label_dir / "pair"
        if pair_root.exists():
            for patient_dir in sorted(pair_root.iterdir()):
                if not patient_dir.is_dir():
                    continue
                pair_images = find_pair_images(patient_dir)
                if pair_images["cfp"] is None or pair_images["vf"] is None:
                    continue
                records.append(
                    MultiModalRecord(
                        cfp_path=pair_images["cfp"],
                        vf_path=pair_images["vf"],
                        label=label,
                        sample_type="pair",
                        patient_id=patient_dir.name,
                        split=split,
                    )
                )
    return records


def pil_loader(path: Path, image_mode: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert(image_mode)


class SingleModalityDataset(Dataset):
    def __init__(
        self,
        records: List[SampleRecord],
        transform: Optional[Callable] = None,
        image_mode: Optional[str] = None,
    ) -> None:
        self.records = records
        self.transform = transform
        if image_mode is None:
            image_mode = "RGB" if records and records[0].modality == "cfp" else "L"
        self.image_mode = image_mode

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | int | str]:
        record = self.records[index]
        image = pil_loader(record.image_path, self.image_mode)
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": record.label,
            "patient_id": record.patient_id,
            "modality": record.modality,
            "path": str(record.image_path),
        }


class MultiModalDataset(Dataset):
    def __init__(
        self,
        records: List[MultiModalRecord],
        cfp_transform: Optional[Callable] = None,
        vf_transform: Optional[Callable] = None,
        modality_dropout_prob: float = 0.0,
        apply_modality_dropout: bool = False,
    ) -> None:
        self.records = records
        self.cfp_transform = cfp_transform
        self.vf_transform = vf_transform
        self.modality_dropout_prob = modality_dropout_prob
        self.apply_modality_dropout = apply_modality_dropout

    def __len__(self) -> int:
        return len(self.records)

    def _load_cfp(self, path: Optional[Path]) -> torch.Tensor:
        if path is None:
            return torch.zeros(3, 224, 224, dtype=torch.float32)
        image = pil_loader(path, "RGB")
        if self.cfp_transform is not None:
            image = self.cfp_transform(image)
        return image

    def _load_vf(self, path: Optional[Path]) -> torch.Tensor:
        if path is None:
            return torch.zeros(1, 224, 224, dtype=torch.float32)
        image = pil_loader(path, "L")
        if self.vf_transform is not None:
            image = self.vf_transform(image)
        return image

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | int | str]:
        record = self.records[index]
        cfp_available = 1 if record.cfp_path is not None else 0
        vf_available = 1 if record.vf_path is not None else 0

        if self.apply_modality_dropout and record.sample_type == "pair" and self.modality_dropout_prob > 0:
            random_value = torch.rand(1).item()
            if random_value < self.modality_dropout_prob / 2:
                cfp_available = 0
            elif random_value < self.modality_dropout_prob:
                vf_available = 0

        cfp_image = self._load_cfp(record.cfp_path if cfp_available else None)
        vf_image = self._load_vf(record.vf_path if vf_available else None)

        mask = torch.tensor([cfp_available, vf_available], dtype=torch.float32)
        observed_type = "pair"
        if cfp_available and not vf_available:
            observed_type = "cfp"
        elif vf_available and not cfp_available:
            observed_type = "vf"

        return {
            "cfp": cfp_image,
            "vf": vf_image,
            "mask": mask,
            "label": record.label,
            "patient_id": record.patient_id,
            "sample_type": observed_type,
            "original_sample_type": record.sample_type,
            "cfp_path": "" if record.cfp_path is None else str(record.cfp_path),
            "vf_path": "" if record.vf_path is None else str(record.vf_path),
        }


def compute_class_weights(records: List[SampleRecord] | List[MultiModalRecord]) -> Dict[int, float]:
    counts = {0: 0, 1: 0}
    for record in records:
        counts[int(record.label)] += 1
    total = counts[0] + counts[1]
    weights = {}
    for label, count in counts.items():
        weights[label] = total / max(2 * count, 1)
    return weights


def build_group_balanced_sampler(records: List[MultiModalRecord]) -> WeightedRandomSampler:
    group_counts: Dict[str, int] = {}
    sample_weights: List[float] = []
    for record in records:
        key = f"{record.sample_type}_{record.label}"
        group_counts[key] = group_counts.get(key, 0) + 1
    for record in records:
        key = f"{record.sample_type}_{record.label}"
        sample_weights.append(1.0 / group_counts[key])
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler
