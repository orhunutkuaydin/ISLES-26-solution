#!/usr/bin/env python3

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


STRUCTURE = np.ones((3, 3, 3), dtype=np.uint8)


def load(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    if path.suffix == ".npz":
        with np.load(path) as archive:
            probabilities = archive["probabilities"]
        if probabilities.ndim != 4 or probabilities.shape[0] != 2:
            raise ValueError(f"{path} does not contain two-class probabilities")
        image = nib.load(path.with_suffix(".nii.gz"))
        probability = probabilities[1].astype(np.float32, copy=False).transpose(2, 1, 0)
        if probability.shape != image.shape:
            raise ValueError(f"{path} probabilities do not match the NIfTI shape")
        return image, probability
    image = nib.load(path)
    return image, image.get_fdata(dtype=np.float32)


def save(reference: nib.Nifti1Image, values: np.ndarray, path: Path, dtype) -> None:
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    image = nib.Nifti1Image(values.astype(dtype), reference.affine, header)
    _, qcode = reference.get_qform(coded=True)
    _, scode = reference.get_sform(coded=True)
    image.set_qform(reference.affine, int(qcode))
    image.set_sform(reference.affine, int(scode))
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, path)


def component_filter(
    mask901: np.ndarray, mask902: np.ndarray, voxel_ml: float
) -> np.ndarray:
    labels901, count901 = ndimage.label(mask901, structure=STRUCTURE)
    sizes901 = np.bincount(labels901.ravel(), minlength=count901 + 1)
    supported901 = np.zeros(count901 + 1, dtype=bool)
    supported901[np.unique(labels901[mask901 & mask902])] = True
    ids901 = np.arange(count901 + 1)
    remove901 = ids901[(ids901 > 0) & ~supported901 & (sizes901 * voxel_ml < 1.0)]
    merged = mask902 | (mask901 & ~np.isin(labels901, remove901))

    labels, count = ndimage.label(merged, structure=STRUCTURE)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    protected = np.zeros(count + 1, dtype=bool)
    protected[np.unique(labels[mask901 & mask902])] = True
    ids = np.arange(count + 1)
    remove = ids[(ids > 0) & ~protected & (sizes * voxel_ml <= 0.01)]
    return merged & ~np.isin(labels, remove)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("probability901", type=Path)
    parser.add_argument("probability902", type=Path)
    parser.add_argument("output_probability", type=Path)
    parser.add_argument("output_segmentation", type=Path)
    args = parser.parse_args()

    image901, probability901 = load(args.probability901)
    image902, probability902 = load(args.probability902)
    if probability901.shape != probability902.shape or not np.allclose(
        image901.affine, image902.affine, rtol=0, atol=1e-5
    ):
        raise ValueError("probability maps do not share a grid")
    if not np.isfinite(probability901).all() or not np.isfinite(probability902).all():
        raise ValueError("probability maps contain non-finite values")

    mask901 = probability901 > 0.5
    mask902 = probability902 > 0.5
    probability = np.maximum(probability901, probability902)
    voxel_ml = abs(float(np.linalg.det(image901.affine[:3, :3]))) / 1000.0
    segmentation = component_filter(mask901, mask902, voxel_ml)
    save(image901, probability, args.output_probability, np.float32)
    save(image901, segmentation, args.output_segmentation, np.uint8)


if __name__ == "__main__":
    main()
