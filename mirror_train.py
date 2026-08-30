#!/usr/bin/env python3
"""Generate the mirror channel used to construct the Dataset902 training set."""

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


PAIRS = (
    (2, 41),
    (3, 42),
    (4, 43),
    (5, 44),
    (7, 46),
    (8, 47),
    (10, 49),
    (11, 50),
    (12, 51),
    (13, 52),
    (17, 53),
    (18, 54),
    (26, 58),
    (28, 60),
)
LEFT = tuple(pair[0] for pair in PAIRS)
RIGHT = tuple(pair[1] for pair in PAIRS)


def run(command: list[str], threads: int) -> None:
    environment = os.environ.copy()
    environment["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
    subprocess.run([str(item) for item in command], check=True, env=environment)


def save(reference: nib.Nifti1Image, values: np.ndarray, path: Path, dtype) -> None:
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    header.set_slope_inter(1.0, 0.0)
    image = nib.Nifti1Image(values.astype(dtype), reference.affine, header)
    _, qcode = reference.get_qform(coded=True)
    _, scode = reference.get_sform(coded=True)
    image.set_qform(reference.affine, int(qcode))
    image.set_sform(reference.affine, int(scode))
    nib.save(image, path)


def brain_mask(values: np.ndarray) -> np.ndarray:
    foreground = np.isfinite(values) & (values > 0)
    labels, count = ndimage.label(foreground)
    if count == 0:
        raise ValueError("input has no nonzero foreground")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    mask = labels == int(np.argmax(sizes))
    return ndimage.binary_fill_holes(ndimage.binary_closing(mask, iterations=2))


def fit_plane(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = np.arange(left.shape[0], dtype=np.int16)[:, None, None]
    left_max = np.where(left, x, -1).max(axis=0)
    right_min = np.where(right, x, left.shape[0]).min(axis=0)
    valid = (left_max >= 0) & (right_min < left.shape[0]) & (left_max < right_min)
    y, z = np.nonzero(valid)
    points = np.column_stack(((left_max[valid] + right_min[valid]) / 2.0, y, z))
    if len(points) < 100:
        raise ValueError("too few bilateral interface samples")
    keep = np.ones(len(points), dtype=bool)
    for _ in range(8):
        design = np.column_stack(
            (points[keep, 1], points[keep, 2], np.ones(keep.sum()))
        )
        coefficient = np.linalg.lstsq(design, points[keep, 0], rcond=None)[0]
        residual = points[:, 0] - (
            coefficient[0] * points[:, 1]
            + coefficient[1] * points[:, 2]
            + coefficient[2]
        )
        center = np.median(residual[keep])
        mad = np.median(np.abs(residual[keep] - center))
        new_keep = np.abs(residual - center) <= max(2.0, 3.5 * 1.4826 * mad)
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return coefficient


def balance_intercept(
    coefficient: np.ndarray, brain: np.ndarray, left: np.ndarray, right: np.ndarray
) -> float:
    y, z = np.indices(brain.shape[1:])
    base = (coefficient[0] * y + coefficient[1] * z + coefficient[2]).ravel()
    columns = base.size
    cumulative_brain = np.cumsum(brain, axis=0, dtype=np.int32).reshape(
        brain.shape[0], columns
    )
    cumulative_left = np.cumsum(left, axis=0, dtype=np.int32).reshape(
        brain.shape[0], columns
    )
    cumulative_right = np.cumsum(right, axis=0, dtype=np.int32).reshape(
        brain.shape[0], columns
    )
    total_brain = cumulative_brain[-1]
    total_left = cumulative_left[-1]
    index = np.arange(columns)[None, :]

    def evaluate(offsets: np.ndarray) -> list[tuple[tuple[int, int, float], float]]:
        starts = np.rint(base[None, :] + offsets[:, None] - 0.5).astype(np.int16)
        np.clip(starts, 1, brain.shape[0] - 3, out=starts)
        left_count = cumulative_brain[starts - 1, index].sum(axis=1, dtype=np.int64)
        right_count = (total_brain[None, :] - cumulative_brain[starts + 1, index]).sum(
            axis=1, dtype=np.int64
        )
        wrong = (
            total_left[None, :]
            - cumulative_left[starts + 1, index]
            + cumulative_right[starts - 1, index]
        ).sum(axis=1, dtype=np.int64)
        return [
            (
                (
                    int(abs(left_count[i] - right_count[i])),
                    int(wrong[i]),
                    abs(float(offset)),
                ),
                float(offset),
            )
            for i, offset in enumerate(offsets)
        ]

    coarse = evaluate(np.linspace(-0.75, 0.75, 31))
    coarse_best = min(coarse, key=lambda row: row[0])[1]
    fine = evaluate(
        np.arange(
            max(-0.75, coarse_best - 0.04),
            min(0.75, coarse_best + 0.04) + 0.0005,
            0.001,
        )
    )
    fine_best = min(fine, key=lambda row: row[0])[1]
    exact = evaluate(
        np.arange(
            max(-0.75, fine_best - 0.001),
            min(0.75, fine_best + 0.001) + 0.00025,
            0.0005,
        )
    )
    return min(coarse + fine + exact, key=lambda row: row[0])[1]


def reflection(segmentation: Path) -> np.ndarray:
    anatomy = nib.as_closest_canonical(nib.load(segmentation))
    labels = np.asanyarray(anatomy.dataobj)
    left = np.isin(labels, LEFT)
    right = np.isin(labels, RIGHT)
    brain = labels > 0
    if not left.any() or not right.any():
        raise ValueError("SynthSeg lacks bilateral anatomy")
    coefficient = fit_plane(left, right)
    coefficient[2] += balance_intercept(coefficient, brain, left, right)
    voxel_plane = np.array([1.0, -coefficient[0], -coefficient[1], -coefficient[2]])
    world_plane = np.linalg.inv(anatomy.affine).T @ voxel_plane
    normal = world_plane[:3] / np.linalg.norm(world_plane[:3])
    distance = world_plane[3] / np.linalg.norm(world_plane[:3])
    matrix = np.eye(4)
    matrix[:3, :3] -= 2.0 * np.outer(normal, normal)
    matrix[:3, 3] = -2.0 * distance * normal
    return matrix


def write_transform(matrix_ras: np.ndarray, path: Path) -> None:
    ras_to_lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    matrix = ras_to_lps @ matrix_ras @ ras_to_lps
    parameters = [*matrix[:3, :3].ravel(), *matrix[:3, 3]]
    path.write_text(
        "#Insight Transform File V1.0\n# Transform 0\n"
        "Transform: AffineTransform_double_3_3\n"
        f"Parameters: {' '.join(str(value) for value in parameters)}\n"
        "FixedParameters: 0 0 0\n",
        encoding="utf-8",
    )


def apply(
    source: Path,
    reference: Path,
    output: Path,
    interpolation: str,
    transforms: list[Path],
    threads: int,
) -> None:
    command = [
        "antsApplyTransforms",
        "-d",
        "3",
        "-i",
        str(source),
        "-r",
        str(reference),
        "-o",
        str(output),
        "-n",
        interpolation,
    ]
    for transform in transforms:
        command.extend(("-t", str(transform)))
    run(command, threads)


def resample(
    source: Path, output: Path, spacing: tuple[float, ...], mask: bool, threads: int
) -> None:
    run(
        [
            "ResampleImageBySpacing",
            "3",
            str(source),
            str(output),
            *[f"{value:.8g}" for value in spacing],
            "0" if mask else "1",
            "0",
            "1" if mask else "0",
        ],
        threads,
    )


def calibrate(
    fixed: np.ndarray, moving: np.ndarray, fixed_mask: np.ndarray, support: np.ndarray
) -> np.ndarray:
    fit = fixed_mask & support & (fixed > 0) & (moving > 0)
    if fit.sum() < 100:
        raise ValueError("too few overlapping tissue voxels")
    moving_anchor = np.percentile(moving[fit], (1, 99))
    fixed_anchor = np.percentile(fixed[fit], (1, 99))
    slope = (fixed_anchor[1] - fixed_anchor[0]) / (moving_anchor[1] - moving_anchor[0])
    result = np.zeros(fixed.shape, dtype=np.float32)
    result[support] = (
        moving[support] * slope + fixed_anchor[0] - slope * moving_anchor[0]
    )
    np.clip(result, fixed.min(), fixed.max(), out=result)
    result[~support] = 0
    return result


def build(
    input_path: Path, output_path: Path, work: Path, synthseg: str, threads: int
) -> None:
    required = (
        synthseg,
        "antsRegistration",
        "antsApplyTransforms",
        "ResampleImageBySpacing",
    )
    missing = [command for command in required if shutil.which(command) is None]
    if missing:
        raise FileNotFoundError(f"missing commands: {', '.join(missing)}")

    work.mkdir(parents=True, exist_ok=False)
    synthseg_path = work / "synthseg.nii.gz"
    fixed_mask_path = work / "fixed_mask.nii.gz"
    reflected_path = work / "reflected_linear.nii.gz"
    moving_mask_path = work / "moving_mask.nii.gz"
    fixed_coarse = work / "fixed_1p5mm.nii.gz"
    moving_coarse = work / "moving_1p5mm.nii.gz"
    fixed_mask_coarse = work / "fixed_mask_1p5mm.nii.gz"
    moving_mask_coarse = work / "moving_mask_1p5mm.nii.gz"
    transform_path = work / "reflection_lps.tfm"
    prefix = work / "syn_"
    registered_raw = work / "registered_raw.nii.gz"
    registered_support = work / "registered_support.nii.gz"

    run(
        [
            synthseg,
            "--i",
            str(input_path),
            "--o",
            str(synthseg_path),
            "--robust",
            "--cpu",
            "--threads",
            str(threads),
        ],
        threads,
    )
    image = nib.load(input_path)
    fixed = image.get_fdata(dtype=np.float32)
    fixed_mask = brain_mask(fixed)
    matrix = reflection(synthseg_path)
    output_to_input = np.linalg.inv(image.affine) @ matrix @ image.affine
    reflected = ndimage.affine_transform(
        fixed,
        output_to_input[:3, :3],
        output_to_input[:3, 3],
        output_shape=image.shape,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.float32)
    moving_mask = brain_mask(reflected)
    save(image, fixed_mask, fixed_mask_path, np.uint8)
    save(image, reflected, reflected_path, np.float32)
    save(image, moving_mask, moving_mask_path, np.uint8)
    write_transform(matrix, transform_path)

    spacing = tuple(max(float(value), 1.5) for value in image.header.get_zooms()[:3])
    resample(input_path, fixed_coarse, spacing, False, threads)
    resample(reflected_path, moving_coarse, spacing, False, threads)
    resample(fixed_mask_path, fixed_mask_coarse, spacing, True, threads)
    resample(moving_mask_path, moving_mask_coarse, spacing, True, threads)
    run(
        [
            "antsRegistration",
            "--dimensionality",
            "3",
            "--float",
            "1",
            "--collapse-output-transforms",
            "1",
            "--output",
            str(prefix),
            "--interpolation",
            "Linear",
            "--use-histogram-matching",
            "0",
            "--winsorize-image-intensities",
            "[0.005,0.995]",
            "--transform",
            "SyN[0.1,2,0]",
            "--metric",
            f"CC[{fixed_coarse},{moving_coarse},1,3]",
            "--convergence",
            "[40x25x10,1e-6,5]",
            "--shrink-factors",
            "4x2x1",
            "--smoothing-sigmas",
            "2x1x0vox",
            "--masks",
            f"[{fixed_mask_coarse},{moving_mask_coarse}]",
            "--verbose",
            "0",
        ],
        threads,
    )
    warp = Path(f"{prefix}0Warp.nii.gz")
    if not warp.is_file():
        raise RuntimeError("SyN warp was not created")
    transforms = [warp, transform_path]
    apply(
        input_path,
        input_path,
        registered_raw,
        "LanczosWindowedSinc",
        transforms,
        threads,
    )
    apply(
        fixed_mask_path,
        input_path,
        registered_support,
        "NearestNeighbor",
        transforms,
        threads,
    )
    moving = nib.load(registered_raw).get_fdata(dtype=np.float32)
    support = nib.load(registered_support).get_fdata(dtype=np.float32) > 0
    save(image, calibrate(fixed, moving, fixed_mask, support), output_path, np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--synthseg", default="mri_synthseg")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.work is not None:
        build(
            args.input.resolve(),
            args.output.resolve(),
            args.work.resolve(),
            args.synthseg,
            args.threads,
        )
    else:
        temporary = Path(tempfile.mkdtemp(prefix="isles26-mirror-"))
        try:
            build(
                args.input.resolve(),
                args.output.resolve(),
                temporary / "work",
                args.synthseg,
                args.threads,
            )
        finally:
            shutil.rmtree(temporary)


if __name__ == "__main__":
    main()
