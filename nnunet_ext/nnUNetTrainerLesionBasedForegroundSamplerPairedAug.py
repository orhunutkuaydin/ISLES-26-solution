from collections import Counter
from math import isclose
from typing import Iterable, List, Tuple, Union

import importlib
import numpy as np
import torch
from scipy import ndimage
from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import (
    BasicTransform,
    ImageOnlyTransform,
)
from batchgeneratorsv2.transforms.intensity.brightness import (
    MultiplicativeBrightnessTransform,
)
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import (
    SimulateLowResolutionTransform,
)
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def connected_lesions(segmentation: np.ndarray, labels: Iterable[int]) -> dict:
    segmentation = np.asarray(segmentation)
    if segmentation.ndim not in (3, 4) or segmentation.shape[0] != 1:
        raise ValueError("expected one segmentation channel")
    foreground = np.isin(segmentation[0], tuple(labels))
    components, count = ndimage.label(
        foreground, structure=np.ones((3,) * foreground.ndim)
    )
    if count == 0:
        return {}
    spatial = np.argwhere(components > 0)
    component_ids = components[tuple(spatial.T)]
    order = np.argsort(component_ids, kind="stable")
    spatial = spatial[order].astype(np.int32, copy=False)
    sizes = np.bincount(component_ids[order], minlength=count + 1)[1:]
    offsets = np.concatenate(([0], np.cumsum(sizes, dtype=np.int64)))
    coordinates = np.concatenate(
        (np.zeros((len(spatial), 1), dtype=np.int32), spatial), axis=1
    )
    return {
        ("lesion_component", component): coordinates[
            offsets[component - 1] : offsets[component]
        ]
        for component in range(1, count + 1)
    }


class LesionDataset:
    def __init__(self, dataset, label_manager):
        self.dataset = dataset
        self.identifiers = list(dataset.identifiers)
        self.annotated_key = tuple([-1] + list(label_manager.all_labels))
        self.has_ignore = label_manager.has_ignore_label
        self.locations = {}
        for identifier in self.identifiers:
            _, segmentation, _, _ = dataset.load_case(identifier)
            self.locations[identifier] = connected_lesions(
                segmentation, label_manager.foreground_labels
            )

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def load_case(self, identifier):
        data, segmentation, previous, properties = self.dataset.load_case(identifier)
        properties = dict(properties)
        locations = {}
        if self.has_ignore:
            locations[self.annotated_key] = properties["class_locations"][
                self.annotated_key
            ]
        locations.update(self.locations[identifier])
        properties["class_locations"] = locations
        return data, segmentation, previous, properties


class LesionLoader(nnUNetDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lesion_probabilities = None
        if isinstance(self._data, LesionDataset):
            counts = np.asarray(
                [len(self._data.locations[key]) for key in self.indices],
                dtype=np.float64,
            )
            if counts.sum() == 0:
                raise RuntimeError("training split contains no lesions")
            self.lesion_probabilities = counts / counts.sum()
            self.stock_oversample = self.get_do_oversample
            self.current_oversample = None
            self.get_do_oversample = self._oversample

    def _oversample(self, index):
        if self.current_oversample is None:
            return bool(self.stock_oversample(index))
        return bool(self.current_oversample[index])

    def get_indices(self):
        if self.lesion_probabilities is None:
            return super().get_indices()
        self.current_oversample = tuple(
            bool(self.stock_oversample(index)) for index in range(self.batch_size)
        )
        return [
            self.indices[
                int(
                    np.random.choice(
                        len(self.indices),
                        p=self.lesion_probabilities
                        if forced
                        else self.sampling_probabilities,
                    )
                )
            ]
            for forced in self.current_oversample
        ]


class PairedNoise(ImageOnlyTransform):
    p_per_channel = 1.0
    synchronize_channels = True

    def __init__(self, variance: RandomScalar):
        super().__init__()
        self.variance = variance

    def get_parameters(self, **data):
        return {"sigma": float(sample_scalar(self.variance, image=data["image"]))}

    def _apply_to_image(self, image: torch.Tensor, **parameters):
        noise = torch.empty(
            (1, *image.shape[1:]), device=image.device, dtype=image.dtype
        )
        image += noise.normal_(0.0, parameters["sigma"])
        return image


PAIRED_TYPES = (
    GaussianNoiseTransform,
    GaussianBlurTransform,
    MultiplicativeBrightnessTransform,
    ContrastTransform,
    SimulateLowResolutionTransform,
    GammaTransform,
)
EXPECTED = Counter(
    {
        "GaussianNoiseTransform": 1,
        "GaussianBlurTransform": 1,
        "MultiplicativeBrightnessTransform": 1,
        "ContrastTransform": 1,
        "SimulateLowResolutionTransform": 1,
        "GammaTransform": 2,
    }
)
PROBABILITIES = {
    "GaussianNoiseTransform": (0.1,),
    "GaussianBlurTransform": (0.1,),
    "MultiplicativeBrightnessTransform": (0.15,),
    "ContrastTransform": (0.15,),
    "SimulateLowResolutionTransform": (0.125,),
    "GammaTransform": (0.1, 0.3),
}


def pair_intensity(pipeline: BasicTransform) -> BasicTransform:
    if not isinstance(pipeline, ComposeTransforms):
        raise TypeError("expected stock nnU-Net transforms")
    counts = Counter()
    probabilities = {}
    for entry in pipeline.transforms:
        if not isinstance(entry, RandomTransform) or not isinstance(
            entry.transform, PAIRED_TYPES
        ):
            continue
        transform = entry.transform
        name = type(transform).__name__
        counts[name] += 1
        entry.apply_probability *= float(transform.p_per_channel)
        if isinstance(transform, GaussianNoiseTransform):
            entry.transform = PairedNoise(transform.noise_variance)
        else:
            transform.p_per_channel = 1.0
            transform.synchronize_channels = True
        probabilities.setdefault(name, []).append(float(entry.apply_probability))
    actual = {name: tuple(sorted(values)) for name, values in probabilities.items()}
    if counts != EXPECTED or any(
        name not in actual
        or len(actual[name]) != len(expected)
        or any(
            not isclose(a, b, rel_tol=0, abs_tol=1e-12)
            for a, b in zip(actual[name], expected)
        )
        for name, expected in PROBABILITIES.items()
    ):
        raise RuntimeError("nnU-Net augmentation contract changed")
    return pipeline


class nnUNetTrainerLesionBasedForegroundSamplerPairedAug(nnUNetTrainer):
    def get_tr_and_val_datasets(self):
        training, validation = super().get_tr_and_val_datasets()
        return LesionDataset(training, self.label_manager), validation

    def get_dataloaders(self):
        module = importlib.import_module(
            "nnunetv2.training.nnUNetTrainer.nnUNetTrainer"
        )
        original = module.nnUNetDataLoader
        module.nnUNetDataLoader = LesionLoader
        try:
            return super().get_dataloaders()
        finally:
            module.nnUNetDataLoader = original

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        if use_mask_for_norm is None or len(use_mask_for_norm) != 2:
            raise RuntimeError("paired augmentation requires two channels")
        stock = nnUNetTrainer.get_training_transforms(
            patch_size=patch_size,
            rotation_for_DA=rotation_for_DA,
            deep_supervision_scales=deep_supervision_scales,
            mirror_axes=mirror_axes,
            do_dummy_2d_data_aug=do_dummy_2d_data_aug,
            use_mask_for_norm=use_mask_for_norm,
            is_cascaded=is_cascaded,
            foreground_labels=foreground_labels,
            regions=regions,
            ignore_label=ignore_label,
        )
        return pair_intensity(stock)
