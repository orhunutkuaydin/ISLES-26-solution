# ISLES 2026: SynthSeg-guided mirror-channel nnU-Net

Code and method summary for the final ISLES 2026 participation pipeline. The method combines a native-T1 nnU-Net with a second nnU-Net that receives the T1 and a registered contralateral mirror.

## Method

Training used 1,444 of 1,453 native T1-weighted MR volumes from the [ISLES'26 cohort](https://isles-26.grand-challenge.org/dataset/): Nine scans were excluded due to image label mismatches. Two nnU-Net v2 datasets were used. Dataset901 used native T1; Dataset902 added an anatomically reflected T1 as a second channel.

For each scan, SynthSeg robust 2.0 produced bilateral anatomical labels. We define the two hemisphere sides using the Fourteen left-right label pairs. Along each left-to-right voxel line intersecting both masks, the midpoint between them was collected. A plane was robustly fitted to these points, and its intercept was refined to balance brain tissue on both sides. The plane was converted from canonical voxel coordinates to native RAS coordinates and used to mirror the T1 channel.

The reflected image was deformably aligned to the original. Registration used Advanced Normalization Tools (ANTs) SyN at 1.5-mm resolution, cross-correlation with radius three, transform SyN[0.1,2,0], iterations 40x25x10, shrink factors 4x2x1, and smoothing sigmas 2x1x0 voxels. Reflection and deformation were composed, and the original T1 was resampled once onto its native grid with Lanczos-windowed sinc interpolation. Intensities were linearly matched using the first and ninety-ninth percentiles of positive overlapping tissue, clipped to the original range, and set to zero outside registered support.

Both datasets used nnU-Net 3d_fullres ResEncM plans, default preprocessing, loss, optimizer, schedule, spatial augmentation, and five-fold training. Dataset901 used the default trainer. Dataset902 retained the same network and schedule while using lesion-balanced sampling for foreground crops and synchronizing intensity augmentation across both channels.

At inference, fold probabilities were averaged within each model. The final probability map was their voxelwise maximum. Binary masks were united. Dataset901-only components below one millilitre were removed. Remaining components at or below 0.01 millilitres were removed unless they contained a voxel predicted by both models.

## Requirements

- Linux with Python 3.10+
- FreeSurfer 7.4.1 SynthSeg (`mri_synthseg` on `PATH`)
- ANTs 2.5.4 (`antsRegistration`, `antsApplyTransforms`, and `ResampleImageBySpacing` on `PATH`)
- CUDA-capable PyTorch for nnU-Net training

Install the Python packages after installing the PyTorch build appropriate for the machine:

```bash
python -m pip install -r requirements.txt
```

Set the standard nnU-Net paths:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results

export nnUNet_extTrainer=./nnunet_ext
```

`nnUNet_extTrainer` lets nnU-Net discover the included Dataset902 trainer without modifying the nnU-Net installation.

## Create channel 2 for one patient

Training:

```bash
python mirror_train.py patient_T1.nii.gz patient_mirror.nii.gz --threads 8
```

Inference:

```bash
python mirror_inference.py patient_T1.nii.gz patient_mirror.nii.gz --threads 4
```

Use `mirror_train.py` to generate Dataset902 training channel 2 with robust SynthSeg 2.0, a 1.5 mm registration-spacing floor, and SyN convergence `[40x25x10,1e-6,5]`.
At inference, `mirror_inference.py` uses SynthSeg `--fast --crop 192 256 192` on four CPU threads, a 2 mm spacing floor, SyN convergence `[20x10x5,1e-6,5]`, and support-cropped one-pass Lanczos reconstruction.
These inference settings were selected to reduce runtime during challenge submission, whereas the training recipe preserves the higher-resolution channel generation used to construct the training set.

For nnU-Net, the folder structure for one patient was:

```text
nnUNet_raw/Dataset901_*/imagesTr/patient_0000.nii.gz  native T1
nnUNet_raw/Dataset901_*/labelsTr/patient.nii.gz       binary lesion mask
nnUNet_raw/Dataset902_*/imagesTr/patient_0000.nii.gz  native T1
nnUNet_raw/Dataset902_*/imagesTr/patient_0001.nii.gz  generated mirror
nnUNet_raw/Dataset902_*/labelsTr/patient.nii.gz       binary lesion mask
```

`Dataset901` names channel `0` as `T1`. `Dataset902` names channel `0` as `T1` and channel `1` as `T1_reflected`.

## Train

Train the stock one-channel model and the exact two-channel variant:

```bash
CUDA_VISIBLE_DEVICES=0 ./train.sh 901
CUDA_VISIBLE_DEVICES=0 ./train.sh 902
```

Each command plans the nnU-Net ResEncM `3d_fullres` configuration and trains folds 0-4. The [custom trainer](nnunet_ext/nnUNetTrainerLesionBasedForegroundSamplerPairedAug.py) changes only connected-lesion foreground sampling and synchronization of stock intensity augmentation; network architecture, preprocessing, loss, optimizer, learning-rate schedule, spatial augmentation, and training length remain nnU-Net defaults.

## Inference

The training commands create the model directories expected by inference:

```text
$nnUNet_results/Dataset901_*/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres
$nnUNet_results/Dataset902_*/nnUNetTrainerLesionBasedForegroundSamplerPairedAug__nnUNetResEncUNetMPlans__3d_fullres
```

For inference run:

```bash

CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict \
  -i inference901 -o predictions901 -d 901 -c 3d_fullres \
  -tr nnUNetTrainer -p nnUNetResEncUNetMPlans -f 0 1 2 3 4 \
  -chk checkpoint_final.pth --save_probabilities

CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict \
  -i inference902 -o predictions902 -d 902 -c 3d_fullres \
  -tr nnUNetTrainerLesionBasedForegroundSamplerPairedAug \
  -p nnUNetResEncUNetMPlans -f 0 1 2 3 4 \
  -chk checkpoint_final.pth --save_probabilities
```

nnU-Net averages the selected folds. Fuse and postprocess one case with:

```bash
python postprocess.py predictions901/patient.npz predictions902/patient.npz probability_final.nii.gz segmentation_final.nii.gz
```

## Models and weights
Model weights are not currently included. If this submission places on the leaderboard, its models, weights, and results will be published.

## References

1. Raina, K., Yahorau, U., & Schmah, T. (2020). [Exploiting Bilateral Symmetry in Brain Lesion Segmentation with Reflective Registration](https://doi.org/10.5220/0008912101160122). In *Proceedings of the 13th International Joint Conference on Biomedical Engineering Systems and Technologies (BIOSTEC 2020), Volume 2: BIOIMAGING*, 116–122.
2. Bao, Q., Mi, S., Gang, B., Yang, W., Chen, J., & Liao, Q. (2022). [MDAN: Mirror Difference Aware Network for Brain Stroke Lesion Segmentation](https://doi.org/10.1109/JBHI.2021.3113460). *IEEE Journal of Biomedical and Health Informatics, 26*(4), 1628–1639.
3. Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H. (2021). [nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation](https://doi.org/10.1038/s41592-020-01008-z). *Nature Methods, 18*, 203–211.
4. Liew, S.-L., et al. (2022). [A large, curated, open-source stroke neuroimaging dataset to improve lesion segmentation algorithms](https://doi.org/10.1038/s41597-022-01401-7). *Scientific Data, 9*, 320. This is the ATLAS R2.0 data descriptor referenced by the official [ATLAS R3.0 release page](https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html), which lists its R3.0 descriptor as forthcoming.
5. Absher, J., Goncher, S., Newman-Norlund, R., et al. (2024). [The Stroke Outcome Optimization Project: Acute ischemic strokes from a comprehensive stroke center](https://doi.org/10.1038/s41597-024-03667-5). *Scientific Data, 11*, 839.
6. de la Rosa, E., Su, R., Reyes, M., et al. (2024). [ISLES'24: Final Infarct Prediction with Multimodal Imaging and Clinical Data. Where Do We Stand?](https://arxiv.org/abs/2408.10966). *arXiv:2408.10966*.
