#!/usr/bin/env bash
set -euo pipefail

dataset="${1:?usage: train.sh 901|902}"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export nnUNet_extTrainer="$root/nnunet_ext"

case "$dataset" in
  901) trainer=nnUNetTrainer ;;
  902) trainer=nnUNetTrainerLesionBasedForegroundSamplerPairedAug ;;
  *) exit 2 ;;
esac

nnUNetv2_plan_and_preprocess -d "$dataset" -pl nnUNetPlannerResEncM --verify_dataset_integrity
for fold in 0 1 2 3 4; do
  nnUNetv2_train "$dataset" 3d_fullres "$fold" -tr "$trainer" -p nnUNetResEncUNetMPlans
done
