#!/usr/bin/env bash
set -euo pipefail
DEVICE="${DEVICE:-cuda}"
python -m scripts.experiments.cli --device "$DEVICE" --sections prh_alignment \
  --prh-metrics mutual_knn --prh-max-samples 1024 --prh-num-permutations 500
python -m scripts.experiments.cli --device "$DEVICE" --sections prh_alignment \
  --prh-metrics mst_overlap --topology-dist-metric euclidean \
  --topology-max-samples 1024 --topology-num-permutations 500
