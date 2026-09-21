#!/usr/bin/env bash
set -euo pipefail
DEVICE="${DEVICE:-cuda}"
LOCAL="mutual_knn_sim_tau1,mutual_knn_sim_tau0p1,mutual_knn_sim_tau0p01,mutual_knn_sim_tau0p001,mutual_knn_sim_tau0p0001"
GLOBAL="mst_overlap_sim_tau1,mst_overlap_sim_tau0p1,mst_overlap_sim_tau0p01,mst_overlap_sim_tau0p001,mst_overlap_sim_tau0p0001"
python -m scripts.experiments.cli --device "$DEVICE" --sections prh_alignment \
  --prh-metrics "$LOCAL" --prh-max-samples 1024 --prh-num-permutations 500
python -m scripts.experiments.cli --device "$DEVICE" --sections prh_alignment \
  --prh-metrics "$GLOBAL" --topology-dist-metric euclidean \
  --topology-max-samples 1024 --topology-num-permutations 500
