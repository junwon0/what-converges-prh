#!/usr/bin/env bash
set -euo pipefail
DEVICE="${DEVICE:-cuda}"
LOCAL_DIST="mutual_knn_dist_tau1,mutual_knn_dist_tau0p1,mutual_knn_dist_tau0p01,mutual_knn_dist_tau0p001,mutual_knn_dist_tau0p0001"
GLOBAL_DIST="mst_overlap_dist_tau1,mst_overlap_dist_tau0p1,mst_overlap_dist_tau0p01,mst_overlap_dist_tau0p001,mst_overlap_dist_tau0p0001"
python -m scripts.experiments.cli --device "$DEVICE" --sections v2t_alignment --prh-metrics mutual_knn,mst_overlap
python -m scripts.experiments.cli --device "$DEVICE" --sections v2t_alignment --prh-metrics "$LOCAL_DIST"
python -m scripts.experiments.cli --device "$DEVICE" --sections v2t_alignment --prh-metrics "$GLOBAL_DIST" --topology-dist-metric euclidean
