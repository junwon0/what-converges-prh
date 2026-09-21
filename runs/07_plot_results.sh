#!/usr/bin/env bash
set -euo pipefail
python -m scripts.plots.experiments --sections prh_alignment v2t_alignment --assets-dir assets --force
