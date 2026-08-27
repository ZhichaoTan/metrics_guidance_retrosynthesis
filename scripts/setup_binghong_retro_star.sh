#!/usr/bin/env bash
# Clone the pinned Retro* source, apply the release patch, and download its assets.
#
# Usage: bash scripts/setup_binghong_retro_star.sh

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
binghong_root="${BINGHONG_RETRO_STAR_ROOT:-$project_root/external/binghong_retro_star}"
upstream_commit="60e7a24213810706a7075d113704c9320158d726"
patch_path="$script_dir/patches/binghong_retro_star.patch"
data_url="https://www.dropbox.com/s/ar9cupb18hv96gj/retro_data.zip?dl=1"
conda_env="${CONDA_ENV_NAME:-metrics_guidance_retrosynthesis}"

if [[ ! -d "$binghong_root/.git" ]]; then
    mkdir -p "$(dirname "$binghong_root")"
    git clone --no-checkout https://github.com/binghong-ml/retro_star.git "$binghong_root"
    git -C "$binghong_root" checkout --detach "$upstream_commit"
    git -C "$binghong_root" apply "$patch_path"
fi

value_model="$binghong_root/retro_star/saved_models/best_epoch_final_4.pt"
if [[ ! -f "$value_model" ]]; then
    archive="$(mktemp "${TMPDIR:-/tmp}/retro-data.XXXXXX.zip")"
    curl --fail --location --retry 3 "$data_url" --output "$archive"
    unzip -q -o "$archive" -d "$binghong_root/retro_star"
    rm -f "$archive"
fi

if [[ "${SKIP_PIP_INSTALL:-0}" != "1" ]]; then
    conda run -n "$conda_env" pip install -e "$binghong_root/retro_star/packages/mlp_retrosyn"
    conda run -n "$conda_env" pip install -e "$binghong_root/retro_star/packages/rdchiral"
fi

echo "Retro* checkout ready: $binghong_root"
