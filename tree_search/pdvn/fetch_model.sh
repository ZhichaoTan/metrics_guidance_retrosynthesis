#!/usr/bin/env bash
# Extract the released SPRS rollout checkpoint used by the PDVN baseline.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
destination="${1:-$script_dir/assets/rollout_model.ckpt}"
archive="${SPRS_ARCHIVE:-${TMPDIR:-/tmp}/sprs_data.zip}"
url="https://ndownloader.figshare.com/files/55654517"
member="data/models/dual_value_networks/self_play/final_results/model/742788_mols/one_step/rollout_model.ckpt"

if [[ -f "$destination" ]]; then
    echo "PDVN-SPRS checkpoint already present: $destination"
    exit 0
fi

mkdir -p "$(dirname "$destination")"
curl --fail --location --retry 3 --continue-at - "$url" --output "$archive"
unzip -p "$archive" "$member" > "$destination"
echo "Installed PDVN-SPRS checkpoint: $destination"
