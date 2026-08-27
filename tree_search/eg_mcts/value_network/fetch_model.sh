#!/usr/bin/env bash
# Download the released EG-MCTS experience-guidance network checkpoint.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
destination="${1:-$script_dir/best_egn_for_emol.pt}"
url="https://raw.githubusercontent.com/jjljkjljk/EG-MCTS/829ee03fadf7966145aaa1f3bc366570c87a5135/eg_mcts/saved_EG_fn/best_egn_for_emol.pt"

if [[ -f "$destination" ]]; then
    echo "EGN checkpoint already present: $destination"
    exit 0
fi

mkdir -p "$(dirname "$destination")"
curl --fail --location --retry 3 "$url" --output "$destination"
echo "Downloaded EGN checkpoint: $destination"
