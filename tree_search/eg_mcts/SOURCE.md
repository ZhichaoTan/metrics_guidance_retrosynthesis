# EG-MCTS provenance

The search rules and Experience Guidance Network in this directory are adapted
from [EG-MCTS](https://github.com/jjljkjljk/EG-MCTS) at commit
`829ee03fadf7966145aaa1f3bc366570c87a5135`, under its MIT license.

The released EGN checkpoint is not committed. Fetch it with:

```bash
bash tree_search/eg_mcts/value_network/fetch_model.sh
```

The script downloads `saved_EG_fn/best_egn_for_emol.pt` from that commit.

The EG-MCTS entry point uses the Retro* MLP one-step policy and Retro* stock
prepared by `scripts/setup_binghong_retro_star.sh`, so it runs under the same
protocol as the other planners. No complexity-distance guidance code is
imported by this baseline.

If RDKit cannot parse a template that the one-step model applied, the reaction
receives the neutral EGN value `0.5` instead of failing the whole target. Each
occurrence is recorded as `unscored_templates` in the target record and
aggregated in `run_summary.json`.
