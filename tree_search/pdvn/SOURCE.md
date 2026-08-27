# PDVN-SPRS + Retro*-0 provenance

This baseline is the public SPRS reproduction of the PDVN rollout policy, not
an original PDVN checkpoint.

- PDVN source: [DiXue98/PDVN](https://github.com/DiXue98/PDVN), commit
  `5cdb5ff4ace9908850a7ccc668734596ea4c42d3`. That repository has no license
  file, so no PDVN source code is copied here.
- SPRS source: [AlanHassen/SPRS](https://github.com/AlanHassen/SPRS), commit
  `11d08c363c149114000d189e7138ef7e98512ff6`, MIT license (see
  `SPRS_LICENSE`).
- SPRS model release: <https://doi.org/10.6084/m9.figshare.29409725.v1>;
  `fetch_model.sh` extracts the released rollout checkpoint from it.

The policy first takes a global template top-k from the Retro* reference model,
gathers the learned SPRS logits at those same indices, and applies a softmax
within that candidate set. RDChiral outcomes are canonicalized, merged, and
normalized, and identity outcomes are removed. Search uses the Retro*
`MolTree` with a zero value function; this is therefore reported as
**PDVN-SPRS policy + Retro*-0**, not full dual-value PDVN. No metric-guidance
module is imported by this baseline.
