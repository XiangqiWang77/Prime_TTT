# Release Bundle

A usable PrimeTTT adapter release must include:

- `H_psi` checkpoint: hypernetwork weights.
- `B_base`: oracle-mean LoRA B tensors.
- Frozen `A` tensors or the LoRA slot seed/config used to regenerate them.
- The slot manifest: layer/module name, input dimension, output dimension, rank.

Oracle checkpoints are training artifacts and are not required for inference.

