# Design note: `graft_subspace` (stub, not implemented)

**Goal:** copy a functional component (a head, a set of neurons, or a subspace of a weight matrix) from a donor
model into the working copy. This is "capability grafting" from the backlog.

**Proposed op:**
```json
{"op": "graft_subspace", "tensor": "...", "donor_source": "hf:...|fixture:...", "donor_tensor": "...",
 "basis": {"rows": "a:b"} | {"head": h} | {"directions": [dir specs]}, "align": "none|procrustes|activation"}
```

**Hard parts to solve first:**
1. **Basis alignment.** Two independently trained models have arbitrary rotations and permutations of hidden
   spaces. Grafting raw weights is meaningless unless the bases are aligned: (a) Procrustes on shared-token
   embeddings for residual spaces, (b) activation matching on a shared corpus for hidden units (Hungarian
   assignment), (c) feature-level alignment through the M6 index.
2. **Read and write sides.** A head reads (q/k/v) from and writes (o) to the residual. Both sides need the
   alignment, and the downstream readers of what it writes may not exist in the recipient.
3. **Scale / normalisation** differences (RMSNorm gains), handled by matching activation statistics.

**Safety:** a graft can move far more behaviour than its parameter count suggests. Grafted patches should always
be flagged for review and run the full regression suite plus an M7 audit against the pre-graft model.
