---
layout: article
title: "[CuTeDSL] Atom Shapes: MMA, TMA, and TMEM"
date: 2026-03-31
---

When using partition functions in CuTe DSL, the returned tensor shape contains an **atom shape** representing hardware instruction constraints. This article covers the three atom types from your examples.

## 1. MMA Atom Shape

```python
tCgA: Unknown = thr_mma.partition_A(gA)
# Shape: ((128, 16), 1, 4, ...)
#        └─mma_atom─┘
```

- **(128, 16)**: MMA atom shape
  - **K=16**: Always 16 for dense FP16 (from [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-matrix-shape))
  - **M**: Architecture-dependent (Hopper M=64, Blackwell M=128, can be 256 with 2 CTAs)
- **1, 4, ...**: Rest dimensions for bM and bK. '...' are remaining dimensions

## 2. TMA Atom Shape

```python
tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
    (((128, 16), 1, 4), 64, 64),  # input
    ...
)
# Input:  (((128, 16), 1, 4), 64, 64)
# Output: ((8192, 1), 64, 64)
#         └─tma_atom─┘
```

- **(8192, 1)**: TMA atom shape for smem
  - **8192**: Total elements in 1 TMA instruction (e.g., 128×64)
  - **1**: Number of TMA instructions issued
- **When num_instrs > 1?**: Per [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#using-the-tensor-memory-accelerator-tma), "the shared memory box's inner dimension must be less or equal to the span of the swizzle pattern"

## 3. TMEM Atom Shape

```python
tTRtC: Unknown = tmem_thr_copy.partition_S(tEPItAcc)
# Shape: (((64, 32), 1), 1, 1, 1, 4, 1, 1)
#        └─ tmem_atom ─┘  epi_rests, and others (mma_rests, tma_rests, all_rests)
```

- **((64, 32), 1)**: TMEM atom shape
  - **64**: Repetition
  - **32**: Number of lanes
  - **1**: Number of instructions (NOTE: only observed 1, unclear when >1)
- The remaining `1, 1` is the epi_rests when dividing the epi tile by this atom shape
- Other dimensions are inherited from the input tensor.

## Comparison Table

| Aspect | MMA | TMA | TMEM |
|--------|-----|-----|------|
| Atom Shape | `(mma_atom_m, mma_atom_k)` | `(tma_atom, num_atoms)` | `((tmem_atom_n, tmem_atom_m), num_atoms)` |
| Num Atom Repeats | Implicit (1) | Explicit | Explicit |
| Num Atom Tiles | Outside atom | Implicit (1) | Outside atom (like MMA) |
| Inner Structure | `(M/N, K)` | `(elements, num)` | `((repetition, lanes), num)` |

## Key Observations

1. **MMA and TMEM**: Remaining num of atom tiles are *outside* the atom shape
2. **TMA**: The flatten shape is not changed before and after tma_partition
3. **Inner format similarity**: TMEM combines MMA's outer behavior with TMA's inner `(atom, num_instrs)` structure
