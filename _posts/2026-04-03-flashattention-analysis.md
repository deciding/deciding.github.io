# Flash Attention Kernel Execution Flow Analysis (SM100)

## Overview

This document analyzes the execution flow of the Flash Attention kernel for SM100 (Blackwell architecture) implemented in `flash_fwd_sm100_simple.py`. The kernel uses warp specialization with 16 warps per CTA (thread block) to efficiently compute attention.

---

Key Questions:
1. Why 2 q stages?
2. Why softmax can reuse a row on a single thread compared to hopper?
3. Is ex2 emulation really the key source of performance gain in softmax?
4. Why split P?
5. Why use custom pipeline implementations?
6. Why use custom gemm implementation?
7. Why tmem is organized like this?


## 1. Kernel Configuration

### Supported Features
- BF16 & FP16 dtype
- Non-causal & causal attention
- MHA, GQA, MQA
- Head dimensions: 64, 96, 128, (192, 128)
- Variable length sequences
- Sliding window attention
- Split-KV

Note that we only focus on the simplest MHA for tutorial purpose

### Key Parameters
| Parameter | Default Value | Description |
|-----------|--------------|-------------|
| m_block_size | 128 | Q block size |
| n_block_size | 128 | K/V block size |
| q_stage | 2 | Number of innermost loop Q stages (double-buffering) |
| kv_stage | 6 | Number of inner loop KV stages (double-buffering) |
| is_persistent | True | Persistent scheduling |
| use_2cta_instrs | True | Use 2-CTA MMA instructions |

---

**Optimization**
- Why use 2 q_stages? because 2 q don't have softmax stats dependency, so O rescale can be isolated from critical path

## 2. Warp Specialization

The kernel uses **16 warps per CTA**, each assigned to a specific role:

| Warp(s) | Role | Function |
|---------|------|----------|
| 14 | Load | TMA load of Q, K, V from GMEM to SMEM |
| 12 | MMA | GEMM operations (Q×K and P×V) |
| 0-3 | Softmax0 | Softmax computation for stage 0 |
| 4-7 | Softmax1 | Softmax computation for stage 1 |
| 8-11 | Correction | Rescale accumulated O, write to sO |
| 13 | Epilogue | Copy O from SMEM to GMEM |
| 15 | Empty | Padding warp (no work) |

---

## 3. Memory Hierarchy

### Shared Memory Layout Details

The shared memory layouts are computed using `make_smem_layout_a` (for Q/P/O) and `make_smem_layout_b` (for K/V). The inner shape represents how data is partitioned for MMA operations:

```
sQ_layout: ((128,16), 1, (4,2), 2)
sK_layout: ((64,16), 1, (4,2), 6)  
sV_layout: ((64,16), 1, 8, 6)
sO_layout: ((8,16), (64,2), (1,2))
```

**Inner Shape Interpretation:**

| Tensor | Shape | Meaning |
|--------|-------|---------|
| **sQ** | `((128,16), 1, (4,2), 2)` | (atom_m=128, atom_k=16) × num_tiles_m=1 × (num_tiles_k=4×2) × q_stage=2 |
| **sK** | `((64,16), 1, (4,2), 6)` | (atom_n=64, atom_k=16) × num_tiles_n=1 × (num_tiles_k=4×2) × kv_stage=6 |
| **sV** | `((64,16), 1, 8, 6)` | (atom_n=64, atom_d=16) × num_tiles_n=1 × num_tiles_d=8 × kv_stage=6 |
| **sO** | `((8,16), (64,2), (1,2))` | (atom_m=8, count_m=16) × (atom_n=64, count_n=2) × (1, q_stage=2) |

Note: For sO, the smem atom is `K_SW128` (K-major with 128-bit swizzle), which has shape (8, 64):
- atom_m=8, atom_n=64 (64×2bytes=128bytes swizzle)
- tile_to_shape: (128,128,2) / (8,64) = (16, 2, 2)

### Why sQ (128) vs sK (64)?

With **cta_group_size = 2** (2 CTA MMA instruction), the MMA tile shape is (256, 128, 16). Both M and N dimensions are divided by cta_group_size for per-CTA shared memory allocation:

So:
- sQ gets 128 = m_block_size / 2
- sK gets 64 = n_block_size / 2

**sK vs sV**: 
- sK has inner stride `(64,1)` → K-major layout (seq_k is contiguous)
- sV has inner stride `(1,64)` → V is transposed (head_dim is contiguous)
- This confirms V is stored as `(head_dim, seq_k)` in smem for P×V MMA efficiency

```
GLOBAL MEMORY
    │
    ├── Q tensor: (seq_q, head_dim) - Query
    ├── K tensor: (seq_k, head_dim) - Key
    ├── V tensor: (head_dim, seq_k) - Value (transposed)
    ├── O tensor: (seq_q, head_dim_v) - Output
    └── LSE tensor: (seq_q, num_heads) - Log-sum-exp (optional)
    │
    │ TMA Load / TMA Store
    ▼
SHARED MEMORY
    │
    ├── sQ: Q stages × m_block × head_dim
    ├── sK: KV stages × n_block × head_dim
    ├── sV: KV stages × n_block × head_dim_v
    ├── sO: Q stages × m_block × head_dim_v
    └── sScale: row_max and row_sum storage
    │
    │ TMEM Load / Store
    ▼
TENSOR MEMORY (TMEM)
    │
    ├── tS: Attention scores S = Q × K^T (128 × 256 elements)
    ├── tP: Softmax probabilities P = exp2(S - row_max)
    └── tO: Accumulated output O (m_block × head_dim_v)
    │
    ▼
REGISTERS
    │
    ├── tSrS: Score fragment (tmem → registers)
    ├── tSrP: Probability fragment (registers → tmem)
    └── tSrO: Output fragment (tmem → registers)
```

---

## 4. Execution Flow

### 4.1 Load Order Pattern

Before understanding the phase details, it's critical to understand the K/V load order:

```
Let n = n_block_max - n_block_min (number of K/V blocks to process)

FIRST ITERATION (Initial Loads):
─────────────────────────────────
1. Load K[n-1]    ← Last K block first
2. Load Q[0]      ← Q stage 0
3. Load Q[1]      ← Q stage 1 (double buffering)
4. Load V[n-1]    ← Last V block

SUBSEQUENT ITERATIONS (Pipelined):
──────────────────────────────────
Then periodic: Load pairs from n-1 down to 0
  K[n-1], V[n-1] → K[n-2], V[n-2] → ... → K[0], V[0]

VISUAL:
────────
Time │ Loads
─────┼─────────────────────────────────────────────────────
  0  │ K[n-1]    │ Q[0]    │ Q[1]    │ V[n-1]
  1  │ K[n-2]    │         │         │ V[n-2]
  2  │ K[n-3]    │         │         │ V[n-3]
  3  │ ...       │         │         │ ...
  4  │ K[1]      │         │         │ V[1]
  5  │ K[0]      │         │         │ V[0]
```


```
Pseudocode:
─────────────

# Prologue
# First iteration: Load K[n-1], Q[0], Q[1], V[n-1]
K[n-1] → sK[buffer_idx]
Q[0] → sQ[stage_0]
Q[1] → sQ[stage_1]  # Prefetch for next Q iteration
V[n-1] → sV[buffer_idx]

# Pipeline loop
for k_block in range(n_block_min, n_block_max):

    # Issue TMA loads
    if k_block < n_block_max - 1:
        K[k_block - 1] → sK[kv_buffer]  # Load next K in reverse
        V[k_block - 1] → sV[kv_buffer]  # Load next V in reverse
    
```

**Pipeline Management:**
- `pipeline_q`: Coordinates Q loading between producer (load warp) and consumer (MMA warp)
- `pipeline_kv`: Coordinates K/V loading between producer and consumer

### 4.3 MMA Warp Operations (Warp 12)

S_{0, n-1}
S_{1, n-1}

loop(i, 0, n-1):
    O_{0, n-1-i}
    O_{1, n-1-i}
    S_{0, n-2-i}
    S_{1, n-2-i}

O_{0, 0}
O_{1, 0}

- QK uses precomputed desc, to avoid in-loop desc computation
- PV uses partial gemm, to allow split-P-arrive optimization

The MMA warp performs two main matrix multiplications:

#### GEMM_QK: Q × K^T → S
```
Prologue
1. Wait for Q[stage] to be ready (pipeline_q)
2. Wait for K[n-1] to be ready (pipeline_kv)
3. Execute GEMM: S[stage] = Q[stage] × K[n-1]^T
4. Signal S[stage] is ready for softmax (pipeline_s_p_o, s producer)
5. Release K[n-1] (advance pipeline_kv consumer)

For each K/V block iteration:
1. Wait for V[i] to be ready (pipeline_kv)
2. Wait for P[i] to be ready (3/4 pipeline_s_p_o, o producer + p producer (actually mma is consumer))
3. Execute GEMM: O[stage] += P[stage][i] × V[i] (1/4 Wait for pipeline_p_lastsplit)
4. Wait for K[i-1] to be ready (pipeline_kv)
5. Execute GEMM: S[stage] = Q[stage] × K[i-1]^T
6. Signal S[stage] is ready for softmax (pipeline_s_p_o, s producer)
7. Release K[i-1] (advance pipeline_kv consumer)
8. Release V[i] (advance pipeline_kv consumer)

Epilogue
1. Release Q[0]Q[1]
2. Wait for V[0] to be ready (pipeline_kv)
3. Wait for P[0] to be ready (pipeline_s_p_o, o producer + p producer (acutally mma is consumer))
4. Execute GEMM: O[stage] += P[stage][0] × V[0] (Wait for pipeline_p_lastsplit)
5. Signal O[stage] is ready for O final rescale (pipeline_o_acc)
6. Release V[0] (advance pipeline_kv consumer)
```

**Key Optimization:** The `split_P_arrive` feature allows partial P to be sent for GEMM_PV while softmax is still computing the rest, overlapping computation.

### 4.4 Softmax Operations (Warps 0-7)

The softmax warps perform numerically stable softmax computation:

sScale:
- used for saving row_sum, row_max
- size: 2 x m_block_size x q_stages, tidx is index of m_block_size

sm_stats_barrier:
- size: q_stages=2 x warps=4

```
Prologue
1. wait for pipeline_sm_stats as producer
2. compute softmax[n-1](is_first)

For each n block: softmax[i]
1. wait for S[stage][i] from mma (pipeline_s_p_o, s consumer)
2. Load S from tmem → registers
3. Apply masking (causal/local/sliding window):
   - Causal: mask future positions
   - Local: mask positions outside window
   - mask_mod: custom mask function
4. Compute row_max:
   - For first iteration: m = max(S)
   - For subsequent: m = max(prev_m, new_m)
5. update acc_scale in sScale
   - if log2 max diff smaller than 8, then acc_scale = 1.0
   - inform correction warps by sm_stats_barrier
6. Apply exp2 transformation:
   - P = exp2(S * scale - row_max * scale)
   - update tSrS_t2r with new row_max
   - apply ex2 and save to tSrP_r2t
      - for 4 frgs (each 32 along 128 cols)
         - for 128 lanes, every other 2
            - if frg 0 and 3 (ex2_emu_start_frg)
               - normal ex2 on SFU
            - if frg 1 and 2, every 12 (ex2_emu_freq) rows
               - 8 (ex2_emu_freq-ex2_emu_res) rows do normal ex2 on SFU
               - 4 (ex2_emu_res) rows do emulated ex2 on ALU
7. Store P to tmem
   - first 3/4:
      - wait for tmem st finished (tcgen05.wait::st)
      - release for P tmem to be used by PV mma (pipeline_s_p_o, p consumer (acutally producer))
   - last 1/4:
      - wait for tmem st finished (tcgen05.wait::st)
      - release for P tmem to be used by PV mma (pipeline_p_lastsplit)
8. Compute row_sum:
   - wait for pipeline_sm_stats as producer
   - For first iteration: r = sum(P)
   - For subsequent: r = prev_r * exp2(prev_m - new_m) + sum(new_P)

Epilogue
1. update sScale[_, tidx, stage]
2. inform correction warps by sm_stats_barrier[stage][wid]

After all work tiles
1. pipeline_sm_stats.producer_tail
```

**Softmax Algorithm (Log-space stability):**
```
Standard: P[i] = exp(S[i]) / sum(exp(S[j]))

Stable:   m = max_row(S)
           P[i] = exp(S[i] - m)
           r = sum(P)
           O = m + log(r)

The kernel maintains: row_max, row_sum in registers
```

**Optimization**:
- save the softmax stats as each row with a thread, to avoid warp sync
- scale threshold
- ALU softmax emulation
- tmem store 32 a time: save register pressure. but need split_p to help save time.
- tmem allocation: 2x128 for O, 2x128 for S overlapping with P, this way, Q stages won't contend each other on S

### 4.5 Correction Operations (Warps 8-11)

The correction warps handle the online normalization and final scaling:

```
Prologue work_tile
1. First O no need scale, release o for mma (pipeline_s_p_o, o consumer, actually producer)

Prologue
1. First O no need correction, release sm stats (pipeline_sm_stats, sm_stats_barrier)

For each block iteration:
1. Wait for softmax statistics sScale(row_sum is now acc_scale) via named barrier (sm_stats_barrier)
2. Vote if any of the thread require rescale (exceed the rescale threshold)
3. Load O from tmem, multiply by scale, store back (correction_rescale)
4. release O for mma (pipeline_s_p_o, o consumer (acutualy producer))
5. release sm stats (pipeline_sm_stats)

Epilogue
1. For final block:
   - wait for row_sum (sm_stats_barrier)
   - wait for row_max (if output lse)
   - Compute final scale = 1.0 / row_sum
   - wait for the last O from mma (pipeline_o_acc)
   - acquire producer lock (pipeline_o_epi)
   - Load O from tmem, multiply by scale, store back (correction_epilogue)
   - release (pipeline_o_acc)
   - commit (pipeline_o_epi)
2. for LSE
   - rescale row_sum and store to global

Epilogue work_tile
1. pipeline_o_epi.producer_tail
```

**Rescaling Logic:**
```
When softmax row_max decreases (new block has larger values):
- Previous P values were computed with smaller exp factor
- Need to adjust: O_new = O_old * exp2(old_m - new_m)

Epilogue
```

**Optimization**
- The rescale threshold is very aggressive, never triggered in my case.

### 4.6 Epilogue Operations (Warp 13)

The epilogue warp writes final output to global memory:

```
For each output stage:
1. Wait for corrected O in sO (pipeline_o_epi)
2. If using TMA_O:
   - Issue TMA store to global memory
   - Commit async bulk group
   - Wait for completion
3. Else (non-TMA path):
   - Copy sO → registers
   - Predicate based on sequence length
   - Store to global memory
4. Signal pipeline release (pipeline_o_epi)
```

---

## 5. Pipeline Synchronization

### Critical Pipeline Paths

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                                                                         │
│                       pipeline_s_p_o    pipeline_sm_stats               │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐            │
│  │ Q Load  │────▶│  MMA    │────▶│ Softmax │────▶│Correct  │            │
│  │(TMA)    │     │ Q×K→S   │     │ S→P     │     │ O/scale │            │
│  └─────────┘     └─────────┘     └─────────┘     └─────────┘            │
│       │                               │  │           ▲  │               │
│       │               ───────────────────────────────│──│               │
│       │               │                  │           │  │               │
│       ▼               ▼                  │           │  ▼               │
│  pipeline_q    pipeline_s_p_o            │           │ pipeline_o_epi   │
│                       │                  │           │        │         │
│                       ▼                  │           │        ▼         │
│  ┌─────────┐     ┌──────┐     ┌───┐      │           │    ┌─────────┐   │
│  │ K/V Load│────▶│  MMA │────▶│1/4│      │           │    │Epilogue │   │
│  │(TMA)    │     │ P×V→O│     │   │      │           │    │ store O │   │
│  └─────────┘     └──────┘     └───┘      │           │    └─────────┘   │
│       │               │         ▲        │           │                  │
│       ▼               ▼         |        ▼           │                  │
│  pipeline_kv     ┌─────────┐   pipeline_p_lastsplit  │                  │
│                  │  last   │                         │                  │
│                  └─────────┘                         │                  │
│                       │                              │                  │
│                       ▼                              │                  │
│                 pipeline_o_acc────────────────────────                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

NOTE: 
1. current version of flash-attention custom PipelineAsync has a bug of producer_commit (it uses AsyncLoad logic) so every producer_cmmit have to be replaced with sm_stats_barrier.arrive
2. pipeline_s_p_o makes mma and softmax both prodcuer and consumer:
   a. mma: finish S, producer_commit
   b. softmax: consumer_wait S, finish P, consumer_release(num=4)
   c. correction: wait softmax finish S_{i}, rescale O_{i-1}, consumer_release(num=4)
   d. mma: producer_acquire(num=8), O += PV

### Pipeline Definitions

| Pipeline | Producer | Consumer | Purpose |
|----------|----------|----------|---------|
| `pipeline_q` | Load warp | MMA warp | Q data availability |
| `pipeline_kv` | Load warp | MMA warp | K/V data availability |
| `pipeline_s_p_o` | MMA warp | Softmax | S ready |
| `pipeline_s_p_o` | Softmax | MMA warp | P 3/4 ready |
| `pipeline_s_p_o` | Correction warp | MMA warp | O rescaled ready |
| `pipeline_p_lastsplit` | Softmax warp | MMA warp | P last 1/4 ready → P×V start |
| `pipeline_o_acc` | MMA warp | Correction warp | final O accumulated ready |
| `pipeline_sm_stats` | Softmax warp | Correction warp | row_max/row_sum ready |
| `pipeline_o_epi` | Correction warp | Epilogue warp | O finalized ready |

---

## 7. Timeline

```
Legend: ████ = Active computation, ░░░░ = Stalled/Waiting

What you thought it is 

TIME ─────────────────────────────────────────────────────────────────────────────▶

LOAD  │████████████████│░░░│████████████████│░░░│████████████████│░░░│
      Load K,V (next tile) 

MMA   │██████│██████│░░░░░░│██████│██████│██████│██████│██████│██████│
       Q0×Kn-1 Q1×Kn-1      P0×Vi Q0×Ki-1 P1×Vi Q1×Ki-1 P0×Vi-1 Q0×Ki-2

SOFT  │░░░░░░│█████████████│█████████████│█████████████│█████████████│
               P0,n-1         P1,n-1   ...  P0,i-1        P1,i-1

What actually it is (rescale O occupy critical path)

TIME ─────────────────────────────────────────────────────────────────────────────▶

LOAD  │████████████████│░░░░░░░░│████████████████│░░░░░░░░│████████████████│░░░░░░░░│
      Load K,V (next tile) 

MMA   │██████│██████│░░░░░░│░░░░│██████│██████│░░░░│██████│██████│░░░░│██████│██████│
       Q0×Kn-1 Q1×Kn-1           P0×Vi  Q0×Ki-1     P1×Vi  Q1×Ki-1     P0×Vi-1 Q0×Ki-2

SOFT  │░░░░░░│█████████████│████│█████████████│████│█████████████│████│█████████████│
               P0,n-1        O0   P1,n-1        O1    P0,i-1       O0    P1,i-1


What Correction Warps Optimize

TIME ─────────────────────────────────────────────────────────────────────────────▶

LOAD  │████████████████│░░░│████████████████│░░░│████████████████│░░░│
      Load K,V (next tile) 

MMA   │██████│██████│░░░░░░│██████│██████│██████│██████│██████│██████│
       Q0×Kn-1 Q1×Kn-1      P0×Vi Q0×Ki-1 P1×Vi Q1×Ki-1 P0×Vi-1 Q0×Ki-2

SOFT  │░░░░░░│█████████████│█████████████│█████████████│█████████████│
               P0,n-1         P1,n-1   ...  P0,i-1        P1,i-1

CORR  │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│████│░░░░░░░░│████│░░░░░░░░│
                                        Correct O0    Correct O1

EPI   │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│████████████│


Why use double Q?

TIME ─────────────────────────────────────────────────────────────────────────────▶

MMA   │██████│░░░░░░░░░░░░░│██████│██████│██████│██████│██████│██████│
       Q×Kn-1               P×Vn-1 Q×Kn-2 P×Vn-2 Q×Kn-3 P×Vn-3 Q×Kn-4

SOFT  │░░░░░░│█████████████│█████████████│█████████████│█████████████│
               P,n-1         P,n-2         P,n-3        P,n-4

CORR  │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│████│░░░░░░░░│████│░░░░░░░░│
                                        Correct O     Correct O (Conflict with PV!)

What Split P Optimize

TIME ─────────────────────────────────────────────────────────────────────────────▶


MMA       ░░░░░░░░░░░░░│██████│██████│██████│██████│██████│██████│
                            

SOFT  │░░░░░░│█████████████│█████████████│█████████████│█████████████│

```

**Optimization**
Why split P can help on the critical path?
I suspect that the tcgen05 mma issue takes a long ALU execution time.

---

## 8. TMEM Layout

```
Tensor Memory Allocation (512 columns total):

┌───────────────────────────────────────────────────┐
│  0-127     │ 128-255    │ 256-383    │ 384-511    │
│  tS[stage0]│  tS[stage1]│  tO[stage0]│  tO[stage1]│
│  (S scores)│  (S scores)│  (O accum) │  (O accum) │
└───────────────────────────────────────────────────┘
     │           │
     │ tmem_s_to_p_offset = 64
     ▼           ▼
┌──────────────────────────────────────────────────────────┐
│  tP (Softmax probabilities) shares storage with tS       │
│  tP[stage] = tS[stage] + 64 (offset by n_block_size/2)   │
└──────────────────────────────────────────────────────────┘
```

---
