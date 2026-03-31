---
layout: article
title: "[CuTeDSL] Understanding Tile Schedulers for Blackwell"
date: 2026-03-31
---

In this article, we explore **Tile Schedulers** - a critical abstraction for GPU kernel work distribution. We cover both CuTe native schedulers (for general GEMM) and FlashAttention's custom schedulers (for attention-specific workloads).

## TL;DR

- **Two families**: CuTe Native (GEMM-focused) vs FlashAttention (attention-specific)
- **Key concepts**: Persistent tiling, CLC dynamic scheduling, LPT load balancing
- **Usage pattern**: Create arguments → Get grid shape → Kernel loop with `initial_work_tile_info()` → `advance_to_next_work()` → `get_current_work()`

---

# Part 1: CuTe Native Tile Schedulers

CuTe provides foundational tile schedulers for general GEMM operations. Located in `cutlass/python/CuTeDSL/cutlass/utils/`.

## 1.1 What is a Tile Scheduler?

In GPU kernels, work must be distributed across CTAs (Cooperative Thread Arrays). For GEMM, each CTA processes a tile of the output matrix. The **Tile Scheduler** abstracts this work distribution, providing:

- `get_grid_shape()`: Grid dimensions for kernel launch (host side)
- `initial_work_tile_info()`: First work assignment (device side)
- `get_current_work()`: Current tile coordinates after advancing
- `advance_to_next_work()`: Move to next tile (persistent kernels)

## 1.2 CuTe Native Scheduler Types

| Scheduler | File | Purpose |
|-----------|------|---------|
| **StaticPersistentTileScheduler** | static_persistent_tile_scheduler.py:337 | Foundational persistent scheduler - each CTA processes multiple tiles with strided advancement |
| **StaticPersistentRuntimeTileScheduler** | static_persistent_tile_scheduler.py:601 | Runtime-aware - always launches all SMs, validity determined at runtime |
| **ClcDynamicPersistentTileScheduler** | dynamic_persistent_tile_scheduler.py:103 | Dynamic work distribution via Cluster Launch Control (CLC) |
| **GroupedGemmTileSchedulerHelper** | grouped_gemm_tile_scheduler_helper.py:139 | Grouped GEMM - translates linear index to group-specific coordinates |
| **StaticPersistentGroupTileScheduler** | grouped_gemm_persistent_tile_scheduler.py:195 | Combines persistence with grouped GEMM |

### 1.2.1 StaticPersistentTileScheduler

The foundational scheduler for persistent GEMM kernels. Each CTA:
1. Starts with `blockIdx.z` as initial work index
2. Processes that tile
3. Advances by `num_persistent_clusters` stride
4. Repeats until work is exhausted

**Why `blockIdx.z`?** The grid is launched as `(cluster_shape_m, cluster_shape_n, num_clusters)`. Each cluster is a group of CTAs working together. `blockIdx.z` identifies which cluster this CTA belongs to, and within persistent scheduling, `bidz` becomes the initial linear work index.

```python
# From static_persistent_tile_scheduler.py:450-453
@staticmethod
def create(params, block_idx, grid_dim):
    bidx, bidy, bidz = block_idx
    
    # Initial work index = cluster index in the grid
    current_work_linear_idx = Int32(bidz)
    
    # Number of persistent clusters = grid_size // cluster_size
    num_persistent_clusters = cute.size(grid_dim) // cute.size(params.cluster_shape_mn)
    
    # CTA position within its cluster
    cta_id_in_cluster = (
        Int32(bidx % params.cluster_shape_mn[0]),
        Int32(bidy % params.cluster_shape_mn[1]),
        Int32(0),
    )
```

**Strided advancement pattern:**
```python
# From advance_to_next_work
def advance_to_next_work(self, advance_count=1):
    self._current_work_linear_idx += Int32(advance_count) * Int32(
        self.num_persistent_clusters
    )
```

If we launch 100 clusters and have 1000 tiles to process:
- Cluster 0 processes tiles: 0, 100, 200, 300, ...
- Cluster 1 processes tiles: 1, 101, 201, 301, ...
- Cluster 99 processes tiles: 99, 199, 299, 399, ...

This ensures even distribution without runtime coordination between clusters.

**Usage example (from sm103_dense_blockscaled_gemm_persistent.py):**

```python
# Host side: setup params and grid
tile_sched_params = utils.PersistentTileSchedulerParams(
    problem_shape_mnl=problem_shape,
    cluster_shape_mn=cluster_shape,
    ...
)
grid = utils.StaticPersistentTileScheduler.get_grid_shape(
    tile_sched_params, max_active_clusters
)

# Device side: create scheduler and loop
tile_sched = utils.StaticPersistentTileScheduler.create(
    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
)
work_tile = tile_sched.initial_work_tile_info()

while work_tile.is_valid_tile:
    # Get tile coordinates (m_tile, n_tile, l_tile)
    cur_tile_coord = work_tile.tile_idx
    
    # ===== YOUR GEMM COMPUTATION HERE =====
    # Load A, B tiles
    # Compute MMA
    # Store C tile
    # ======================================
    
    # Advance to next tile
    tile_sched.advance_to_next_work()
    work_tile = tile_sched.get_current_work()
```

**Grid shape calculation (host):**
```python
# Limit grid to number of SMs available
max_active_clusters = min(sm_count, problem_clusters)
return (cluster_shape_m, cluster_shape_n, max_active_clusters)
```

### 1.2.2 StaticPersistentRuntimeTileScheduler

Use when you always want to launch all SMs, regardless of problem size. Makes `is_valid_tile` always true and relies on runtime bounds checking.

```python
# Key difference from StaticPersistent:
# - Always launches max SMs
# - is_valid_tile is always True
# - Validity determined by runtime tile coordinates
```

**Use case:** When problem size varies and you want deterministic launch overhead.

### 1.2.3 ClcDynamicPersistentTileScheduler

**New in Blackwell (SM100):** Uses NVIDIA's "Cluster Launch Control" for dynamic work distribution.

**The static scheduler problem:** If some SMs are unavailable (e.g., running other work), persistent clusters become imbalanced.

**CLC solution:** Each cluster queries runtime for work dynamically.

```python
# From dense_gemm_persistent_dynamic.py:817-822
# Device side - requires clc_response_ptr in shared memory
tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
    tile_sched_params,
    cute.arch.block_idx(),
    cute.arch.grid_dim(),
    clc_response_ptr,  # Shared memory pointer for CLC response
)
work_tile = tile_sched.initial_work_tile_info()

while work_tile.is_valid_tile:
    cur_tile_coord = work_tile.tile_idx
    # ... compute on tile ...
    
    # Advance uses CLC query mechanism
    clc_pipeline.consumer_wait(clc_consumer_state)
    tile_sched.advance_to_next_work(mbarrier_addr)
    clc_pipeline.consumer_release(clc_consumer_state)
    work_tile = tile_sched.get_current_work()
```

**Key difference from static:**
- Static: `advance_to_next_work()` just increments index
- CLC: `advance_to_next_work()` queries runtime via `issue_clc_query`

**Use case:** When SM availability varies (mixed workloads, MIG), CLC provides better load balancing.

**Reference:** [NVIDIA CUTLASS Blackwell CLC Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_cluster_launch_control.html)

### 1.2.4 GroupedGemmTileSchedulerHelper

For grouped GEMM where each group has different (M, N, K) dimensions. Uses binary search via warp primitives to find which group a linear index belongs to.

```python
# Host side: setup with problem sizes for each group
group_gemm_ts_helper = utils.GroupedGemmTileSchedulerHelper(
    problem_sizes_mnkl=problem_sizes,  # List of (M, N, K, L) for each group
    cluster_shape_mn=cluster_shape,
    ...
)

# Device side: find group for linear tile index
# Uses warp-level vote and shuffle for efficient search
group_idx = group_gemm_ts_helper.get_group_idx(linear_idx)
tile_coord = group_gemm_ts_helper.get_tile_coord(linear_idx)
```

**Use case:** Batched GEMM with varying problem sizes per batch.

### 1.2.5 StaticPersistentGroupTileScheduler

Combines persistent tiling with grouped GEMM. Each persistent CTA processes tiles from different groups while tracking group state.

```python
# From grouped_gemm.py:768-776
# Host side: launch one CTA per SM for persistent grouped GEMM
tile_sched = utils.StaticPersistentGroupTileScheduler.create(
    tile_sched_params,
    bid,                         # block index (linear)
    grid_dim,                    # grid dimensions
    cluster_tile_shape_mnk,      # cluster tile shape
    utils.create_initial_search_state(),
    group_count,                 # number of GEMM groups
    problem_sizes_mnkl,          # (M, N, K, L) for each group
)
work_tile = tile_sched.initial_work_tile_info()

while work_tile.is_valid_tile:
    # Get group info and tile coordinates
    group_info = work_tile.group_search_result
    cur_group_idx = group_info.group_idx
    cur_m_tile = group_info.problem_shape_m  # Varies per group!
    cur_n_tile = group_info.problem_shape_n
    cur_k_tiles = group_info.cta_tile_count_k
    
    # Handle tensormap updates when group changes
    if cur_group_idx != last_group_idx:
        # Update tensormaps for new group's A, B matrices
        update_tensormap_for_group(cur_group_idx, ...)
    
    # Compute on tile for current group
    ...
    
    tile_sched.advance_to_next_work()
    work_tile = tile_sched.get_current_work()
```

**Why group tracking matters:** Each group has different problem dimensions, so tensor descriptors (tensormaps for TMA) need updating when transitioning between groups.

---

# Part 2: FlashAttention Tile Schedulers

FlashAttention builds custom schedulers on top of CuTe's abstractions, optimized for attention-specific workloads. Located in `flash_attn/cute/tile_scheduler.py`.

## 2.1 Why FlashAttention Needs Custom Schedulers

FlashAttention has fundamentally different requirements than GEMM:

| Aspect | GEMM | Attention |
|--------|------|-----------|
| Work dimensions | (m_tile, n_tile) | (block, head, batch, split_kv) |
| Typical pattern | Regular | Variable (causal, varlen) |
| Data reuse | A, B matrices | K, V repeatedly read |
| Persistence | Reduce launch overhead | + L2 cache optimization |

Key challenges:
1. **Split-KV support** - Long KV sequences need chunking across CTAs
2. **L2 cache optimization** - K/V repeatedly loaded, need smart ordering
3. **Variable sequence length** - Different Q lengths per batch
4. **Load balancing** - Causal masking creates uneven work

## 2.2 FlashAttention Scheduler Types

| Scheduler | Line | Use Case |
|-----------|------|----------|
| **SingleTileScheduler** | 56 | Basic non-persistent attention |
| **StaticPersistentTileScheduler** | 155 | Persistent attention kernel |
| **SingleTileLPTScheduler** | 251 | Forward pass with LPT load balancing |
| **SingleTileLPTBwdScheduler** | 377 | Backward pass with LPT |
| **SingleTileVarlenScheduler** | 501 | Variable-length sequences |

### 2.2.1 Scheduler Selection Logic

FlashAttention automatically selects the scheduler based on problem characteristics:

```python
# From flash_fwd_sm100.py:542-552
if mCuSeqlensQ is not None or mSeqUsedQ is not None:
    # Variable sequence length → use varlen scheduler
    TileScheduler = SingleTileVarlenScheduler
else:
    if is_causal or is_local:
        # Causal/local attention → LPT scheduler for load balancing
        TileScheduler = SingleTileLPTScheduler
    else:
        # Standard attention
        TileScheduler = (
            SingleTileScheduler           # Non-persistent
            if not is_persistent
            else StaticPersistentTileScheduler  # Persistent
        )
```

**Why this selection:**
- **Varlen**: Different sequence lengths require runtime tile validity checking
- **Causal/Local**: Load imbalance from masking requires LPT scheduling
- **Persistent vs Non-persistent**: Based on grid size relative to SM count

### 2.2.2 WorkTileInfo

All FlashAttention schedulers return the same structure:

```python
work_tile.tile_idx        # (m_block, head_idx, batch_idx, split_idx)
work_tile.is_valid_tile   # Whether CTA has valid work
```

### 2.2.3 SingleTileScheduler

Basic non-persistent scheduler. Each CTA processes one tile.

```python
# Grid shape: (num_block, num_head * num_splits, num_batch)
@staticmethod
def get_grid_shape(params):
    return (
        round_up(params.num_block, params.cluster_shape_mn[0]),
        params.num_head * params.num_splits,
        params.num_batch,
    )
```

**Use case:** Simple attention kernels, supports Split-KV.

### 2.2.4 StaticPersistentTileScheduler

Persistent scheduler for attention. Each CTA processes multiple tiles in strided fashion.

```python
# Grid shape: limited by SM count
@staticmethod
def get_grid_shape(params):
    sm_count = hardware_info.get_device_multiprocessor_count()
    max_ctas = (sm_count // params.cluster_shape_m) * params.cluster_shape_m
    return (min(max_ctas, params.total_blocks_cluster), 1, 1)
```

**Use case:** Better GPU utilization when grid size is small relative to SM count.

## 2.3 LPT (Longest Processing Time First) Scheduling

**The load balancing problem:** In attention kernels, worktiles have varying execution times:
- Causal masking: tiles near diagonal process more elements
- Varlen: different sequence lengths per batch
- Result: SMs finish at different times, causing inefficiency

### 2.3.1 Why Standard Ordering is Suboptimal

For causal masking, standard grid order `(mblocks, heads, batches)` processes tiles left-to-right. But scores above diagonal are masked, so for fixed head and batch, SMs process **worktiles from shortest to longest** – opposite of optimal load balancing.

A naive "longest first" ordering is also suboptimal:
- Different batches won't hit L2 cache for KV loads
- Loading all KV heads first can thrash L2 if they exceed capacity

### 2.3.2 LPT Solution

FlashAttention-4 applies the classical **Longest Processing Time First (LPT)** scheduling from parallel processor theory:

1. **Always process batches as outermost dimension** – ensures KV cache locality
2. **Swizzle over heads** – divide heads into sections that don't overflow L2
3. **Traverse grid**: heads per section → mblocks in **reverse order** → sections → batches

```
Traditional order: (mblocks, heads, batches) left-to-right
                   → Shortest to longest worktiles (suboptimal)

LPT order:          batches(outer) → sections → mblocks(reverse) → heads(section)
                   → Longest to shortest within each batch, L2-optimized
```

For MQA/GQA: traverse all query heads per KV head before varying mblocks.

**Empirical results** (H200 GPU, BF16, head_dim=128):
- MHA: 4-8% FLOPS gain
- MQA-8: 7-14% FLOPS gain

### 2.3.3 Swizzle Calculation

The number of heads per L2 section is computed based on L2 cache size:

```python
size_one_head = seqlen_k * (headdim + headdim_v) * element_size
size_l2 = 50 * 1024 * 1024  # 50 MB for K & V
swizzle = power_of_2(size_l2 // size_one_head)  # Round to power of 2
```

### 2.3.4 LPT for Variable Sequence Length (varlen)

Load imbalance from variation among batches:
- Different batches attend to different context lengths
- Mixed prefill + decode workloads

**Solution:** Preprocessing kernel sorts batches by max per-worktile execution time:

```python
# LPT varlen approach:
# 1. Preprocess: sort batches by seqlen_q * seqlen_k
# 2. Store: batch_mapping[virtual_batch] = actual_batch
# 3. Scheduler: traverse batches in sorted order
# 4. Attention kernel: use mapping for correct memory access
```

**Key insight:** Sorting metadata can be cached, no performance loss from preprocessing.

## 2.4 Split-KV

**Problem:** Very long KV sequences don't fit in shared memory.

**Solution:** Split KV into chunks, each CTA processes one chunk, then combine results.

```python
# Standard: One CTA processes ALL KV
# Split-KV: num_splits CTAs each process 1/num_splits of KV

# Grid becomes: (num_block, num_head * num_splits, num_batch)
# Split index stored in work_tile.tile_idx[3]
```

## 2.5 Example Custom Schedulers

| Scheduler | File | Purpose |
|-----------|------|---------|
| **FmhaStaticTileScheduler** | helpers/fmha_helpers.py:84 | FMHA with persistent/non-persistent toggle |
| **MLAStaticTileScheduler** | blackwell/mla.py:176 | Multi-Latent Attention (DeepSeek) |
| **Mamba2SSDTileScheduler** | blackwell/mamba2_ssd/... | Mamba2 state space models |

### 2.5.1 MLAStaticTileScheduler

DeepSeek's Multi-Latent Attention compresses KV cache into a latent representation. The scheduler supports both persistent and non-persistent modes:

```python
# From mla.py:1052-1080
# Host side: create params
tile_sched_params = create_mla_static_tile_scheduler_params(
    is_persistent=True,
    problem_shape_b=batch_size,
    cluster_shape_mnk=cluster_shape,
    split_kv=split_kv_count
)
grid_shape = MLAStaticTileScheduler.get_grid_shape(tile_sched_params, max_active_clusters)

# Device side: create scheduler and loop
tile_sched = create_mla_static_tile_scheduler(
    tile_sched_params,
    cute.arch.block_idx(),
    cute.arch.grid_dim()
)
work_tile = tile_sched.initial_work_tile_info()

while work_tile.is_valid_tile:
    blk_coord = work_tile.tile_idx
    
    # For split-KV: determine KV chunk for this tile
    k_index, k_tile_count, local_split_kv = self.get_k_tile_count(split_kv, ...)
    
    # ... attention computation ...
    
    tile_sched.advance_to_next_work()
    work_tile = tile_sched.get_current_work()
```

**Key feature:** `is_persistent` flag switches between persistent and non-persistent modes.

### 2.5.2 FmhaStaticTileScheduler

FMHA-specific scheduler with support for masked attention patterns.

```python
# From fmha_helpers.py:148-177
@staticmethod
def get_grid_shape(params):
    if params.is_persistent:
        # Persistent: limited by SM count
        sm_count = hardware_info.get_device_multiprocessor_count()
        return (min(sm_count, size(params.problem_shape_mbh)), 1, 1)
    else:
        # Non-persistent: matches problem shape
        return params.problem_shape_mbh
```

**Grid coordinates:** `(m_block, batch, head)` for FMHA.

### 2.5.3 Mamba2SSDTileScheduler

State space models use different coordinate semantics:

```python
# From mamba2_ssd_tile_scheduler.py:165-178
def _get_current_work_for_linear_idx(self, current_work_linear_idx):
    is_valid = current_work_linear_idx < size(problem_shape_ntiles)
    
    # Decode (batch, expert_idx, group_idx) from linear index
    eh_idx = current_work_linear_idx % params.eh
    b_idx = current_work_linear_idx // params.eh
    g_idx = eh_idx // params.ngroup_ratio
    
    return WorkTileInfo((b_idx, eh_idx, g_idx), is_valid)
```

**Grid coordinates:** `(batch_idx, expert_hidden_idx, group_idx)` - specific to Mamba2 SSD.

---

# Part 3: How to Use

## 3.1 Host Side: Create Arguments and Grid

```python
from flash_attn.cute.tile_scheduler import (
    TileSchedulerArguments,
    StaticPersistentTileScheduler,
)

# Create arguments
tile_sched_args = TileSchedulerArguments(
    num_block=ceil_div(seqlen_q, m_block_size),
    num_head=num_heads,
    num_batch=batch_size,
    num_splits=1,           # 1 = no split-KV
    seqlen_k=seqlen_k,
    headdim=head_dim,
    headdim_v=head_dim_v,
    total_q=seqlen_q * batch_size,
    tile_shape_mn=(128, 128),
    cluster_shape_mn=(2, 1),
    is_persistent=True,
    is_split_kv=False,
    lpt=False,
)

# Get parameters and grid
tile_sched_params = StaticPersistentTileScheduler.to_underlying_arguments(tile_sched_args)
grid_dim = StaticPersistentTileScheduler.get_grid_shape(tile_sched_params)

# Launch kernel
kernel[grid_dim](..., tile_sched_params)
```

## 3.2 Device Side: Process Tiles

```python
@cute.jit
def kernel(..., tile_sched_params):
    # Create scheduler
    tile_scheduler = StaticPersistentTileScheduler.create(tile_sched_params)
    
    # Get initial work
    work_tile = tile_scheduler.initial_work_tile_info()
    
    # Main loop
    while work_tile.is_valid_tile:
        m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
        
        # ===== YOUR COMPUTATION HERE =====
        # Load Q, K, V for tile (m_block, head_idx, batch_idx)
        # Compute attention
        # Write output
        # ==================================
        
        # Advance to next tile
        tile_scheduler.prefetch_next_work()
        tile_scheduler.advance_to_next_work()
        work_tile = tile_scheduler.get_current_work()
```

## 3.3 Key Methods

| Method | Purpose | Side |
|--------|---------|------|
| `get_grid_shape()` | Compute grid dimensions | Host only |
| `to_underlying_arguments()` | Convert args to params | Host only |
| `create()` | Instantiate scheduler | Device only |
| `initial_work_tile_info()` | Get first work tile | Device only |
| `get_current_work()` | Get current tile | Device only |
| `advance_to_next_work()` | Move to next tile | Device only |

---

# Summary

## Scheduler Selection Guide

| Scenario | Recommend Scheduler |
|----------|---------------------|
| Generic GEMM | `StaticPersistentTileScheduler` |
| Variable SM availability | `ClcDynamicPersistentTileScheduler` |
| Grouped GEMM | `StaticPersistentGroupTileScheduler` |
| Attention forward pass | `SingleTileLPTScheduler` |
| Attention backward pass | `SingleTileLPTBwdScheduler` |
| Attention varlen | `SingleTileVarlenScheduler` |
| Custom attention | Derive from `WorkTileInfo` |

## Key Takeaways

1. **CuTe Native schedulers** provide foundational work distribution for GEMM
2. **CLC (Cluster Launch Control)** enables dynamic work queries in Blackwell
3. **FlashAttention customizes** for attention-specific needs (L2 cache, varlen, Split-KV)
4. **LPT scheduling** solves load imbalance from causal masking and varlen
5. **Persistent kernels** reduce launch overhead when grid < SM count

The Tile Scheduler is a powerful abstraction that separates work distribution logic from core computation, enabling experimentation with different tiling strategies.