---
layout: article
title: "[FlashAttention] Variable-Length Sequences with SeqlenInfo"
date: 2026-04-01
---

In FlashAttention, sequences often have different lengths within a batch. This article explains how **SeqlenInfo** handles variable-length sequences efficiently.

## TL;DR

- **Varlen** = Variable-length sequences packed into one tensor
- **cu_seqlens** = Cumulative sequence lengths (prefix sum), used to index into packed tensors
- **SeqlenInfo** consolidates all sequence-length-related reads into one placefor efficiency
- **offset_batch()** extracts the tensor slice for a specific batch from packed data

---

# Part1: The Problem - Variable-Length Sequences

## 1.1 Padded vs Packed Tensors

### Naive Approach: Padding

```python
# Batch with 3 sequences of different lengths:
# Batch 0: 100 tokens
# Batch 1: 50 tokens
# Batch 2: 200 tokens

# Padded tensor (wasteful):
# Shape: (max_seqlen=200, num_heads=16, batch_size=3)
# Memory: 200 *16 * 128 * 3 = 1,228,800 elements
# But only using: (100 + 50 + 200) * 16 * 128 = 716,800 elements
# Wasted: 42%!
```

### Efficient Approach: Packed Tensor

```python
# Packed tensor (efficient):
# Shape: (total_tokens=350, num_heads=16)
# Memory: 350 * 16 * 128 = 716,800 elements
# No waste!

# Layout in memory:
# Indices [0:100)    → Batch 0 tokens
# Indices [100:150)  → Batch 1 tokens
# Indices [150:350)  → Batch 2 tokens
```

## 1.2 cu_seqlens Format

To navigate the packed tensor, we use **cumulative sequence lengths**:

```python
# Sequence lengths: [100, 50, 200]
seqlens = torch.tensor([100, 50, 200])

# Cumulative sequence lengths (prefix sum):
cu_seqlens = torch.zeros(len(seqlens) + 1, dtype=torch.int32)
cu_seqlens[1:] = seqlens.cumsum(0)
# Result: [0, 100, 150, 350]
#          ↑  ↑    ↑    ↑
#          |  |    |    └── total_tokens (end of all sequences)
#          |  |    └─────── end of batch 1 (offset 150)
#          |  └──────────── end of batch 0 (offset 100)
#          └─────────────── start (always 0)

# To find batch boundaries:
# Batch i occupies: cu_seqlens[i] : cu_seqlens[i+1]
# Batch 0: [0:100)
# Batch 1: [100:150)
# Batch 2: [150:350)

# To get sequence length for batch i:
seqlen[i] = cu_seqlens[i+1] - cu_seqlens[i]
```

### Visual Representation

```
Packed Tensor:  [Batch 0 | Batch 1 | Batch 2         ]
Indices:        [0     99|100   149|150          349]
                 \       / \      / \              /
cu_seqlens:     [0,100,150,350]
                   ↑  ↑   ↑   ↑
                   |  |   |   └─ batch 2 ends at 350
                   |  |   └───── batch 1 ends at 150
                   |  └───────── batch 0 ends at 100
                   └──────────── start at 0
```

---

# Part 2: SeqlenInfo Classes

## 2.1 Three Classes

| Class | Purpose |
|-------|---------|
| **SeqlenInfo** | Single tensor varlen (Q or K alone) |
| **SeqlenInfoQK** | Both Q and K varlen (most common for attention) |
| **SeqlenInfoQKNewK** | Append-KV with left-padding support (for incremental decode) |

## 2.2 SeqlenInfo Class

```python
@dataclass(frozen=True)
class SeqlenInfo:
    offset: Int32          # Start position in packed tensor
    offset_padded: Int32   # Aligned offset (for memory alignment)
    seqlen: Int32          # Actual sequence length
    has_cu_seqlens: bool   # Whether using cu_seqlens or fixed length
```

### Creating SeqlenInfo

```python
@staticmethod
def create(batch_idx, seqlen_static, cu_seqlens=None, seqused=None, tile=128):
    """
    Create SeqlenInfo for a specific batch.
    
    Args:
        batch_idx: Which batch to get info for
        seqlen_static: Fixed sequence length (used if no cu_seqlens)
        cu_seqlens: Cumulative sequence lengths tensor, shape (num_batch + 1,)
        seqused: Alternative to cu_seqlens, shape (num_batch,) - just sequence lengths
        tile: Tile size for alignment (default 128)
    
    Returns:
        SeqlenInfo with offset, offset_padded, seqlen
    """
    # Compute offset (start position in packed tensor)
    offset = 0 if cu_seqlens is None else cu_seqlens[batch_idx]
    
    # Compute aligned offset for better memory access
    # Rounds up to nearest tile boundary
    offset_padded = 0 if cu_seqlens is None else align(offset + batch_idx * tile, tile)
    
    # Compute sequence length
    if seqused is not None:
        seqlen = seqused[batch_idx]
    elif cu_seqlens is not None:
        seqlen = cu_seqlens[batch_idx + 1] - cu_seqlens[batch_idx]
    else:
        seqlen = seqlen_static
    
    return SeqlenInfo(offset, offset_padded, seqlen, has_cu_seqlens=cu_seqlens is not None)
```

### offset_batch() Method

```python
def offset_batch(self, mT, batch_idx, dim, padded=False, multiple=1):
    """
    Offset a packed tensor to access data for a specific batch.
    
    Args:
        mT: Packed tensor, shape (total_tokens, num_heads, head_dim) or similar
        batch_idx: Which batch to access
        dim: Which dimension is the batch dimension
        padded: Use aligned offset (for TMA loads)
        multiple: Multiplier for offset (for paged attention)
    
    Returns:
        Tensor slice for the specified batch
    """
    if not self.has_cu_seqlens:
        # Fixed-length: just index into batch dimension
        # mT.shape = (seqlen, num_heads, batch_size, head_dim)
        idx = (None,) * dim + (batch_idx,) + (None,) * (rank(mT) - 1 - dim)
        return mT[idx]
    else:
        # Variable-length: offset into packed tensor
        off = self.offset if not padded else self.offset_padded
        off *= multiple
        # mT.shape = (total_tokens, num_heads, head_dim)
        idx = (off,) + (None,) * (rank(mT) - 1)
        return cute.domain_offset(idx, mT)
```

---

# Part 3: SeqlenInfoQK - For Attention Kernels

## 3.1 Class Definition

```python
@dataclass(frozen=True)
class SeqlenInfoQK:
    offset_q: Int32        # Q start position in packed tensor
    offset_k: Int32        # K start position in packed tensor
    padded_offset_q: Int32 # Aligned Q offset
    padded_offset_k: Int32 # Aligned K offset
    seqlen_q: Int32        # Q sequence length
    seqlen_k: Int32        # K sequence length
    has_cu_seqlens_q: bool# Whether Q uses varlen
    has_cu_seqlens_k: bool # Whether K uses varlen
    has_seqused_q: bool    # Whether Q uses seqused format
    has_seqused_k: bool    # Whether K uses seqused format
```

## 3.2 Creating SeqlenInfoQK

```python
@staticmethod
def create(
    batch_idx,
    seqlen_q_static,  # Fixed Q length (unused if cu_seqlens provided)
    seqlen_k_static,  # Fixed K length (unused if cu_seqlens provided)
    mCuSeqlensQ=None, # Q cumulative lengths
    mCuSeqlensK=None, # K cumulative lengths
    mSeqUsedQ=None,  # Alternative: Q actual lengths per batch
    mSeqUsedK=None,  # Alternative: K actual lengths per batch
    tile_m=128,      # Q tile size for alignment
    tile_n=128,      # K tile size for alignment
):
    # Compute Q offset and length
    offset_q = 0 if mCuSeqlensQ is None else mCuSeqlensQ[batch_idx]
    padded_offset_q = 0 if mCuSeqlensQ is None else align(offset_q + batch_idx * tile_m, tile_m)
    
    if mSeqUsedQ is not None:
        seqlen_q = mSeqUsedQ[batch_idx]
    elif mCuSeqlensQ is not None:
        seqlen_q = mCuSeqlensQ[batch_idx + 1] - offset_q
    else:
        seqlen_q = seqlen_q_static
    
    # Similarly for K...
    
    return SeqlenInfoQK(...)
```

---

# Part 4: Complete Usage Example

## 4.1 Setup: Input Data

```python
# Example attention problem:
# Batch 0: Q=64 tokens, K=128 tokens
# Batch 1: Q=32 tokens, K=256 tokens
# Batch 2: Q=48 tokens, K=512 tokens
# num_heads = 16, head_dim = 128

# Pack sequences into tensors:
# mQ: shape (total_q=144,num_heads=16, head_dim=128)
# mK: shape (total_k=896, num_heads=16, head_dim=128)
# mV: shape (total_v=896, num_heads=16, head_dim=128)

# Cumulative lengths:
# mCuSeqlensQ = [0, 64, 96, 144]   # 64+32+48=144
# mCuSeqlensK = [0, 128, 384, 896] # 128+256+512=896

# Memory layout visualization:
# Q tensor:  [Batch 0 (64)  | Batch 1 (32) | Batch 2 (48)   ]
# Indices:    [0           63|64         95|96           143]
# K tensor:  [Batch0(128)|Batch1(256)    |Batch2(512)        ]
# Indices:   [0       127|128       383|384            895]
```

## 4.2 Kernel Entry Point

```python
# In FlashAttention forward kernel:
# Tile scheduler gives us: (m_block, head_idx, batch_idx, split_idx)
work_tile = tile_scheduler.initial_work_tile_info()
m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx

# Example: batch_idx = 1, head_idx = 3, m_block = 0
```

## 4.3 Create SeqlenInfoQK

```python
# Create sequence length info for this batch
seqlen_info = SeqlenInfoQK.create(
    batch_idx=batch_idx,        # = 1
    seqlen_q_static=seqlen_q,   # Unused when cu_seqlens provided
    seqlen_k_static=seqlen_k,   # Unused when cu_seqlens provided
    mCuSeqlensQ=mCuSeqlensQ,    # [0, 64, 96, 144]
    mCuSeqlensK=mCuSeqlensK,    # [0, 128, 384, 896]
    tile_m=128,
    tile_n=128,
)

# After create(), seqlen_info contains:
# offset_q = mCuSeqlensQ[1] = 64
# seqlen_q = mCuSeqlensQ[2] - mCuSeqlensQ[1] = 96 - 64 = 32
# offset_k = mCuSeqlensK[1] = 128
# seqlen_k = mCuSeqlensK[2] - mCuSeqlensK[1] = 384 - 128 = 256

print(f"Batch {batch_idx}: Q[{offset_q}:{offset_q+seqlen_q}], K[{offset_k}:{offset_k+seqlen_k}]")
# Output: Batch 1: Q[64:96], K[128:384]
```

## 4.4 Extract Tensor Slices

```python
# Global packed tensors:
# mQ: shape (144, 16, 128) = (total_q_tokens, num_heads, head_dim)
# mK: shape (896, 16, 128) = (total_k_tokens, num_heads, head_dim)

# Get Q slice for batch 1:
mQ_cur = seqlen_info.offset_batch_Q(mQ, batch_idx, dim=1)
# mQ_cur shape: (32, 16, 128)
# This is equivalent to: mQ[offset_q:offset_q+seqlen_q, :, :]
#                     = mQ[64:96, :, :]

# Get specific head from Q:
mQ_head = mQ_cur[:, head_idx, :]# shape: (32, 128)
# Equivalent to: mQ[64:96, 3, :]

# Get K slice for batch 1:
mK_cur = seqlen_info.offset_batch_K(mK, batch_idx, dim=1)
# mK_cur shape: (256, 16, 128)
# This is: mK[offset_k:offset_k+seqlen_k, :, :]
#        = mK[128:384, :, :]
```

## 4.5 Compute Iteration Bounds

```python
# Tile sizes:
tile_m = 128  # Q block size
tile_n = 128  # K/V block size

# Number of Q blocks for this batch:
m_blocks = cute.ceil_div(seqlen_info.seqlen_q, tile_m)
# = ceil(32 / 128) = 1 block

# Number of K blocks for this batch:
n_blocks = cute.ceil_div(seqlen_info.seqlen_k, tile_n)
# = ceil(256 / 128) = 2 blocks

# Why from seqlen_k, not mK_cur?
# - seqlen_k is the EXACT length (integer)
# - mK_cur might have padding beyond seqlen_k for alignment
# - We need integer for loop bounds
```

## 4.6 Complete Attention Loop

```python
# Full attention computation for batch 1, head 3:

# Get Q for this tile (all 32 tokens fit in one block)
# Q tile shape: (tile_m=128, head_dim=128), but only 32 valid
sQ = load_Q_tile(mQ_cur, m_block * tile_m, (m_block + 1) * tile_m)
# Loads Q[0:32] from mQ_cur, pads to 128

# Loop over K blocks
for n_block in range(n_blocks):  # n_blocks = 2
    # Load K tile
    k_start = n_block * tile_n
    k_end = min((n_block + 1) * tile_n, seqlen_info.seqlen_k)
    # n_block=0: k_start=0, k_end=128
    # n_block=1: k_start=128, k_end=256
    
    sK = load_K_tile(mK_cur, k_start, k_end)
    # n_block=0: Loads K[0:128]from mK_cur
    # n_block=1: Loads K[128:256] from mK_cur
    
    # Load V tile (same indices as K)
    sV = load_V_tile(mV_cur, k_start, k_end)
    
    # Compute attention: Q @ K^T
    sS = sQ @ sK.T# shape: (128, 128) but only (32, 128) valid
    
    # Apply causal mask if needed
    if is_causal:
        apply_causal_mask(sS, m_block * tile_m, k_start)
    
    # Softmax
    sP = softmax(sS / sqrt(head_dim))
    
    # Compute output: P @ V
    sO = sP @ sV# shape: (128, head_dim)

# After loop, write output
write_O_tile(sO, mQ_cur, m_block * tile_m)
```

## 4.7 Handling Variable Q Lengths

```python
# What if Q lengths vary significantly?
# Batch0: Q=200, K=500
# Batch 1: Q=10, K=100
# Batch 2: Q=50, K=300

# For batch 0:
# m_blocks = ceil(200 / 128) = 2 blocks (m_block=0, 1)
# n_blocks = ceil(500 / 128) = 4 blocks (n_block=0, 1, 2, 3)

# m_block=0: Q[0:128], process K[0:500]
# m_block=1: Q[128:200], process K[0:500] (only 72 valid Q tokens)

# In kernel:
mask_valid = torch.arange(tile_m) < (seqlen_q - m_block * tile_m)
# For m_block=1: mask_valid = [True]*72 + [False]*56
```

---

# Part 5: Advanced Usage

## 5.1 Padded Offset

```python
# offset_padded aligns to tile boundaries
# Useful for TMA loads that require alignment

# Example:
batch_idx = 2
tile = 128
offset = 150# Raw offset from cu_seqlens
offset_padded = align(offset + batch_idx * tile, tile)
# = align(150 + 256, 128) = align(406, 128) = 512

# This ensures TMA loads start at aligned addresses
mQ_padded = seqlen_info.offset_batch_Q(mQ, batch_idx, dim=1, padded=True)
```

## 5.2 Multiple Offset

```python
# For paged attention, K might be stored in pages
# Need to offset by page_table entry

# multiple = page_size
mK_page = seqlen_info.offset_batch_K(mK, batch_idx, dim=1, multiple=page_size)
# offset_k is multiplied by page_size internally
```

## 5.3 Ragged Tensors

```python
# Ragged tensors have variable first dimension
# Used when memory layout doesn't match cu_seqlens assumptions

mQ_ragged = seqlen_info.offset_batch_Q(mQ, batch_idx, dim=1, ragged=True)
# Handles ptr_shift and other complexities for ragged layouts
```

---

# Part 6: SeqlenInfoQKNewK - For Incremental Decode

## 6.1 Use Case: Append-KV

```python
@dataclass(frozen=True)
class SeqlenInfoQKNewK:
    """For incremental decoding where we append new K/V to cache.
    
    Key fields:
    - leftpad_k: Left padding in KV cache (tokens to skip)
    - seqlen_k_og: Original KV length (before appending)
    - seqlen_k_new: New K tokens to append
    - seqlen_k: Total = seqlen_k_og + seqlen_k_new
    """
    leftpad_k: Int32
    offset_q: Int32
    offset_k: Int32        # Offset into KV cache
    offset_k_new: Int32    # Offset into new K tensor
    seqlen_q: Int32
    seqlen_k_og: Int32     # Original cache length
    seqlen_k_new: Int32    # New tokens to append
    seqlen_k: Int32        # Total K length
    seqlen_rotary: Int32   # Position for rotary embedding
```

## 6.2 Example: Decode Step

```python
# Initial KV cache: 1000 tokens
# Generate 10 new tokens
# Left padding: 50 tokens (for alignment or other reasons)

# After SeqlenInfoQKNewK.create():
# seqlen_k_og = 1000 - 50 = 950  (excluding leftpad)
# seqlen_k_new = 10
# seqlen_k = 950 + 10 = 960
# seqlen_rotary = 1000(leftpad + original)
# offset_k = 50(leftpad offset)
# offset_k_new = 0 (new K starts at beginning of new K tensor)

# During decode:
# - Read K[0:950] from cache (original)
# - Read K[950:960] from new K tensor
# - Concatenate for full KV
```

---

# Summary

## Key Concepts

| Concept | Description |
|---------|-------------|
| **cu_seqlens** | Cumulative sequence lengths, shape (num_batch + 1,) |
| **Packed tensor** | All sequences concatenated, shape (total_tokens, ...) |
| **offset** | Start position of a batch in packed tensor |
| **seqlen** | Actual sequence length for a batch |
| **offset_padded** | Aligned offset for TMA loads |

## Why SeqlenInfo Matters

1. **Efficiency**: Read cu_seqlens once per tile, not repeatedly
2. **Flexibility**: Support both `cu_seqlens` and `seqused` formats
3. **Correctness**: Handle edge cases (leftpad, append-KV, ragged tensors)
4. **Alignment**: `offset_padded` ensures memory alignment for TMA

## Usage Pattern

```python
# 1. Create SeqlenInfo at start of each tile
seqlen_info = SeqlenInfoQK.create(batch_idx, ..., mCuSeqlensQ, mCuSeqlensK)

# 2. Extract tensor slices
mQ_cur = seqlen_info.offset_batch_Q(mQ, batch_idx, dim)
mK_cur = seqlen_info.offset_batch_K(mK, batch_idx, dim)

# 3. Compute iteration bounds
m_blocks = ceil_div(seqlen_info.seqlen_q, tile_m)
n_blocks = ceil_div(seqlen_info.seqlen_k, tile_n)

# 4. Loop over tiles
for m_block in range(m_blocks):
    for n_block in range(n_blocks):
        # Load and compute...
```