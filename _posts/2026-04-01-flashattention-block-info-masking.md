---
layout: article
title: "[FlashAttention] Block-Level Masking with BlockInfo"
date: 2026-04-01
---

In FlashAttention, **causal** and **local attention** masks restrict which Q tokens can attend to which K tokens. The `BlockInfo` class computes **block-level valid ranges** to avoid computing invalid Q-K pairs entirely, saving memory bandwidth and computation.

## TL;DR

- **BlockInfo** computes which K/V blocks a Q block can legally attend to
- **Causal:** Q[i] attends to K[0:i+1+shift] where shift = seqlen_k - seqlen_q
- **Local:** Q[i] attends to K[i-window_left+shift : i+window_right+1+shift]
- **GQA/MQA:** Q positions are divided by `qhead_per_kvhead_packgqa` before computing masks
- **Split-KV:** Further divides K blocks among splits for long sequences

---

# Part 1: The Problem - Element-Level vs Block-Level Masking

## 1.1 Standard Attention (No Mask)

In standard attention, every Q token can attend to every K token:

```
Q\K  0  1  2  3  4
0    ✓  ✓  ✓  ✓  ✓
1    ✓  ✓  ✓  ✓  ✓
2    ✓  ✓  ✓  ✓  ✓
3    ✓  ✓  ✓  ✓  ✓
4    ✓  ✓  ✓  ✓  ✓

All 5×5 = 25 positions are valid.
```

## 1.2 Causal Attention

In causal attention, Q[i] can only attend to K[0:i+1+shift]:

```
Q\K  0  1  2  3  4
0    ✓  ✗  ✗  ✗  ✗    # Q[0] attends to K[0]
1    ✓  ✓  ✗  ✗  ✗    # Q[1] attends to K[0:2]
2    ✓  ✓  ✓  ✗  ✗
3    ✓  ✓  ✓  ✓  ✗
4    ✓  ✓  ✓  ✓  ✓

Only 15 positions are valid (lower triangle + diagonal).
```

**Why shift?** When `seqlen_k > seqlen_q` (common in decode), the attention pattern shifts:
- `shift = seqlen_k - seqlen_q`
- Q[i] attends to K[0: i+1+shift]
- Ensures Q[token] attends to all KV context

## 1.3 Local (Sliding Window) Attention

In local attention, Q[i] attends to K[i-window_left+shift : i+window_right+1+shift]:

```
window_left=1, window_right=1, seqlen_k = seqlen_q (shift=0)

Q\K  0  1  2  3  4
0    ✓  ✓  ✗  ✗  ✗    # Q[0] attends to K[0:2]
1    ✓  ✓  ✓  ✗  ✗    # Q[1] attends to K[0:3]
2    ✗  ✓  ✓  ✓  ✗
3    ✗  ✗  ✓  ✓  ✓
4    ✗  ✗  ✗  ✓  ✓

Each Q position attends to a local window.
```

**Causal + Local:** Can combine both masks.

## 1.4 The Naive Approach: Compute Then Mask

```python
# Load all Q and K tiles
for m_block in range(num_m_blocks):
    sQ = load_Q(m_block)
    for n_block in range(num_n_blocks):
        sK = load_K(n_block)
        
        # Compute attention for ALL positions
        sS = sQ @ sK.T  # (tile_m, tile_n)
        
        # Apply mask element-by-element
        sS = apply_mask(sS, m_block, n_block, causal_mask)
        
        # Many positions are zeroed!
        # Still paid memory bandwidth for K, still computed matmul
```

**Problem:** We load K blocks and compute matmul for positions that will be zeroed.

## 1.5 The Efficient Approach: Skip Invalid Blocks

```python
# Only iterate over valid K blocks
for m_block in range(num_m_blocks):
    sQ = load_Q(m_block)
    
    # Compute VALID K block range
    n_block_min, n_block_max = get_valid_k_blocks(m_block)
    
    for n_block in range(n_block_min, n_block_max):
        sK = load_K(n_block)
        
        # All positions in this block are valid!
        # (or only need minimal masking at boundaries)
        sS = sQ @ sK.T
        
        # No wasted loads or computation
```

---

# Part 2: BlockInfo Class

## 2.1 Class Definition

```python
@dataclass(frozen=True)
class BlockInfo:
    tile_m: Constexpr[int]        # Q block size (typically 128)
    tile_n: Constexpr[int]        # K/V block size (typically 128)
    is_causal: Constexpr[bool]    # Causal masking enabled
    is_local: Constexpr[bool]    # Local/sliding window masking
    window_size_left: Int32       # Local window left size
    window_size_right: Int32      # Local window right size
    is_split_kv: Constexpr[bool] # Split-KV mode
    qhead_per_kvhead_packgqa: int# GQA: Q heads per KV head
```

## 2.2 Key Methods

| Method | Purpose |
|--------|---------|
| `get_n_block_min_max(seqlen_info, m_block)` | Which K blocks can Q block attend to? |
| `get_m_block_min_max(seqlen_info, n_block)` | Which Q blocks can attend to K block? |
| `get_n_block_k_new_min_max(seqlen_info, m_block)` | For append-KV: which new-K blocks? |
| `get_n_block_min_causal_local_mask(...)` | Where does causal mask end? |
| `get_n_block_min_before_local_mask(...)` | Where does local mask start? |

---

# Part 3: get_n_block_min_max() - Finding Valid K Blocks

## 3.1 Method Signature

```python
def get_n_block_min_max(
    self,
    seqlen_info: SeqlenInfoQK,
    m_block: Int32,
    split_idx: Int32 = 0,      # For split-KV
    num_splits: Int32 = 1,      # For split-KV
) -> Tuple[Int32, Int32]:
    """Return (n_block_min, n_block_max) for this Q block."""    ...
```

**Inputs:**
- `seqlen_info`: Contains `seqlen_q`, `seqlen_k` for this batch
- `m_block`: Which Q block we're processing
- `split_idx`, `num_splits`: For split-KV (long sequences)

**Returns:**
- `n_block_min`: First valid K block (inclusive)
- `n_block_max`: Last valid K block (exclusive)

## 3.2 Algorithm

```python
def get_n_block_min_max(self, seqlen_info, m_block, split_idx=0, num_splits=1):
    # Step 1: Start with full K range
    n_block_max = ceil_div(seqlen_info.seqlen_k, self.tile_n)
    
    # Step 2: Apply causal/local UPPER bound
    if self.is_causal or (self.is_local and self.window_size_right is not None):
        # Compute maximum Q position in this block
        m_idx_max = (m_block + 1) * self.tile_m
        
        # GQA: Convert Q position to KV position
        if self.qhead_per_kvhead_packgqa > 1:
            # Multiple Q heads share one KV head
            # Q position q maps to KV position q // qhead_per_kvhead_packgqa
            m_idx_max = ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
        
        # Compute corresponding K index
        # Q[i] attends to K[j] where j <= i + shift
        # shift = seqlen_k - seqlen_q (handles different Q/K lengths)
        n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
        
        # For local attention: extend by window_size_right
        n_idx_right = n_idx if self.is_causal else n_idx + self.window_size_right
        
        # Convert to block index
        n_block_max = min(n_block_max, ceil_div(n_idx_right, self.tile_n))
    
    # Step 3: Apply local LOWER bound
    n_block_min = 0
    if self.is_local and self.window_size_left is not None:
        # Compute minimum Q position in this block
        m_idx_min = m_block * self.tile_m
        
        # GQA adjustment
        if self.qhead_per_kvhead_packgqa > 1:
            m_idx_min = m_idx_min // self.qhead_per_kvhead_packgqa
        
        # Compute corresponding K index
        # Q[i] attends to K[j] where j >= i - window_size_left + shift
        n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q
        n_idx_left = n_idx - self.window_size_left
        
        # Convert to block index (clamp to 0)
        n_block_min = max(n_idx_left // self.tile_n, 0)
    
    # Step 4: Apply split-KV if enabled
    if self.is_split_kv:
        num_n_blocks_per_split = (n_block_max - n_block_min + num_splits - 1) // num_splits
        n_block_min = n_block_min + split_idx * num_n_blocks_per_split
        n_block_max = min(n_block_min + num_n_blocks_per_split, n_block_max)
    
    return n_block_min, n_block_max
```

## 3.3 Visual Example: Causal Attention

```
seqlen_q = 256, seqlen_k = 256
tile_m = 128, tile_n = 128
is_causal = True
m_block = 0 (Q indices [0: 128])

Step 1: n_block_max = ceil(256 / 128) = 2

Step 2: Apply causal upper bound
  m_idx_max = (0 + 1) * 128 = 128
  
  shift = seqlen_k - seqlen_q = 0
  
  # Q[i] attends to K[j] where j <= i+---------------------------
  # Q[127] attends to K[0:128]
  # So all K blocks [0: 128/128] = 1 are valid for causal
  
  n_idx = 128 + 0 = 128
  n_block_max = min(2, ceil(128/128)) = min(2, 1) = 1
  
  # ERROR: Actually for Q[0:128], Q[0] attends to K[0:1]
  # Q[127] attends to K[0:128], all K blocks valid
  # Let me reconsider...

Actually for causal:
  # Q[i] attends to K[0:i+1]
  # Q block [0:128] has Q[0]...Q[127]
  # Q[0] attends to K[0:1]  → n_block_min = 0
  # Q[127] attends to K[0:128] → n_block_max = 1
  
  # For m_block = 0:
  # n_block_min = 0
  # n_block_max = ceil(128/128) = 1
  
  # K block indices: [0, 1)
  # Q[0:128] attends to K[0:128]

But wait, celld328 gets n_block_max from:
  m_idx_max = 128 (max Q index + 1for this block)
  n_idx = 128+ shift = 128
  n_block_max = ceil(128 / 128) = 1

Result: n_block_min=0, n_block_max=1

For m_block = 1 (Q[128:256]):
  m_idx_max = 256
  n_idx = 256
  n_block_max = ceil(256/128) = 2
  
  Result: n_block_min=0, n_block_max=2
  # Q[128:256] attends to K[0:256]

Visual:
  K blocks:|'||  0|   |   1||   |
  Q block 0 (Q[0:128]):   [0~~~~~~~~~~~~~~~~|   |   |   |   ]
                           ||   ||   |
  Q block 1 (Q[128:256]): [0~~~~~~~~~~~~~~~~~] + [128~~~~~~~~~~]
```

##3.4 Visual Example: Different Q/K Lengths

```
seqlen_q = 1, seqlen_k = 1000( decode step)
tile_m = 128, tile_n = 128
is_causal = True
m_block = 0 (Q indices [0:1)only Q[0])

shift = seqlen_k - seqlen_q = 999

# Q[0] attends to K[0:0+1+999] = K[0:1000]
# All K blocks are valid!

m_idx_max = 1
n_idx = 1 + 999 = 1000
n_block_max = min(ceil(1000/128), ceil(1000/128)) = ceil(1000/128) = 8

Result: n_block_min=0, n_block_max=8

Visual:
  Q\K:  [0~~~~~~~K[1000]~~~~~~~]
  Q[0]: [||||||||||||||||||||]  (all K blocks valid)
```

---

# Part 4: get_m_block_min_max() - Finding Valid Q Blocks

##4.1 Method Signature

```python
def get_m_block_min_max(
    self,
    seqlen_info: SeqlenInfoQK,
    n_block: Int32,
) -> Tuple[Int32, Int32]:
    """Return (m_block_min, m_block_max) for this K block.
    
    For K block n_block, which Q blocks [m_block_min, m_block_max) are valid?
    """...
```

**Use case:** When iterating over K blocks instead of Q blocks (for some optimizations).

## 4.2 Algorithm

```python
def get_m_block_min_max(self, seqlen_info, n_block):
    # Start with full Q range
    m_block_max = ceil_div(seqlen_info.seqlen_q, self.tile_m)
    m_block_min = 0
    
    # Apply causal/local lower bound on Q
    if self.is_causal or (self.is_local and self.window_size_right is not None):
        # K[j] can be attended by Q[i] where i >= j - shift
        n_idx_min = n_block * self.tile_n
        
        m_idx = n_idx_min + seqlen_info.seqlen_q - seqlen_info.seqlen_k
        m_idx_right = m_idx if self.is_causal else m_idx - self.window_size_right
        
        m_block_min = max(m_block_min, m_idx_right // self.tile_m)
    
    # Apply local upper bound on Q
    if self.is_local and self.window_size_left is not None:
        n_idx_max = (n_block + 1) * self.tile_n
        
        m_idx = n_idx_max + seqlen_info.seqlen_q - seqlen_info.seqlen_k
        m_idx_left = m_idx + self.window_size_left
        
        m_block_max = min(m_block_max, ceil_div(m_idx_left, self.tile_m))
    
    return m_block_min, m_block_max
```

## 4.3 Example

```
seqlen_q = 256, seqlen_k = 256
tile_m = 128, tile_n = 128
is_causal = True
n_block = 0 (K indices [0:128])

# K[0:128] can be attended by Q[i] where i >= 0
# All Q blocks valid!

Result: m_block_min=0, m_block_max=2

n_block = 1 (K indices [128:256]):
  
  # K[j] where j >= 128
  # Can be attended by Q[i] where i >= 128
  # Shift = 0, so m_idx_right = 128
  
  m_block_min = 128 // 128 = 1
  
  Result: m_block_min=1, m_block_max=2
  # Only Q[128:256] can attend to K[128:256]
```

---

# Part 5: GQA/MQA Handling

##5.1 The Problem

In Multi-Query Attention (MQA) and Grouped Query Attention (GQA):
- Multiple Q heads share one KV head
- Q positions must be mapped to KV positions before computing masks

```
Example: 8 Q heads,1 KV head (MQA)
qhead_per_kvhead_packgqa = 8

Q tensor shape: (seqlen, num_q_heads=8, head_dim)
KV tensor shape: (seqlen, num_kv_heads=1, head_dim)

For Q position q:
  KV position = q // qhead_per_kvhead_packgqa
  
Q position 0→ KV position 0//8 = 0
Q position 7 → KV position 7//8 = 0
Q position 8 → KV position 8//8 = 1
```

## 5.2 Implementation

```python
# In get_n_block_min_max:
m_idx_max = (m_block + 1) * self.tile_m

if self.qhead_per_kvhead_packgqa > 1:
    # Convert Q position to KV position
    # Q positions [0:128] with8 Q heads per KV head
    # Map to KV position ceil(128/8) = 16
    m_idx_max = ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)

n_idx = m_idx_max + seqlen_k - seqlen_q
# Now n_idx is in KVposition space
```

## 5.3 Visual Example

```
seqlen_q = 128, seqlen_k = 16 (after GQA packing)
qhead_per_kvhead_packgqa = 8
tile_m = 128, tile_n = 128
is_causal = True
m_block = 0

# Without GQA adjustment:
m_idx_max = 128
n_idx = 128 + 16 - 128 = 16
n_block_max = ceil(16/128) = 1

# With GQA adjustment:
m_idx_max = ceil(128/8) = 16  # Conver to KV position
n_idx = 16 + 16 - 16 = 16
n_block_max = ceil(16/128) = 1

# But the correct answer is n_block_max = 1
# All K blocks are valid since seqlen_k = 16 < tile_n = 128
```

---

# Part 6: Split-KV Mode

##6.1 The Problem

For very long KV sequences, we split KV across multiple CTAs:

```
seqlen_k = 10000, num_splits = 4
Each split processes seqlen_k / 4 = 2500 tokens

Split 0: K[0:2500]
Split 1: K[2500:5000]
Split 2: K[5000:7500]
Split 3: K[7500:10000]
```

Each CTA processes one split, then results are combined.

## 6.2 Implementation

```python
if self.is_split_kv:
    # Compute blocks per split
    num_n_blocks_per_split = (n_block_max - n_block_min + num_splits - 1) // num_splits
    
    # Adjust for this split
    n_block_min = n_block_min + split_idx * num_n_blocks_per_split
    n_block_max = min(n_block_min + num_n_blocks_per_split, n_block_max)
```

##6.3 Example

```
seqlen_k =2560, tile_n = 128
num_n_blocks = ceil(2560/128) = 20
num_splits = 4

Without split: n_block_min=0, n_block_max=20

With split_kv:  num_n_blocks_per_split = (20 + 3) // 4 = 5

split_idx=0: n_block_min=0, n_block_max=5   # K[0:640]
split_idx=1: n_block_min=5, n_block_max=10  # K[640:1280]
split_idx=2: n_block_min=10, n_block_max=15 # K[1280:1920]
split_idx=3: n_block_min=15, n_block_max=20 # K[1920:2560]
```

---

# Part 7: Append-KV (SeqlenInfoQKNewK)

## 7.1 Use Case

For incremental decode, we have:
- Original KV cache: seqlen_k_og tokens
- New K/V to append: seqlen_k_new tokens

Need to compute which NEW K blocks a Q block needs.

## 7.2 get_n_block_k_new_min_max()

```python
def get_n_block_k_new_min_max(self, seqlen_info, m_block, split_idx=0, num_splits=1):
    # First get full K block range
    n_block_min, n_block_max = self.get_n_block_min_max(
        seqlen_info, m_block, split_idx, num_splits
    )
    
    # Map to NEW K index space (subtract original K length)
    idx_k_new_min = max(n_block_min * tile_n - seqlen_k_og, 0)
    idx_k_new_max = min(n_block_max * tile_n - seqlen_k_og, seqlen_k_new)
    
    # Convert back to block indices
    n_block_new_min = idx_k_new_min // tile_n
    n_block_new_max = ceil_div(idx_k_new_max, tile_n)
    
    return n_block_new_min, n_block_new_max
```

## 7.3 Example

```
seqlen_k_og = 1000 (original KV cache)
seqlen_k_new = 10 (new tokens to append)
seqlen_q = 1 (decode, single Q token)
tile_n = 128
is_causal = True

# Q[0] attends to K[0:1010]
# n_block_min=0, n_block_max=ceil(1010/128)=8

# Map to new K space:
# Original K occupies blocks [0: ceil(1000/128)] = [0:8)
# New K occupies blocks [0: ceil(10/128)] = [0:1)

idx_k_new_min = max(0* 128 - 1000, 0) = 0
idx_k_new_max = min(8 * 128 - 1000, 10) = min(24, 10) = 10

n_block_new_min = 0// 128 = 0
n_block_new_max = ceil(10/128) = 1

Result: Load NEW K block [0:1) for append
```

---

# Part 8: Complete Usage Example

```python
# Setup
block_info = BlockInfo(
    tile_m=128,
    tile_n=128,
    is_causal=True,
    is_local=False,
    qhead_per_kvhead_packgqa=1,
)

seqlen_info = SeqlenInfoQK.create(
    batch_idx=0,
    seqlen_q_static=256,
    seqlen_k_static=256,
    mCuSeqlensQ=None,
    mCuSeqlensK=None,
)
# seqlen_q = 256, seqlen_k = 256

# In kernel:
for m_block in range(ceil_div(seqlen_q, tile_m)):
    # Load Q tile
    sQ = load_Q_tile(mQ, m_block * tile_m, (m_block+1) * tile_m)
    
    # Get valid K blocks for this Q block
    n_block_min, n_block_max = block_info.get_n_block_min_max(
        seqlen_info, m_block
    )
    
    # Only iterate valid K blocks!
    for n_block in range(n_block_min, n_block_max):
        # Load K tile
        sK = load_K_tile(mK, n_block * tile_n, (n_block+1) * tile_n)
        
        # Compute attention
        sS = sQ @ sK.T
        sP = softmax(sS / sqrt(d))
        sO = sP @ sV
        
        # Accumulate output
        accumulate(sO, sAccum)
```

---

# Summary

## BlockInfo Responsibilities

| Responsibility | Method |
|----------------|--------|
| Which K blocks to load for Q block | `get_n_block_min_max()` |
| Which Q blocks can use K block | `get_m_block_min_max()` |
| Which NEW K blocks for append-KV | `get_n_block_k_new_min_max()` |
| Causal mask boundary | `get_n_block_min_causal_local_mask()` |
| Local mask boundary | `get_n_block_min_before_local_mask()` |

## Key Formulas

```python
# Causal: Q[i] attends to K[j] where j <= i + shift
# shift = seqlen_k - seqlen_q

# Local: Q[i] attends to K[j] where
#        i - window_left + shift <= j <= i + window_right + shift

# GQA: Q position q → KV position q // qhead_per_kvhead_packgqa
```

## Performance Impact

**Without block-level masking:**
- Load all K tiles, compute all Q-K matmuls
- Zero out invalid positions
- Wasted memory bandwidth and computation

**With block-level masking:**
- Load only valid K tiles
- Skip invalid Q-K pairs entirely
- Significant speedup for causal/local attention (30-50% for autoregressive models)