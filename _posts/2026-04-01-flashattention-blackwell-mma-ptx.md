---
layout: article
title: "[FlashAttention] Blackwell MMA: PTX Inline Assembly for GEMM"
date: 2026-04-01
---

In FlashAttention for Blackwell (SM100), the MMA (Matrix Multiply-Accumulate) operations use **PTX inline assembly** instead of CuTe's high-level `cute.gemm()`. This article explains the `blackwell_helpers.py` functions and how they're used in `flash_fwd_sm100_simple_18.py`.

## TL;DR

- **Blackwell MMA** uses `tcgen05.mma` PTX instruction with **smem descriptors**
- **smem descriptor** = 64-bit address (lo + hi) encoding shared memory location
- **idesc** = 32-bit instruction descriptor encoding MMA shape and dtype
- **Two paths**: SMEM→SMEM→TMEM (Q×K) and TMEM→SMEM→TMEM (P×V)
- **FlashAttention helpers** wrap PTX inline asm with descriptor computation
- **Key pattern**: `declare_ptx_smem_desc` → `declare_ptx_idesc` → `gemm_ptx_precomputed_varname`

---

# Part 1: FlashAttention Blackwell Helpers

## 1.1 Overview

`flash_attn/cute/blackwell_helpers.py` provides **PTX-level GEMM wrappers** for Blackwell. All functions emit `tcgen05.mma` via `llvm.inline_asm`.

### Function Categories

| Category | Functions | Purpose |
|----------|-----------|---------|
| **High-level** | `gemm`, `gemm_w_idx` | Use `cute.gemm()` (CuTe abstraction) |
| **PTX full** | `gemm_ptx`, `gemm_ptx_loop` | Compute descriptors inside, emit PTX |
| **PTX partial** | `gemm_ptx_partial`, `gemm_ptx_partial1` | Support split_arrive for pipelining |
| **PTX precomputed** | `gemm_ptx_precomputed`, `gemm_ptx_precomputed_varname` | Use pre-declared PTX registers |
| **PTX declare** | `declare_ptx_smem_desc`, `declare_ptx_idesc` | Declare PTX registers for descriptors |

### Function Dependency Graph

```
gemm_w_idx ──→ cute.gemm()                    # High-level, no PTX

gemm_ptx_w_idx ──→ gemm_ptx_partial()          # PTX partial

gemm ──→ cute.gemm()                           # High-level, no PTX

gemm_ptx ──→ llvm.inline_asm()                 # Full PTX, computes descriptors

gemm_ptx_loop ──→ llvm.inline_asm()            # Full PTX with precomputed offsets

gemm_ptx_partial ──→ llvm.inline_asm()         # Partial PTX, supports split_arrive

gemm_ptx_partial1 ──→ llvm.inline_asm()        # Variant with precomputed base+stage

gemm_ptx_precomputed ──→ llvm.inline_asm()     # Precomputed descriptors

gemm_ptx_precomputed_varname ──→ llvm.inline_asm()  # Uses pre-declared PTX vars
     ↑
     └── declare_ptx_smem_desc()  # Declares smem_desc_<N> registers
     └── declare_ptx_idesc()      # Declares idesc register

i64_to_i32x2 ──→ utility (splits 64-bit to 2×32-bit)
```

**Key dependencies:**
- `gemm_ptx_w_idx` calls `gemm_ptx_partial`
- `gemm_ptx_precomputed_varname` requires `declare_ptx_smem_desc` + `declare_ptx_idesc` to be called first
- All PTX functions call `sm100_desc.mma_op_to_idesc()` and `sm100_desc.make_smem_desc_base()`

---

## 1.2 Core Concepts

### smem Descriptor

A **64-bit shared memory descriptor** for `tcgen05.mma`:

```
smem_desc = {lo: 32-bit, hi: 32-bit}
  lo = smem_desc_base_lo | smem_desc_start_addr
  hi = smem_desc_base_hi

smem_desc_base = make_smem_desc_base(layout, swizzle, major)
smem_desc_start_addr = make_smem_desc_start_addr(iterator)
```

### idesc (Instruction Descriptor)

A **32-bit value** encoding MMA shape and data type:

```python
idesc = sm100_desc.mma_op_to_idesc(mma_op)
# Encodes: shape (M, N, K), dtype (A, B, C), operand source (SMEM/TMEM)
```

### Two MMA Paths

| Path | A Source | B Source | C Source | Use Case |
|------|----------|----------|----------|----------|
| **SMEM→SMEM→TMEM** | SMEM (Q) | SMEM (K) | TMEM (S) | Q×K→S |
| **TMEM→SMEM→TMEM** | TMEM (P) | SMEM (V) | TMEM (O) | P×V→O |

---

## 1.3 Function Details

### gemm_w_idx (High-level)

```python
def gemm_w_idx(tiled_mma, acc, tCrA, tCrB, A_idx=None, B_idx=None, zero_init=False, swap_AB=False):
    """High-level GEMM using cute.gemm().
    
    Uses CuTe's gemm abstraction. No PTX inline asm.
    A_idx/B_idx index into the stage dimension.
    """
```

**When to use:** When you don't need PTX-level control.

### gemm (High-level)

```python
def gemm(tiled_mma, acc, tCrA, tCrB, zero_init=False):
    """Simple high-level GEMM using cute.gemm().
    
    Iterates over K dimension with loop.
    """
```

### gemm_ptx_w_idx (PTX Partial Wrapper)

```python
def gemm_ptx_w_idx(tiled_mma, acc, tCrA, tCrB, sA, sB, A_idx=None, B_idx=None, 
                   zero_init=False, cta_group=1, **kwargs):
    """PTX-based GEMM, delegates to gemm_ptx_partial.
    
    Converts tiled_mma to mma_atom, then calls gemm_ptx_partial.
    """
```

**Dependency:** Calls `gemm_ptx_partial`.

### gemm_ptx (Full PTX)

```python
def gemm_ptx(op, acc, tCrA, tCrB, sA, sB, zero_init=False):
    """Full PTX GEMM with inline asm.
    
    Computes smem descriptors inside the function.
    Emits one tcgen05.mma per K tile.
    """
```

**Key features:**
- Computes `smem_desc_base` from layout
- Computes `smem_desc_start_addr` from iterator
- Emits `tcgen05.mma.cta_group::1.kind::f16` per K iteration

### gemm_ptx_loop (Full PTX with Precomputed Offsets)

```python
def gemm_ptx_loop(op, acc, tCrA, tCrB, sA, sB, zero_init=False):
    """PTX GEMM with precomputed offset differences.
    
    Precomputes offset_a_diff and offset_b_diff for efficient loop.
    Uses add.u32 instead of full descriptor recomputation.
    """
```

**Optimization:** Precomputes `offset_diff[k] = offset[k] - offset[k-1]` for loop efficiency.

### gemm_ptx_partial (Partial PTX)

```python
def gemm_ptx_partial(op, acc_tmem_addr, tCrA, tCrB, sA, sB, mbar_ptr=None, 
                     mbar_phase=None, split_arrive=None, zero_init=False, cta_group=1):
    """Partial PTX GEMM with split_arrive support.
    
    split_arrive: Number of K tiles to process before mbarrier wait.
    Used for pipelining: process 3/4 of K tiles, wait for softmax, process 1/4.
    """
```

**Why P×V uses `gemm_ptx_partial`:**

P×V is **TMEM→SMEM→TMEM**:
- A = P (in TMEM, from softmax output)
- B = V (in SMEM, from global memory)
- C = O (in TMEM, accumulator)

The key feature is **`split_arrive`**: splits the K loop into two parts with a mbarrier wait in between.

**Why split_arrive?**

```
Without split_arrive:
  Softmax computes P[0:128] → wait → MMA does P×V[0:128]
  Softmax computes P[128:256] → wait → MMA does P×V[128:256]
  ...serial, no overlap

With split_arrive:
  Softmax computes P[0:96]  → MMA does P×V[0:96]  (overlap!)
  Softmax computes P[96:128] → wait for mbarrier
  MMA does P×V[96:128]
```

**Input-to-PTX mapping for TS (Tensor Memory Source) path:**

```python
# Given inputs for P×V:
op                  = pv_mma_op          # MMA operation
acc_tmem_addr       = 0x1000             # $3: TMEM address for O
tCrA                = P register frag    # A is in TMEM
tCrB                = V register frag    # B is from SMEM
sA                  = None               # A is in TMEM, not SMEM
sB                  = V in SMEM          # B is in SMEM
mbar_ptr            = 0x5000             # $4: mbarrier address
mbar_phase          = 1                  # $5: mbarrier phase
split_arrive        = 96                 # Split at 96 elements (3/4 of 128)
zero_init           = False
cta_group           = 2
```

**Generated PTX:**
```ptx
{
.reg .pred leader_thread;
.reg .pred p;
.reg .b32 idesc;
.reg .b32 tmem_acc;
.reg .b32 tmem_a;
.reg .b32 smem_desc_b_lo_start;
.reg .b32 smem_desc_b_lo;
.reg .b32 smem_desc_b_hi;
.reg .b64 smem_desc_b;

elect.sync _|leader_thread, -1;
mov.b32 idesc, 0x...;              # hardcoded from op
mov.b32 tmem_acc, $3;              # $3 = acc_tmem_addr = 0x1000
mov.b32 tmem_a, $0;                # $0 = tA_addr (TMEM address for P)
mov.b32 smem_desc_b_lo_start, $1;  # $1 = smem_desc_start_b_lo
mov.b32 smem_desc_b_hi, 0x...;     # hardcoded from smem_desc_base_b
mov.b64 smem_desc_b, {smem_desc_b_lo_start, smem_desc_b_hi};
setp.ne.b32 p, $2, 0;              # $2 = not zero_init

# Part 1: Process first split_arrive_idx K tiles (no mbarrier wait)
@leader_thread tcgen05.mma.cta_group::2.kind::f16 
    [tmem_acc], [tmem_a], smem_desc_b, idesc, p;

# k=1..split_arrive_idx-1:
@leader_thread tcgen05.mma.cta_group::2.kind::f16 
    [tmem_acc], [tmem_a + 0x20], smem_desc_b, idesc, 1;
# ... more iterations

# Mbarrier wait: wait for softmax to finish computing rest of P
.reg .pred P1; 
LAB_WAIT: 
mbarrier.try_wait.parity.shared::cta.b64 P1, [$4], $5, 10000000; 
@P1 bra DONE; 
bra     LAB_WAIT; 
DONE: 

# Part 2: Process remaining K tiles (after mbarrier)
@leader_thread tcgen05.mma.cta_group::2.kind::f16 
    [tmem_acc], [tmem_a + 0x60], smem_desc_b, idesc, 1;
# ... more iterations
}
```

**Input-to-PTX mapping table:**

| Input | PTX Usage | Example |
|-------|-----------|---------|
| `tA_addr` | `$0` → `mov.b32 tmem_a, $0` | TMEM address for P |
| `smem_desc_start_b` | `$1` → `mov.b32 smem_desc_b_lo_start, $1` | V start address |
| `not zero_init` | `$2` → `setp.ne.b32 p, $2, 0` | predicate |
| `acc_tmem_addr` | `$3` → `mov.b32 tmem_acc, $3` | TMEM address for O |
| `mbar_ptr` | `$4` → mbarrier wait | mbarrier address |
| `mbar_phase` | `$5` → mbarrier wait | phase bit |
| `op` | hardcoded `idesc` | MMA operation |
| `smem_desc_base_b` | hardcoded `smem_desc_b_hi` | V base descriptor |
| `tCrA_layout` | used to compute `offset_a[k]` | P offsets |
| `tCrB_layout` | used to compute `offset_b[k]` | V offsets |
| `split_arrive` | splits K loop at `split_arrive_idx` | 96 |
| `cta_group` | hardcoded in `tcgen05.mma.cta_group::2` | 2 |

**Key insight:** `split_arrive` enables pipelining by splitting the K loop:
1. Process first 3/4 of K tiles (while softmax computes rest of P)
2. Wait for mbarrier (softmax signals P is ready)
3. Process remaining 1/4 of K tiles

### gemm_ptx_partial1 (Variant)

```python
def gemm_ptx_partial1(op, acc_tmem_addr, tCrA, tCrB, sA_base_addr_for_desc, 
                      sA_addr_offset_for_desc, sA_stage, sB_base_addr_for_desc,
                      sB_addr_offset_for_desc, sB_stage, sA_layout, sB_layout,
                      sA_swizzle, sB_swizzle, zero_init=False):
    """Variant with precomputed base addresses and stage indices.
    
    Uses mad.lo.u32 to compute descriptor: base + stage * offset.
    """
```

### gemm_ptx_precomputed (Precomputed Descriptors)

```python
def gemm_ptx_precomputed(acc_tmem_addr, smem_desc_start_a, smem_desc_start_b, idesc,
                         smem_desc_base_a, smem_desc_base_b, tCrA_layout, tCrB_layout,
                         mbar_ptr=None, mbar_phase=None, zero_init=False, cta_group=1):
    """PTX GEMM with precomputed descriptors.
    
    Descriptors are passed as arguments instead of computed inside.
    Supports mbarrier wait for pipelining.
    """
```

### gemm_ptx_precomputed_varname (Pre-declared Variables)

```python
def gemm_ptx_precomputed_varname(acc_tmem_addr, smem_desc_start_b, smem_desc_base_b,
                                 tCrB_layout, smem_var_name_prefix, idesc_var_name,
                                 smem_offset, zero_init=False, cta_group=1):
    """PTX GEMM using pre-declared PTX register variables.
    
    Requires declare_ptx_smem_desc and declare_ptx_idesc to be called first.
    Uses PTX variables: {smem_var_name_prefix}_<k> and {idesc_var_name}.
    """
```

**What's precomputed vs computed inside:**

| Descriptor | Where computed | When computed | Why |
|------------|---------------|---------------|-----|
| **A (Q)** | `declare_ptx_smem_desc` (before) | Once at kernel entry | A has only 2 stages, address changes by fixed stride |
| **B (K)** | Inside this function | Every GEMM call | B changes every K block iteration (circular buffer) |

**Why precompute A but not B?**

- **A (Q)**: 2 stages with fixed stride between them. Compute once, then just `add.s32` with stride.
- **B (K)**: Circular buffer with `kv_stage=6` stages. Address changes every iteration anyway.

**The optimization:**
```python
# Without precompute: Compute A descriptor from scratch for each stage
smem_desc_base_a = smem_desc_base_from_tensor(sQ[stage])  # Every call
smem_desc_start_a = make_smem_desc_start_addr(sQ[stage])  # Every call

# With precompute: Compute A descriptor once, adjust with stride
# declare_ptx_smem_desc(sQ[stage=1])  # Once at kernel entry
# Inside gemm_ptx_precomputed_varname:
mov.b64 {smem_desc_a_lo, smem_desc_a_hi}, fa_fwd_q_smem_desc_0;  # LOAD precomputed
add.s32 smem_desc_a_lo, smem_desc_a_lo, -16;                     # Just adjust stride
mov.b64 fa_fwd_q_smem_desc_0, {smem_desc_a_lo, smem_desc_a_hi};  # STORE back
```

**Input-to-PTX mapping:**

```python
# Given inputs:
acc_tmem_addr       = 0x1000        # $2 in PTX (TMEM address)
smem_desc_start_b   = 0x2000        # $0 in PTX (B start address)
smem_desc_base_b    = 0xBEEF_DEAD_0000_5678  # 64-bit B base
tCrB_layout         # shape[2] = 4 → num_k_tile = 4
smem_var_name_prefix= "fa_fwd_q_smem_desc"
idesc_var_name      = "fa_fwd_qk_mma_idesc"
smem_offset         = -16           # hardcoded offset for stage 0
zero_init           = True          # $1 in PTX
cta_group           = 2             # hardcoded in instruction

# Step 1: Split B base
smem_desc_base_b_lo = 0x0000_5678
smem_desc_b_hi      = 0xBEEF_DEAD

# Step 2: Compute B offsets from layout
offset_b = [crd2idx((0,0,k), tCrB_layout) for k in range(4)]
# = [0, 32, 64, 96]

# Step 3: Combine B start + base_lo
smem_desc_start_b_lo = 0x0000_5678 | 0x2000 = 0x0000_7678
```

**Generated PTX:**
```ptx
{
.reg .pred leader_thread;
.reg .pred p;
.reg .b32 tmem_acc;
.reg .b32 smem_desc_b_lo_start;
.reg .b32 smem_desc_a_lo, smem_desc_b_lo;
.reg .b32 smem_desc_a_hi, smem_desc_b_hi;
.reg .b64 smem_desc_b_<4>;

elect.sync _|leader_thread, -1;
mov.b32 tmem_acc, $2;                    # $2 = acc_tmem_addr
mov.b32 smem_desc_b_lo_start, $0;        # $0 = smem_desc_start_b_lo
mov.b32 smem_desc_b_hi, 0xBEEFDEAD;      # hardcoded

# k=0: Load A descriptor, apply offset, store back
mov.b64 {smem_desc_a_lo, smem_desc_a_hi}, fa_fwd_q_smem_desc_0;
add.s32 smem_desc_a_lo, smem_desc_a_lo, -16;    # smem_offset
mov.b64 fa_fwd_q_smem_desc_0, {smem_desc_a_lo, smem_desc_a_hi};
mov.b64 smem_desc_b_0, {smem_desc_b_lo_start, smem_desc_b_hi};

# k=1..3: Same pattern with offset_b[k]
mov.b64 {smem_desc_a_lo, smem_desc_a_hi}, fa_fwd_q_smem_desc_1;
add.s32 smem_desc_a_lo, smem_desc_a_lo, -16;
add.s32 smem_desc_b_lo, smem_desc_b_lo_start, 0x20;  # offset_b[1]
mov.b64 fa_fwd_q_smem_desc_1, {smem_desc_a_lo, smem_desc_a_hi};
mov.b64 smem_desc_b_1, {smem_desc_b_lo, smem_desc_b_hi};
# ... k=2, k=3 same pattern

setp.ne.b32 p, $1, 0;                    # $1 = not zero_init

# Execute MMA for each K tile
@leader_thread tcgen05.mma.cta_group::2.kind::f16 
    [tmem_acc], fa_fwd_q_smem_desc_0, smem_desc_b_0, fa_fwd_qk_mma_idesc, 0;
@leader_thread tcgen05.mma.cta_group::2.kind::f16 
    [tmem_acc], fa_fwd_q_smem_desc_1, smem_desc_b_1, fa_fwd_qk_mma_idesc, 1;
# ... k=2, k=3
}
```

**Input-to-PTX mapping table:**

| Input | PTX Usage | Example |
|-------|-----------|---------|
| `acc_tmem_addr` | `$2` → `mov.b32 tmem_acc, $2` | `0x1000` |
| `smem_desc_start_b` | `$0` → `mov.b32 smem_desc_b_lo_start, $0` | `0x2000` |
| `smem_desc_base_b` (lo) | hardcoded in `mov.b64 smem_desc_b_0` | `0x00005678` |
| `smem_desc_base_b` (hi) | hardcoded in `mov.b32 smem_desc_b_hi` | `0xBEEFDEAD` |
| `tCrB_layout` | used to compute `offset_b[k]` | `[0, 32, 64, 96]` |
| `smem_var_name_prefix` | PTX variable names for A | `fa_fwd_q_smem_desc_<k>` |
| `idesc_var_name` | hardcoded in `tcgen05.mma` | `fa_fwd_qk_mma_idesc` |
| `smem_offset` | hardcoded `add.s32` | `-16` |
| `zero_init` | `$1` → `setp.ne.b32 p, $1, 0` | `True` → `$1=0` |
| `cta_group` | hardcoded in `tcgen05.mma.cta_group::2` | `2` |

**Key insight:** Only 3 values are runtime inputs (`$0`, `$1`, `$2`). Everything else (base addresses, offsets, variable names) is hardcoded at compile time.

### declare_ptx_smem_desc (Declare Registers)

```python
def declare_ptx_smem_desc(smem_desc_start_a, smem_desc_base_a, tCrA_layout, var_name_prefix):
    """Declare PTX registers for smem descriptors.
    
    Inputs:
    - smem_desc_start_a: 32-bit start address offset (from iterator)
    - smem_desc_base_a:  64-bit base descriptor (from layout + swizzle + major)
    - tCrA_layout:       Layout to compute per-K-tile offsets
    - var_name_prefix:   PTX variable name prefix
    """
```

**How each input maps to generated PTX:**

```python
# Given inputs:
smem_desc_start_a = 0x1234        # $0 in inline asm (32-bit)
smem_desc_base_a  = 0xDEAD_BEEF_CAFE_0000  # 64-bit, split to lo/hi
tCrA_layout       # Used to compute offset_a[k] for each K tile
var_name_prefix   = "fa_fwd_q_smem_desc"

# Step 1: Split base into lo/hi
smem_desc_base_a_lo = 0xCAFE_0000   # lower 32 bits
smem_desc_a_hi      = 0xDEAD_BEEF   # upper 32 bits

# Step 2: Combine start + base_lo
smem_desc_start_a_lo = smem_desc_base_a_lo | smem_desc_start_a
# = 0xCAFE_0000 | 0x1234 = 0xCAFE_1234

# Step 3: Compute offsets from layout
offset_a = [crd2idx((0,0,k), tCrA_layout) for k in range(num_k_tile)]
# = [0, 16, 32, 48, ...]  # depends on layout stride

# Generated PTX:
.reg .b32 fa_fwd_q_smem_desc_lo;              # temp register
.reg .b64 fa_fwd_q_smem_desc_<4>;             # array of 4 descriptors

# k=0: combine $0 (start_lo) with base_hi
mov.b64 fa_fwd_q_smem_desc_0, {$0, 0xDEADBEEF};
# fa_fwd_q_smem_desc_0 = {0xCAFE1234, 0xDEADBEEF}

# k=1: add offset[1] to start, combine with base_hi
add.s32 fa_fwd_q_smem_desc_lo, $0, 0x10;      # offset[1] = 16 = 0x10
mov.b64 fa_fwd_q_smem_desc_1, {fa_fwd_q_smem_desc_lo, 0xDEADBEEF};
# fa_fwd_q_smem_desc_1 = {0xCAFE1244, 0xDEADBEEF}

# k=2: add offset[2] to start, combine with base_hi
add.s32 fa_fwd_q_smem_desc_lo, $0, 0x20;      # offset[2] = 32 = 0x20
mov.b64 fa_fwd_q_smem_desc_2, {fa_fwd_q_smem_desc_lo, 0xDEADBEEF};
# fa_fwd_q_smem_desc_2 = {0xCAFE1254, 0xDEADBEEF}

# k=3: add offset[3] to start, combine with base_hi
add.s32 fa_fwd_q_smem_desc_lo, $0, 0x30;      # offset[3] = 48 = 0x30
mov.b64 fa_fwd_q_smem_desc_3, {fa_fwd_q_smem_desc_lo, 0xDEADBEEF};
# fa_fwd_q_smem_desc_3 = {0xCAFE1264, 0xDEADBEEF}
```

**Input-to-PTX mapping:**

| Input | PTX Usage | Example |
|-------|-----------|---------|
| `smem_desc_start_a` | `$0` (inline asm input) | `0x1234` |
| `smem_desc_base_a` (lo) | OR'd with `$0`, used in `mov.b64` | `0xCAFE0000` |
| `smem_desc_base_a` (hi) | Hardcoded in `mov.b64` | `0xDEADBEEF` |
| `tCrA_layout` | Used to compute `offset_a[k]` | `[0, 16, 32, 48]` |
| `var_name_prefix` | PTX variable names | `fa_fwd_q_smem_desc_<k>` |

**Important:** Only generates PTX when `smem_desc_base_a is not None` (i.e., A is in SMEM, not TMEM).

### declare_ptx_idesc (Declare Register)

```python
def declare_ptx_idesc(op, var_name):
    """Declare PTX register for idesc.
    
    Emits:
      .reg .b32 {var_name};
      mov.b32 {var_name}, {idesc};
    """
```

### i64_to_i32x2 (Utility)

```python
def i64_to_i32x2(i):
    """Split 64-bit int into (lo, hi) 32-bit tuple."""
    return i & 0xFFFF_FFFF, (i >> 32) & 0xFFFF_FFFF
```

---

# Part 2: Usage in FlashAttention-4

## 2.1 Setup Phase

In `flash_fwd_sm100.py`, the MMA setup happens once at kernel entry:

```python
# Step 1: Get MMA operations from TiledMma
qk_mma_op, pv_mma_op = tiled_mma_qk.op, tiled_mma_pv.op

# Step 2: Convert ops to idesc (instruction descriptors)
qk_mma_idesc, pv_mma_idesc = (
    sm100_desc.mma_op_to_idesc(qk_mma_op),   # For Q×K→S
    sm100_desc.mma_op_to_idesc(pv_mma_op),   # For P×V→O
)

# Step 3: Compute smem descriptor bases
q_smem_base = sm100_desc.smem_desc_base_from_tensor(sQ, sm100_desc.Major.K)
k_smem_base = sm100_desc.smem_desc_base_from_tensor(sK, sm100_desc.Major.K)
v_smem_base = sm100_desc.smem_desc_base_from_tensor(sV, sm100_desc.Major.MN)

# Step 4: Compute smem descriptor start addresses for each Q stage
q_smem_start = [
    sm100_desc.make_smem_desc_start_addr(sQ[None, None, None, stage].iterator)
    for stage in range(self.q_stage)  # stage 0 and 1
]
```

**What each function does:**

| Function | Purpose | Returns |
|----------|---------|---------|
| `mma_op_to_idesc` | Encode MMA shape/dtype into 32-bit idesc | `int` (e.g., `0x12345678`) |
| `smem_desc_base_from_tensor` | Compute 64-bit base from tensor layout | `int` (64-bit) |
| `make_smem_desc_start_addr` | Compute start address from iterator | `int` (32-bit offset) |

## 2.2 PTX Register Declaration

```python
# Declare PTX registers for Q smem descriptors (one per K tile)
sm100_utils.declare_ptx_smem_desc(
    q_smem_start[self.q_stage - 1],  # Start address for last Q stage
    q_smem_base,                      # Base address for Q
    tSrQ[None, None, None, 0].layout, # Q layout for descriptor computation
    var_name_prefix="fa_fwd_q_smem_desc",
)

# Declare PTX registers for idesc
sm100_utils.declare_ptx_idesc(qk_mma_op, var_name="fa_fwd_qk_mma_idesc")
sm100_utils.declare_ptx_idesc(pv_mma_op, var_name="fa_fwd_pv_mma_idesc")
```

**What this emits (PTX):**
```ptx
.reg .b32 fa_fwd_q_smem_desc_lo;
.reg .b64 fa_fwd_q_smem_desc_<num_k_tiles>;
mov.b64 fa_fwd_q_smem_desc_0, {start_lo, base_hi};
// For each k tile:
add.s32 fa_fwd_q_smem_desc_lo, start_lo, offset[k];
mov.b64 fa_fwd_q_smem_desc_k, {fa_fwd_q_smem_desc_lo, base_hi};

.reg .b32 fa_fwd_qk_mma_idesc;
mov.b32 fa_fwd_qk_mma_idesc, 0x...;

.reg .b32 fa_fwd_pv_mma_idesc;
mov.b32 fa_fwd_pv_mma_idesc, 0x...;
```

## 2.3 GEMM Function Setup

```python
# Compute Q stage stride for descriptor offset
sQ_stage_stride = (sQ.layout.stride[-1] * sQ.element_type.width // 8) >> 4

# Setup GEMM functions for Q×K→S (one per Q stage)
gemm_Si = [
    partial(
        sm100_utils.gemm_ptx_precomputed_varname,
        self.tmem_s_offset[stage],           # TMEM address for S[stage]
        smem_desc_base_b=k_smem_base,        # K smem base
        tCrB_layout=tSrK[None, None, None, 0].layout,  # K layout
        smem_var_name_prefix="fa_fwd_q_smem_desc",     # Pre-declared Q descriptors
        idesc_var_name="fa_fwd_qk_mma_idesc",          # Pre-declared idesc
        smem_offset=-sQ_stage_stride if stage == 0 else sQ_stage_stride,  # Stage offset
        zero_init=True,                      # First K tile zeros accumulator
        cta_group=self.cta_group_size,       # 2-CTA instruction
    )
    for stage in range(self.q_stage)
]

# Setup GEMM functions for P×V→O (one per Q stage)
gemm_Pi = [
    partial(
        sm100_utils.gemm_ptx_partial,
        pv_mma_op,                           # MMA operation
        self.tmem_o_offset[stage],           # TMEM address for O[stage]
        tOrP[None, None, None, stage],       # P register fragment
        sA=None,                             # A is in TMEM (P), not SMEM
        split_arrive=self.split_P_arrive,    # Split P arrival for pipelining
        cta_group=self.cta_group_size,       # 2-CTA instruction
    )
    for stage in range(self.q_stage)
]
```

**What `partial` does:** Creates a callable with pre-filled arguments. When called later, only needs the remaining arguments.

## 2.4 GEMM Execution

### Q×K→S (gemm_Si)

```python
# Inside the MMA loop, for each K/V block:
for k_block in range(n_block_min, n_block_max):
    # Wait for Q and K to be ready
    pipeline_q.consumer_wait(q_consumer_state)
    pipeline_kv.consumer_wait(kv_consumer_state)
    
    # Execute Q×K→S for current Q stage
    stage = q_mma_phase
    gemm_Si[stage]()  # Calls gemm_ptx_precomputed_varname
    
    # Signal S is ready for softmax
    pipeline_s_p_o.producer_commit(P_full_O_rescaled_phase)
```

**What happens inside `gemm_Si[stage]()`:**

```ptx
// For each K tile:
@leader_thread tcgen05.mma.cta_group::2.kind::f16 
    [tmem_acc],                    // TMEM accumulator (S)
    fa_fwd_q_smem_desc_k,          // Q descriptor (pre-declared)
    smem_desc_b_k,                 // K descriptor (computed)
    fa_fwd_qk_mma_idesc,           // Instruction descriptor (pre-declared)
    p;                             // Predicate (accumulate or zero-init)
```

### P×V→O (gemm_Pi)

```python
# After softmax computes P:
for k_block in range(n_block_min, n_block_max):
    # Wait for V to be ready
    pipeline_kv.consumer_wait(kv_consumer_state)
    
    # Execute P×V→O for current Q stage
    stage = q_mma_phase
    gemm_Pi[stage]()  # Calls gemm_ptx_partial
    
    # Signal O is ready
    pipeline_o_acc.producer_commit(...)
```

**What happens inside `gemm_Pi[stage]()`:**

```ptx
// Split arrival: process first 3/4 of K tiles
@leader_thread tcgen05.mma.cta_group::2.kind::f16 
    [tmem_acc],                    // TMEM accumulator (O)
    [tmem_a + offset_a[k]],        // P descriptor (TMEM)
    smem_desc_b_k,                 // V descriptor (SMEM)
    idesc,                         // Instruction descriptor
    p;

// Wait for mbarrier (softmax finished computing rest of P)
mbarrier.try_wait.parity.shared::cta.b64 P1, [mbar_ptr], mbar_phase, 10000000;

// Process remaining 1/4 of K tiles
@leader_thread tcgen05.mma.cta_group::2.kind::f16 
    [tmem_acc],
    [tmem_a + offset_a[k]],
    smem_desc_b_k,
    idesc,
    1;
```

---

# Part 3: Function Comparison

## 3.1 Which Function to Use?

| Scenario | Recommended Function | Why |
|----------|---------------------|-----|
| **Q×K→S** (SMEM→SMEM→TMEM) | `gemm_ptx_precomputed_varname` | Pre-declared descriptors, no runtime computation |
| **P×V→O** (TMEM→SMEM→TMEM) | `gemm_ptx_partial` | Supports split_arrive for pipelining with softmax |
| **Simple GEMM** | `gemm` or `gemm_w_idx` | High-level, no PTX needed |
| **Custom descriptor computation** | `gemm_ptx_precomputed` | Pass descriptors as arguments |

## 3.2 Performance Hierarchy

```
Fastest (least runtime work):
  gemm_ptx_precomputed_varname  ← Pre-declared PTX registers
  gemm_ptx_precomputed          ← Precomputed descriptors
  gemm_ptx_loop                 ← Precomputed offset diffs
  gemm_ptx_partial              ← Partial with split_arrive
  gemm_ptx                      ← Full descriptor computation
  gemm_ptx_w_idx                ← Wrapper around partial
  gemm_w_idx / gemm             ← High-level cute.gemm
Slowest (most runtime work)
```

## 3.3 Key Differences

| Feature | `gemm_ptx_precomputed_varname` | `gemm_ptx_partial` |
|---------|-------------------------------|-------------------|
| **Descriptor computation** | Pre-declared PTX registers | Computed inside function |
| **A source** | SMEM (via pre-declared) | TMEM (for P×V) |
| **B source** | SMEM | SMEM |
| **split_arrive** | No | Yes |
| **mbarrier wait** | No | Yes |
| **Use case** | Q×K→S | P×V→O |

---

# Summary

## Key Concepts

| Concept | Description |
|---------|-------------|
| **smem descriptor** | 64-bit address (lo + hi) for shared memory |
| **idesc** | 32-bit instruction descriptor for MMA shape/dtype |
| **tcgen05.mma** | Blackwell PTX instruction for tensor core MMA |
| **split_arrive** | Split K loop for pipelining with softmax |
| **precomputed** | Descriptors computed at setup, not runtime |

## Usage Pattern

```python
# 1. Setup (once at kernel entry)
idesc = sm100_desc.mma_op_to_idesc(mma_op)
smem_base = sm100_desc.smem_desc_base_from_tensor(tensor, major)
smem_start = sm100_desc.make_smem_desc_start_addr(iterator)

# 2. Declare PTX registers
declare_ptx_smem_desc(smem_start, smem_base, layout, prefix)
declare_ptx_idesc(mma_op, var_name)

# 3. Create GEMM function
gemm_fn = partial(gemm_ptx_precomputed_varname, acc_addr, smem_start_b, 
                  smem_base_b, layout_b, prefix, var_name, offset, ...)

# 4. Execute GEMM (in loop)
gemm_fn()  # Emits tcgen05.mma per K tile
```

## FlashAttention vs CuTe Native

- **CuTe native**: Uses `cute.gemm()` with `TiledMma` abstraction
- **FlashAttention**: Uses PTX inline asm with precomputed descriptors
- **Why PTX**: More control over descriptor computation, split_arrive for pipelining, pre-declared registers for efficiency
