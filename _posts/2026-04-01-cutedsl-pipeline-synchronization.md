---
layout: article
title: "[CuTeDSL] Pipeline Abstraction for GPU Kernel Synchronization"
date: 2026-04-01
---

GPUs achieve high performance through **pipelining** - overlapping computation with memory access. This article explains the Pipeline abstraction in CuTe DSL, covering both CuTe native pipelines and FlashAttention's extensions.

## TL;DR

- **Pipeline** = Producer-consumer synchronization using mbarriers
- **Pattern**: Producer `acquire → write → commit`; Consumer `wait → read → release`
- **PipelineTmaAsync**: TMA is producer (async load), AsyncThread is consumer (compute)
- **All other pipelines**: Same pattern with different producer types
- **FlashAttention adds**: `elect_one` flags, `*_w_index_phase` methods, `PipelineStateSimple`
- **Memory barrier relationships**:

```
┌──────────┬───────────┬──────────┬────────┬───────────┐
│          │ Acquire(p)│ commit(p)│ wait(c)│ release(c)│
├──────────┼───────────┼──────────┼────────┼───────────┤
│ Full (p) │     -     │  arrive  │  wait  │     -     │
├──────────┼───────────┼──────────┼────────┼───────────┤
│ Empty (c)│   wait    │    -     │    -   │  arrive   │
└──────────┴───────────┴──────────┴────────┴───────────┘
```

---

# Part 1: CuTe Native Pipelines

## 1.1 The Problem: Producer-Consumer Synchronization

In GPU kernels, we often have:

```
Producer Thread(s)          Consumer Thread(s)
     |                            |
     V                            |
Load data to smem---------------→|
     |                           Wait
     |←-------- "READY" ---------
     |                           |
     |                         Compute
     |←------- "EMPTY" ----------
```

**Without proper synchronization:**
- Consumer might read before producer writes (garbage data)
- Producer might overwrite before consumer reads (data loss)
- Both need coordinated access to shared memory

## 1.2 The Solution: Pipeline with Barriers

A **Pipeline** implements a circular buffer of synchronization barriers:

```
Array of mbarriers as circular buffer:

     Advance Direction
   <------------------

        Producer   Consumer
            |         ^
            V         |
       +-----------------+
     --|X|X|W|D|D|D|D|R|X|<-.
    /  +-----------------+   \
    |                        |
    `------------------------'

Where:
- X: Empty buffer (initial state)
- W: Producer writing (waiting for buffer to be empty)
- D: Data ready (producer has written)
- R: Consumer reading (consuming data)
```

## 1.3 PipelineAsync - Base Class

**File:** `cutlass/python/CuTeDSL/cutlass/pipeline/sm90.py`

```python
@dataclass(frozen=True)
class PipelineAsync:
    """Generic producer-consumer pipeline with barrier synchronization.
    
    State transitions for one pipeline entry (mbarrier):
    
    +-----------+-----------+-----------+-----------+-----------+-----------+
    | Barrier   | State     | p.acquire | p.commit  | c.wait    | c.release |
    +===========+===========+===========+===========+===========+===========+
    | empty_bar | empty     | <Return>  | n/a       | n/a       | -         |
    +-----------+-----------+-----------+-----------+-----------+-----------+
    | empty_bar | wait      | <Block>   | n/a       | n/a       | -> empty  |
    +-----------+-----------+-----------+-----------+-----------+-----------+
    | full_bar  | wait      | n/a       | -> full   | <Block>   | n/a       |
    +-----------+-----------+-----------+-----------+-----------+-----------+
    | full_bar  | full      | n/a       | -         | <Return>  | n/a       |
    +-----------+-----------+-----------+-----------+-----------+-----------+
    """
    sync_object_full: SyncObject    # Barrier for "data ready"
    sync_object_empty: SyncObject   # Barrier for "buffer empty"
    num_stages: int                # Number of pipeline stages
    producer_mask: Optional[Int32] # Threads participating in producer
    consumer_mask: Optional[Int32] # Threads participating in consumer
```

### Key Methods

```python
# Producer side:
producer.acquire(state)     # Wait for buffer to be empty
producer.commit(state)# Signal buffer is full

# Consumer side:
consumer.wait(state)        # Wait for buffer to be full
consumer.release(state)     # Signal buffer is empty
```

## 1.4 Usage Example (CuTe Native)

```python
# From dense_gemm_persistent.py

# Create pipeline with num_stages buffers
ab_producer, ab_consumer = pipeline.PipelineAsync.create(
    num_stages=self.num_ab_stage,# e.g., 2 or 3
    producer_group=producer_warp,
    consumer_group=consumer_warp,
)

# Initialize state for producer and consumer
producer_state = pipeline.make_pipeline_state(
    pipeline.PipelineUserType.Producer, self.num_ab_stage
)
consumer_state = pipeline.make_pipeline_state(
    pipeline.PipelineUserType.Consumer, self.num_ab_stage
)

# Producer loop (e.g., TMA load warp)
for k_tile in range(k_tile_cnt):
    # Wait for buffer to be empty
    ab_producer.acquire(producer_state)
    
    # Get buffer index
    buffer_idx = producer_state.index()
    
    # Write data to shared memory
    cute.copy(tma_atom_A, gA_slice, sA[(None, buffer_idx)])
    cute.copy(tma_atom_B, gB_slice, sB[(None, buffer_idx)])
    
    # Signal buffer is full
    ab_producer.commit(producer_state)
    
    # Advance to next stage
    producer_state.advance()

# Consumer loop (e.g., MMA warps)
for k_tile in range(k_tile_cnt):
    # Wait for buffer to be full
    ab_consumer.wait(consumer_state)
    
    # Get buffer index
    buffer_idx = consumer_state.index()
    
    # Read data from shared memory
    cute.cp_async_wait(ab_buffer.barrier)  # Ensure TMA arrived
    # ... compute MMA ...
    
    # Signal buffer is empty
    ab_consumer.release(consumer_state)
    
    # Advance to next stage
    consumer_state.advance()
```

## 1.5 Pipeline Types

All pipelines follow the same producer-consumer pattern. The difference is only in who is the producer:

| Pipeline Type | Producer | Consumer | Barrier Trigger |
|---------------|----------|----------|-----------------|
| **PipelineAsync** | AsyncThread | AsyncThread | Software `barrier.arrive()` |
| **PipelineTmaAsync** | TMA | AsyncThread | Hardware (TMA arrives automatically) |
| **PipelineCpAsync** | cp_async | AsyncThread | Hardware (`cp.async.commit_group`) |
| **PipelineTmaUmma** | TMA | UMMA | Hardware (TMA arrives automatically) |
| **PipelineUmmaAsync** | UMMA | AsyncThread | Software `barrier.arrive()` |

**Key insight:** All types share `acquire/commit` (producer) and `wait/release` (consumer) interface.
- **TMA**: Hardware arrives automatically when load completes
- **cp_async**: Hardware arrives when `commit_group` is called
- **AsyncThread**: Software explicitly calls `arrive()`

```python
# Pattern is the same for all pipelines:
# Producer: acquire → write → commit
# Consumer: wait → read → release

# PipelineTmaAsync example (TMA is producer):
producer, consumer = PipelineTmaAsync.create(num_stages=2, ...)
producer.acquire(state)      # Wait for buffer empty
# TMA loads to smem (hardware arrives automatically)
producer.commit(state)         # Signal buffer full

consumer.wait(state)          # Wait for buffer full
# Read from smem
consumer.release(state)      # Signal buffer empty
```

### FlashAttention Enhancements

FlashAttention wraps CuTe native pipelines with additional flexibility:

| Feature | CuTe Native | FlashAttention |
|---------|-------------|----------------|
| Base classes | `PipelineAsync`, `PipelineTmaAsync`, etc. | Same, suffixed `*Og` |
| `*_w_index_phase` | No | Yes - pass index/phase separately |
| `elect_one_*` | No | Yes - one thread signals barrier |
| `syncwarp_before_*` | No | Yes - warp sync before elect_one |
| `PipelineStateSimple` | No | Yes - optimized single Int32 |
| `extra_tx_count` | No | Yes - additional TMA transactions |

## 1.6 Make Pipeline State

```python
def make_pipeline_state(type: PipelineUserType, stages: int):
    """Create initial state for producer or consumer.
    
    Args:
        type: Producer or Consumer
        stages: Number of pipeline stages
    
    Returns:
        PipelineState with initial index and phase
    """
    if type is PipelineUserType.Producer:
        return PipelineState(stages, Int32(stages))# Start at stages, phase = 1
    elif type is PipelineUserType.Consumer:
        return PipelineState(stages, Int32(0))    # Start at 0, phase = 0
```

**State contains:**
- `index`: Current buffer position (0 to stages-1)
- `phase`: Parity bit for barrier wait (alternates each iteration)

---

# Part 2: FlashAttention Pipelines

##2.1 Why FlashAttention Needs Custom Pipelines

FlashAttention extends CuTe pipelines with:

1. **`elect_one` optimization**: Only one thread signals barrier (saves synchronization)
2. **`syncwarp` coordination**: Warp-level sync before barrier operations
3. **`extra_tx_count`**: Extra transaction count for TMA loads
4. **`PipelineStateSimple`**: Optimized state for single-stage pipelines

## 2.2 PipelineStateSimple

**File:** `flash_attn/cute/pipeline.py`

```python
class PipelineStateSimple:
    """Optimized pipeline state using single Int32 for index + phase.
    
    Encoding: phase_index = index + phase * stages
    - index = phase_index % stages
    - phase = phase_index // stages
    """
    def __init__(self, stages: int, phase_index: Int32):
        self._stages = stages
        self._phase_index = phase_index
    
    @property
    def index(self) -> Int32:
        if self._stages == 1:
            return Int32(0)# Optimized: no modulo needed
        else:
            return self._phase_index % self._stages
    
    @property
    def phase(self) -> Int32:
        if self._stages == 1:
            return self._phase_index  # Optimized: no division needed
        else:
            return self._phase_index // self._stages
    
    def advance(self):
        if self._stages == 1:# Single stage: flip phase bit
            self._phase_index ^= 1
        else:
            self._phase_index += 1
```

**Works for any number of stages:**
- `stages == 1`: Phase alternates 0→1→0→1 (XOR optimization)
- `stages > 1`: Index cycles 0→1→2→...→stages-1→0, phase increments when index wraps

**Why single Int32?**
- Reduces register pressure (1 register vs 2)
- Simpler to pass across function boundaries
- Compiles to efficient bit operations for power-of-2 stages

## 2.3 *_w_index_phase Methods

FlashAttention adds convenience methods that take `index` and `phase` as separate Int32 arguments:

```python
class _PipelineIndexPhaseMixin:
    """Mixin providing *_w_index_phase methods."""
    
    def producer_acquire_w_index_phase(self, index: Int32, phase: Int32, ...):
        state = _make_state(index, phase)  # Create PipelineState
        self.producer_acquire(state, ...)   # Call parent method
    
    def producer_commit_w_index(self, index: Int32, ...):
        state = _make_state(index, Int32(0))  # Phase not needed for commit
        self.producer_commit(state, ...)
    
    def consumer_wait_w_index_phase(self, index: Int32, phase: Int32, ...):
        state = _make_state(index, phase)
        self.consumer_wait(state, ...)
    
    def consumer_release_w_index(self, index: Int32, ...):
        state = _make_state(index, Int32(0))
        self.consumer_release(state, ...)
```

**Why use *_w_index_phase?**

1. **No PipelineState object creation**:
```python
# With PipelineState:
state = PipelineState(stages=2, index=idx, phase=ph)
pipeline.producer_acquire(state)

# With *_w_index_phase:
pipeline.producer_acquire_w_index_phase(idx, ph)  # No object creation
```

2. **Pass as kernel arguments**:
```python
# Kernel argument: (index, phase) as separate Int32s
# vs PipelineState object which needs __extract_mlir_values__
```

3. **Compute index/phase at runtime**:
```python
# When you compute index and phase separately:
idx = calculate_buffer_index(...)
ph = calculate_phase(...)
pipeline.consumer_wait_w_index_phase(idx, ph)
```

**When to use which:**
- Use `*_w_index_phase` when you have separate index/phase values
- Use PipelineState when you need to track state across iterations

## 2.4 PipelineAsync with elect_one

```python
@dataclass(frozen=True)
class PipelineAsync(_PipelineIndexPhaseMixin, PipelineAsyncOg):
    """PipelineAsync with optional elect_one for barrier operations.
    
    Args:
        elect_one_commit: If True, only elected thread signals producer_commit
        syncwarp_before_commit: If True, syncwarp before elect_one
        elect_one_release: If True, only elected thread signals consumer_release
        syncwarp_before_release: If True, syncwarp before elect_one
    """
    _elect_one_commit: bool = False
    _syncwarp_before_commit: bool = True
    _elect_one_release: bool = False
    _syncwarp_before_release: bool = True
```

### When to Use elect_one

**Without elect_one** (all threads signal):
```python
# All threads in consumer_group call barrier.arrive_and_release
# Good for: barrel equals thread count
```

**With elect_one** (one thread signals):
```python
# Only elected thread signals barrier
# Good for: barrier count is 1 per warp
with cute.arch.elect_one():
    pipeline.consumer_release(state)# Only one thread executes
```

## 2.4 Usage Example (FlashAttention)

```python
# From flash_fwd_sm90.py

# Create pipeline for KV loading
pipeline_k = pipeline.PipelineTmaAsync.create(
    num_stages=self.num_stages,# e.g., 2
    producer_group=load_warp,
    consumer_group=mma_warps,
)

# Initialize state
kv_producer_state = pipeline.make_pipeline_state(
    pipeline.PipelineUserType.Producer, self.num_stages
)
kv_consumer_state = pipeline.make_pipeline_state(
    pipeline.PipelineUserType.Consumer, self.num_stages
)

# Producer (TMA load warp)
for k_tile in range(k_tile_cnt):
    # Wait for buffer to be empty (TMA waits on empty barrier)
    pipeline_k.producer_acquire(kv_producer_state)
    
    # Issue TMA load
    # TMA writes to sK[(None, kv_producer_state.index)] directly
    cute.copy(tma_atom_K, gK, sK[(None, kv_producer_state.index)])
    
    # Signal buffer is full (sets mbarrier arrival count)
    pipeline_k.producer_commit(kv_producer_state)
    
    kv_producer_state.advance()

# Consumer (MMA warps)
for k_tile in range(k_tile_cnt):
    # Wait for buffer to be full
    pipeline_k.consumer_wait(kv_consumer_state)
    
    # Compute MMA
    # ... uses sK[(None, kv_consumer_state.index)]
    
    # Signal buffer is empty
    pipeline_k.consumer_release(kv_consumer_state)
    
    kv_consumer_state.advance()
```

## 2.5 PipelineTmaAsync with extra_tx_count

```python
def producer_acquire(
    self,
    state: PipelineState,
    try_acquire_token: Optional[Boolean] = None,
    extra_tx_count: int = 0,  # Additional TMA transactions to wait for
    ...
):
    # Wait for buffer to be empty
    if try_acquire_token is None or try_acquire_token == 0:
        self.sync_object_empty.wait(state.index, state.phase)
    
    # Arrive on full barrier with expected transaction count
    if extra_tx_count == 0:
        self.sync_object_full.arrive(state.index, self.producer_mask)
    else:
        # Total transactions = base_tx_count + extra_tx_count
        tx_count = self.sync_object_full.tx_count + extra_tx_count
        self.sync_object_full.arrive_and_expect_tx(state.index, tx_count)
```

**What is extra_tx_count?**

TMA loads use **transaction barriers** (`mbarrier`) that track expected byte count:
- `tx_count` = expected bytes from TMA operation (set when barrier created)
- `extra_tx_count` = additional bytes/transactions to wait for

**Use cases:**
1. **Multiple TMAs to same buffer**: When Q and K load to adjacent shared memory
2. **TMA + async operation**: When TMA and cp.async write to same buffer

```python
# Example: TMA load Q + another async operation
pipeline.producer_acquire(state, extra_tx_count=additional_bytes)
# Consumer waits until BOTH TMA bytes + async operation arrive
```

## 2.6 FlashAttention Class Hierarchy

FlashAttention wraps CuTe native classes:

```python
# FlashAttention imports CuTe native classes with "Og" suffix (Original)
from cutlass.pipeline import PipelineAsync as PipelineAsyncOg
from cutlass.pipeline import PipelineTmaAsync as PipelineTmaAsyncOg
from cutlass.pipeline import PipelineCpAsync as PipelineCpAsyncOg
# ... etc

@dataclass(frozen=True)
class PipelineAsync(_PipelineIndexPhaseMixin, PipelineAsyncOg):
    """Adds elect_one and syncwarp flags."""
    _elect_one_commit: bool = False
    _syncwarp_before_commit: bool = True
    _elect_one_release: bool = False
    _syncwarp_before_release: bool = True
```

**What FlashAttention adds:**

| Feature | CuTe Native | FlashAttention |
|---------|-------------|----------------|
| Base classes | PipelineAsync, PipelineTmaAsync, etc. | Same, wrapped as `*Og` |
| `*_w_index_phase` methods | No | Yes, via `_PipelineIndexPhaseMixin` |
| `elect_one_commit` flag | No | Yes |
| `syncwarp_before_commit` | No | Yes |
| `elect_one_release` flag | No | Yes |
| `syncwarp_before_release` | No | Yes |
| `PipelineStateSimple` | No | Yes |
| `extra_tx_count` parameter | No | Yes (on TMA pipelines) |

**Class hierarchy:**
```
CuTe Native                              FlashAttention
───────────                              ──────────────
PipelineAsyncOg      ─────────────────►  PipelineAsync
                                            + _PipelineIndexPhaseMixin
                                            + elect_one flags

PipelineTmaAsyncOg   ─────────────────►  PipelineTmaAsync
                                            + _PipelineIndexPhaseMixin
                                            + extra_tx_count parameter

PipelineCpAsyncOg    ─────────────────►  PipelineCpAsync
                                            + _PipelineIndexPhaseMixin
                                            + elect_one_release flag

PipelineTmaUmmaOg─────────────────►  PipelineTmaUmma
                                            + _PipelineIndexPhaseMixin
                                            + extra_tx_count parameter
```

## 2.7 Complete FlashAttention Pipeline Setup

```python
# From flash_fwd_sm90.py

# Setup for Q, K, V loading
pipeline_q = pipeline.PipelineTmaAsync.create(
    num_stages=1,
    producer_group=tma_warp,
    consumer_group=consumer_warps,
)

pipeline_k = pipeline.PipelineTmaAsync.create(
    num_stages=self.num_stages,
    producer_group=tma_warp,
    consumer_group=consumer_warps,
)

pipeline_v = pipeline.PipelineTmaAsync.create(
    num_stages=self.num_stages,
    producer_group=tma_warp,
    consumer_group=consumer_warps,
)

# Shared memory barriers
@mbar_load_Q: MemRange[Int64, num_stages * 2]# *2 for full and empty
@mbar_load_KV: MemRange[Int64, num_stages * 2]
```

---

# Part 3: Comparison

## 3.1 Feature Comparison

| Feature | CuTe Native | FlashAttention |
|---------|-------------|----------------|
| **Base classes** | PipelineAsync, PipelineTmaAsync, PipelineCpAsync, PipelineTmaUmma, PipelineUmmaAsync, etc. | Same classes wrapped (`*Og`), no new classes |
| **State management** | PipelineState | PipelineStateSimple (single Int32) |
| **elect_one support** | Manual `cute.arch.elect_one()` wrapper | Built-in `elect_one_commit`, `elect_one_release` flags |
| **syncwarp coordination** | Manual `cute.arch.sync_warp()` call | Built-in `syncwarp_before_commit`, `syncwarp_before_release` flags |
| **extra_tx_count** | Manual `arrive_and_expect_tx()` call | Built-in parameter on `producer_acquire` |
| **Index/phase methods** | Only via PipelineState object | Both PipelineState and `*_w_index_phase` variants |

## 3.2 Code Comparison

### CuTe Native

```python
# Create
pipeline = PipelineAsync.create(
    num_stages=2,
    producer_group=producer_warp,
    consumer_group=consumer_warp,
)

# State
producer_state = make_pipeline_state(PipelineUserType.Producer, 2)
consumer_state = make_pipeline_state(PipelineUserType.Consumer, 2)

# Producer
producer.acquire(producer_state)
# ... write to buffer ...
producer.commit(producer_state)
producer_state.advance()

# Consumer
consumer.wait(consumer_state)
# ... read from buffer ...
consumer.release(consumer_state)
consumer_state.advance()
```

### FlashAttention

```python
# Create (same)
pipeline = PipelineAsync.create(
    num_stages=2,
    producer_group=producer_warp,
    consumer_group=consumer_warp,
    elect_one_commit=True,# New: only elected thread commits
    syncwarp_before_commit=True,  # New: sync before elect_one
)

# State (optimized)
producer_state = make_pipeline_state(PipelineUserType.Producer, 2)
consumer_state = make_pipeline_state(PipelineUserType.Consumer, 2)

# Producer (same pattern)
pipeline.producer_acquire(producer_state)
# ... write to buffer ...
pipeline.producer_commit(producer_state)  # Uses elect_one internally
producer_state.advance()

# Consumer (same pattern)
pipeline.consumer_wait(consumer_state)
# ... read from buffer ...
pipeline.consumer_release(consumer_state)  # Uses elect_one internally
consumer_state.advance()
```

## 3.3 When to Use Which

**Use CuTe Native Pipelines when:**
- Building general GEMM kernels
- All threads participate in synchronization
- Standard producer-consumer pattern

**Use FlashAttention Pipelines when:**
- Need `elect_one` optimization (one thread per warp signals)
- Single-stage pipelines (use PipelineStateSimple)
- Need `extra_tx_count` for TMA

## 3.4 Performance Considerations

1. **Pipeline depth (`num_stages`)**:
   - More stages → more overlap, more smem
   - Typical: 2-3 stages for attention, 1 for simple kernels

2. **elect_one reduces barrier overhead**:
   - Instead of all threads calling `barrier.arrive`
   - One thread calls `barrier.arrive` with count = num_threads

3. **syncwarp before elect_one**:
   - Ensures all threads have written to smem
   - Before elected thread signals barrier

---

# Part 4: Common Patterns

## 4.1 Double Buffering (2-stage)

```python
# 2-stage pipeline overlaps load and compute
# While computing buffer 0, loading buffer 1

Time | Producer        | Consumer
-----|-----------------|------------------
t0| acq(0), load(0) | wait(prev)
t1| commit(0)       | wait(0)
t2| acq(1), load(1) | read(0), compute
t3| commit(1)        | release(0)
t4| acq(0), load(0) | wait(1)
t5| ...              | read(1), compute
```

## 4.2 Single Buffering (1-stage)

```python
# 1-stage pipeline with phase alternation
# Producer and consumer alternate on same buffer

Time | Producer      | Consumer
-----|---------------|--------
t0   | acq, load     | -
t1   | commit        | wait
t2   | -             | read, compute
t3   | acq (block)   | release
t4   | load          | -
```

## 4.3 Multi-stage (3+)

```python
# 3-stage pipeline for complex kernels
# More overlap, more smem, more latency hiding

Producer: acq(2) → load(2) → commit(2) → acq(0) → ...
Consumer:             wait(0) → read(0) → release(0) → wait(1) → ...
```

---

# Summary

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Pipeline** | Synchronization primitive for producer-consumer |
| **Barrier** | Memory barrier (`mbarrier`) for thread coordination |
| **Stage** | Buffer slot in circular buffer |
| **Phase** | Parity bit for barrier wait (alternates each iteration) |
| **acquire** | Producer waits for "buffer empty" |
| **commit** | Producer signals "buffer full" |
| **wait** | Consumer waits for "buffer full" |
| **release** | Consumer signals "buffer empty" |

## Pipeline Types Summary

| Type | Use Case |
|------|----------|
| PipelineAsync | General async threads |
| PipelineTmaAsync | TMA load→ smem → async compute |
| PipelineCpAsync | cp.async load → smem → async compute |
| PipelineTmaUmma | TMA load → tmem → UMMA compute |

## Best Practices

1. **Choose `num_stages` carefully**:
   - 1 stage: Minimal smem, no overlap
   - 2 stages: Good balance, double buffering
   - 3+ stages: Max overlap, high smem usage

2. **Use `elect_one` when barrier count is per-warp**:
   - Saves synchronization overhead
   - Must `syncwarp` before `elect_one`

3. **Initialize states correctly**:
   - Producer starts at index=stages, phase=1
   - Consumer starts at index=0, phase=0

4. **Advance state after each iteration**:
   - Updates index and phase
   - Critical for circular buffer
