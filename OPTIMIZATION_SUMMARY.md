# Project Out Experiment - Optimization Opportunities

## Current Bottlenecks Identified:

### 1. **Memory: Unnecessary Cloning** ⚠️
**Location:** `project_out_runner.py:235`
```python
modified = router.original_weights.clone()  # EVERY variant iteration
```
- Clones entire router matrix (8 x 4096) for each variant/k combination
- For k=[1,4,16,64,128] × 5 variants = 25 clones
- **Solution:** Reuse one clone, reset per variant

### 2. **Memory: Batch Pre-Loading** 
**Location:** `evaluation.py:23`
```python
for batch in batches:
    batch = batch.to(self._device)  # Moves to device repeatedly
```
- Batches moved from CPU→GPU on every evaluation
- Called ~30 times (baseline + zero + shuffle + variants×k)
- **Solution:** Pre-move all batches to device once at start

### 3. **Computation: Random Vector Generation**
**Location:** `project_out_runner.py:98`
```python
return intervention.make_random(v_first.shape[0], seed=seed)  # Per expert
```
- For "random" variant: generates 8 random vectors (one per expert)
- But they're all the same dimension, just different seeds
- **Minor optimization:** Could batch-generate

### 4. **Memory: Quantized Model + Small GPU**
**Location:** Line 135 (first forward pass)
- Even with mini-batch fix, quantized model uses 42+ GB
- Leaves <2 GB for activations
- **Solution:** Reduce batch_size for evaluation when quantized

### 5. **I/O: Expert Vectors Already Optimized** ✅
**Location:** `project_out_runner.py:204-216`
- Already loads max(k) once and slices - GOOD!
- No further optimization needed here

## Recommended Optimizations (Priority Order):

### HIGH PRIORITY - Memory Savings

#### 1. Pre-move Batches to Device (Saves repeated CPU→GPU transfers)
```python
# In runner, after loading batches:
with torch.no_grad():
    batches = [b.to(device) for b in batches]
```

#### 2. Reuse Cloned Weights (Saves 24 clones)
```python
# Outside variant loop:
modified = router.original_weights.clone()
for variant in self.config.variations:
    # Reset modified to original at start of each variant
    modified.copy_(router.original_weights)
    # Then modify in-place
```

#### 3. Reduce Batch Size for Quantized Models
```python
# In config/runner:
if self.config.quantization and self.config.batch_size > 2:
    logger.warning("Reducing batch_size to 2 for quantized model")
    actual_batch_size = 2
```

### MEDIUM PRIORITY - Speed

#### 4. Cache Router Original Weights
```python
# In router_manager.py:
# Instead of cloning on every .original_weights call, cache once
@property
def original_weights_cached(self):
    if self._cached_original is None:
        self._cached_original = self._original.clone()
    return self._cached_original
```

### LOW PRIORITY - Minor Gains

#### 5. Shuffle Without Clone
```python
# Line 190:
router.apply_weights(orig[perm])  # Remove .clone() - perm creates new view
```

## Expected Impact:

| Optimization | Memory Saved | Speed Gain | Effort |
|--------------|--------------|------------|--------|
| Pre-move batches | ~1-2 GB | ~15-20% | Low |
| Reuse cloned weights | ~200 MB | Minimal | Low |
| Reduce batch_size (quant) | ~500 MB | -10% (slower) | Low |
| Cache original weights | ~100 MB | Minimal | Medium |
| Remove shuffle clone | ~50 MB | Minimal | Low |

## Implementation Plan:

1. **Immediate (for quantized models):**
   - Pre-move batches to device
   - Reduce batch_size for quantization
   
2. **Next iteration:**
   - Reuse cloned weights
   - Cache router weights

3. **Nice to have:**
   - Remove unnecessary clones
