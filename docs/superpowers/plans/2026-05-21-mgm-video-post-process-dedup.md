# MGM Video Post-Process Dedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate duplicate post-processing in MGM Video pipeline by making `get_mgm_video_post_process_func` detect NPU-preprocessed uint8 tensors and skip redundant clamp/normalize/permute operations.

**Architecture:** Use tensor `dtype` as an implicit state flag (`uint8` = already post-processed on NPU, `float32` = raw VAE output). Zero changes to vllm-omni core code — only touch MGM pipeline file.

**Tech Stack:** Python, PyTorch

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `vllm_omni/diffusion/models/mgm_video/pipeline_mgm_video.py` | Modify | Single file: update `get_mgm_video_post_process_func`, fix `DiffusionOutput` return, update comments |

---

### Task 1: Update `get_mgm_video_post_process_func` to detect uint8

**Files:**
- Modify: `vllm_omni/diffusion/models/mgm_video/pipeline_mgm_video.py:111-137`

**Context:** The current `post_process_func` unconditionally applies clamp → normalize → permute → uint8 conversion. When `forward()` has already done this on NPU, the function runs the same operations again on CPU, producing wrong results (uint8 values passed through float arithmetic).

**Fix:** Add a `dtype == torch.uint8` guard at the top — if already processed, just `.numpy()` and return.

- [ ] **Step 1: Apply the edit**

Replace lines 111-137 with:

```python
def get_mgm_video_post_process_func(od_config: OmniDiffusionConfig):
    """Post-process function for MGM-Video.

    forward() 中已在 NPU 上完成的后处理：
      clamp → normalize → uint8 → permute → CPU

    本函数通过 dtype 检测状态：
      - uint8：NPU 后处理已完成，直接转 numpy
      - float32：兜底路径，执行完整 CPU 后处理
    """

    def post_process_func(video: torch.Tensor, output_type: str = "np"):
        if output_type == "latent":
            return video

        # forward() 已在 NPU 上完成后处理（dtype=uint8, 已 permute+CPU）
        if video.dtype == torch.uint8:
            return video.numpy()

        # 兜底：原始 VAE 输出，执行完整后处理
        low, high = -1.0, 1.0
        video = torch.clamp(video, min=low, max=high)
        video = video.sub(low).div(max(high - low, 1e-5))
        video = video.mul(255).add(0.5).clamp(0, 255)
        video = video.permute(0, 2, 3, 4, 1)
        video = video.to("cpu", torch.uint8).numpy()
        return video

    return post_process_func
```

- [ ] **Step 2: Verify the edit**

Run: `grep -n "video.dtype == torch.uint8" vllm_omni/diffusion/models/mgm_video/pipeline_mgm_video.py`
Expected: One match on the new line.

---

### Task 2: Fix `DiffusionOutput` return statement

**Files:**
- Modify: `vllm_omni/diffusion/models/mgm_video/pipeline_mgm_video.py:731`

**Context:** Line 731 passes `_post_processed=(output_type != "latent")` to `DiffusionOutput`, but `DiffusionOutput` dataclass (defined in `data.py`) has no such field. This raises `TypeError` at runtime.

**Fix:** Remove the invalid kwarg.

- [ ] **Step 1: Apply the edit**

Replace:
```python
return DiffusionOutput(output=output, _post_processed=(output_type != "latent"))
```
With:
```python
return DiffusionOutput(output=output)
```

- [ ] **Step 2: Verify the edit**

Run: `grep -n "DiffusionOutput(output=output)" vllm_omni/diffusion/models/mgm_video/pipeline_mgm_video.py`
Expected: One match on line 731.

Run: `grep -n "_post_processed" vllm_omni/diffusion/models/mgm_video/pipeline_mgm_video.py`
Expected: No matches.

---

### Task 3: Update forward() NPU post-process comments

**Files:**
- Modify: `vllm_omni/diffusion/models/mgm_video/pipeline_mgm_video.py:718-724`

**Context:** The existing comment claims "The engine-side post_process_func will detect the pre-processed uint8 tensor and skip." This is false — the engine does not perform any detection. The skipping is done inside `get_mgm_video_post_process_func` via dtype check.

**Fix:** Update the comment to accurately describe the mechanism.

- [ ] **Step 1: Apply the edit**

Replace lines 718-724:
```python
            # Post-process on GPU before returning: convert float32 [-1,1]
            # to uint8 [0,255] and permute [B,C,T,H,W] -> [B,T,H,W,C].
            # Doing this on GPU is ~20x faster than CPU (NPU parallel vs
            # serial memory traversal), and reduces the data transferred
            # through IPC from 1.27 GB (float32) to 334 MB (uint8) for
            # 121-frame 720p video. The engine-side post_process_func
            # will detect the pre-processed uint8 tensor and skip.
```
With:
```python
            # Post-process on GPU before returning: convert float32 [-1,1]
            # to uint8 [0,255] and permute [B,C,T,H,W] -> [B,T,H,W,C].
            # Doing this on GPU is ~20x faster than CPU (NPU parallel vs
            # serial memory traversal), and reduces the data transferred
            # through IPC from 1.27 GB (float32) to 334 MB (uint8) for
            # 121-frame 720p video.
            #
            # get_mgm_video_post_process_func() detects the pre-processed
            # uint8 tensor (via dtype check) and skips redundant ops.
```

- [ ] **Step 2: Verify the edit**

Run: `grep -n "detects the pre-processed" vllm_omni/diffusion/models/mgm_video/pipeline_mgm_video.py`
Expected: One match.

---

### Task 4: Syntax / import check

- [ ] **Step 1: Verify Python syntax**

Run: `python -m py_compile vllm_omni/diffusion/models/mgm_video/pipeline_mgm_video.py`
Expected: No output (success).

- [ ] **Step 2: Check no new imports needed**

The `torch.uint8` reference already works — `torch` is imported at the top of the file. No new imports required.

---

### Task 5: Commit

- [ ] **Step 1: Stage and commit**

```bash
git add vllm_omni/diffusion/models/mgm_video/pipeline_mgm_video.py
git commit -m "fix(mgm_video): eliminate duplicate post-processing via dtype check

- get_mgm_video_post_process_func now detects uint8 tensors (already
  post-processed on NPU in forward()) and skips redundant clamp/
  normalize/permute ops, only calling .numpy()
- Remove invalid _post_processed kwarg from DiffusionOutput constructor
- Update comments to accurately describe the skip mechanism

This avoids double post-processing without modifying any vllm-omni
core code (data.py, diffusion_engine.py, registry.py)."
```

---

## Self-Review

### Spec Coverage Check

| Spec Section | Implementing Task |
|--------------|-------------------|
| `get_mgm_video_post_process_func` uint8 guard | Task 1 |
| `DiffusionOutput` return fix | Task 2 |
| Comment update | Task 3 |
| Zero core code changes | Enforced by only touching MGM pipeline file |

No gaps — all design points covered.

### Placeholder Scan

- No "TBD", "TODO", "implement later" found.
- No vague "add error handling" or "write tests" without code.
- All code blocks contain complete, ready-to-paste code.

### Type Consistency

- `video.dtype == torch.uint8` — consistent with PyTorch API
- `DiffusionOutput(output=output)` — matches dataclass constructor signature
- No type mismatches across tasks.
