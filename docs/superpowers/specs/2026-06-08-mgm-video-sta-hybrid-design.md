# MGM-Video MMDiT — STA Hybrid (NABLA-style OR with ASA) Design

**Date:** 2026-06-08
**Author:** jiangyu741
**Status:** Draft for review
**Related:**
- Prior design: `docs/superpowers/specs/2026-06-02-mgm-video-asa-adaptation-design.md`
- Prior plan: `docs/superpowers/plans/2026-06-02-mgm-video-asa-adaptation.md`
- Quality probe: `docs/analysis/2026-06-02-mgm-video-asa-quality-probe.md`
- Paper: ∇NABLA (arXiv 2507.13546); analysis at `/home/j00935189/code/omnia/omnia/paper_reading/2026.06/2026-06-08-NABLA.md`
- Reference impl: `/home/j00935189/code/t2v/Wan2.1-NABLA/wan/modules/attention.py:217-280`

## 1. Background & Motivation

Commit `dfc7620d` adapted BLADE ASA (Adaptive Sparse Attention) to the MGM-Video MMDiT inference path. ASA prunes attention block-pairs based on per-row energy of a sample-pooled importance matrix, producing a dynamic, content-aware sparse mask.

Validation runs surfaced a quality regression: visible **noise points** in the generated video. The ∇NABLA paper documents the same failure mode of pure dynamic block-sparse attention — visible boundary artifacts between regions selected/unselected by per-row routing — and proposes a remedy: combine the dynamic mask with a **static local neighborhood mask (STA, Sliding Tile Attention)** via logical OR. The static prior guarantees every token always sees its local 3D neighborhood; the dynamic component still routes long-range dependencies.

This design adapts that hybrid to MGM-Video's existing ASA stack with the smallest viable code surface, no NPU kernel changes, and bit-exact fall-back to today's behavior when disabled.

## 2. Goals & Non-Goals

**Goals (scope 1, MVP):**
- Add a static STA block-level mask, OR'd into ASA's existing block mask
- Coupled to ASA's existing step gate (warmup_steps + step_scheme_path)
- Single fixed STA window per run, env-configurable
- Bit-exact reproduction of current ASA behavior when STA is disabled
- Pure tensor algebra; no new NPU operator dependencies

**Non-goals (deferred):**
- Standalone STA without ASA
- Per-step STA scheme independent of ASA's
- Retuning ASA `block_size=128` to a NABLA-aligned 64
- Hybrid with `variant="asa_g"` (LSE/global path)
- Per-layer or per-head window tuning
- STA mask polarity stats / collect_stats integration

## 3. Architecture Overview

Insertion point: one OR statement at `vllm_omni/diffusion/models/mgm_video/mmdit/asa.py` line 488 (between `build_asa_block_mask` and `expand_block_to_token_mask`).

```
JoinAttentionInference.infer  (mmdit_blocks_inference.py:340)
  └─ asa_attention  (asa.py:474)
      ├─ Gilbert reorder q/k/v (video segment)
      ├─ sample_pool_attn → imp [B,1,nq,nk]
      ├─ build_asa_block_mask → m_asa_block [B,1,nq,nk]
      ├─ if cfg.sta_enable:                                    ◀── NEW
      │     m_sta_block = sta_cache.get_or_build(...)          ◀── NEW
      │     m_block = m_asa_block | m_sta_block                ◀── NEW
      ├─ expand_block_to_token_mask → m_token [B,1,S,S]
      └─ fa(~m_token) → npu_fusion_attention
```

**Files:**
- New: `vllm_omni/diffusion/models/mgm_video/mmdit/sta.py` (~150 LOC)
- Modified: `vllm_omni/diffusion/models/mgm_video/mmdit/asa.py` — extend `AsaConfig`, 6-line OR block at line 488
- Modified: `vllm_omni/diffusion/models/mgm_video/mmdit/mmdit_blocks_inference.py` — wire `_sta_cache` slot next to `_asa_rearranger`
- New: `tests/mgm_video/test_sta.py` (~20 cases)

**Why block-mask level (not token-mask level):** OR on `[1,1,nq,nk]` (~200 KB at nq=453) vs OR on `[1,1,S,S]` (~3.3 GB at S=57.6k for `torch.bool`, 1 byte per element). Block-mask OR is ~16000× cheaper, and downstream `expand_block_to_token_mask` is block-aligned anyway, so no precision loss.

**Why STA is gated under ASA:** in scope (1) STA fires only when ASA does. Standalone STA would need its own branch through `JoinAttentionInference.infer`; coupling is one OR line.

## 4. STA Block-Mask Construction

ASA pools and masks in **Gilbert block order**, not row-major (T,H,W). NABLA's reference `sta()` (`Wan2.1-NABLA/wan/modules/attention.py:217-240`) builds outer-product masks on raw (T,H,W) axes, assuming row-major layout. It cannot be ported as-is. We replace it with a per-block **AABB-within-window** test computed directly in Gilbert order.

### 4.1 Bounding-box reduction

For each Gilbert block `i ∈ [0, nq_video)`, look up its constituent original-coord tokens via the existing `gilbert2original` buffer (built in `GilbertRearranger.__init__`, `asa.py:242-249`). Compute a (t, h, w) bounding box per block:

```
bbox_i = (t_min_i, t_max_i, h_min_i, h_max_i, w_min_i, w_max_i)
```

Two blocks have an STA edge iff their AABBs are within the half-window along **every** axis:

```
sta_block[i, j] = (max(t_min_i - t_max_j, t_min_j - t_max_i) <= wT // 2)
                AND (same test for H using wH)
                AND (same test for W using wW)
```

**Equivalence to token-level STA:** STA at the token level is `|Δt| ≤ wT//2 ∧ |Δh| ≤ wH//2 ∧ |Δw| ≤ wW//2`. The block-level OR over all token pairs in (i, j) reduces exactly to "the AABBs come within window along every axis" — standard AABB-within-distance test, no approximation.

**Cost:** integer index ops + 6 reductions over `block_size` tokens + 3 broadcast comparisons over `[nq, nq]`. For nq=453 that's ~600 KB intermediate bool. Computed once, cached for the run.

### 4.2 Padding tokens and text blocks

- **Last incomplete video block** may have fewer than `block_size` real tokens. Padding entries get sentinel coords with a large negative value (`-(2**30)` for `t_min/h_min/w_min`-side reductions and `+(2**30)` for `t_max/h_max/w_max`-side reductions) so they fail every AABB-within-window test. Choose `2**30` rather than `INT_MIN` to avoid overflow when the AABB test computes signed differences. Existing key-padding logic in ASA already neutralizes the same tokens at the attention stage.
- **Text blocks** (the trailing `ceil(L / block_size)` blocks containing the 256 text tokens) are forced `True` in both rows and columns. This matches `expand_block_to_token_mask`'s text-dense convention (`asa.py:437-438`). STA never restricts text↔video or text↔text — only video↔video.

### 4.3 Caching

`StaMaskCache` keyed by `(T, H, W, block_size, window, n_video_tokens, gilbert_id)`, holding the `[1,1,nq,nk]` bool tensor. Built lazily on the first ASA-active forward pass, lives on `JoinAttentionInference._sta_cache` for the run, reused for all remaining steps and all remaining MMDiT blocks.

### 4.4 Skeleton

```python
@dataclass
class StaConfig:
    enable: bool = False
    window: tuple[int, int, int] = (7, 13, 13)  # (wT, wH, wW)

def build_sta_block_mask_in_gilbert_order(
    T: int, H: int, W: int,
    block_size: int,
    n_blocks_total: int,
    n_video_tokens: int,
    gilbert2original: Tensor,
    window: tuple[int, int, int],
    device,
) -> Tensor:                                # [1, 1, n_blocks_total, n_blocks_total] bool
    # 1) Unravel gilbert2original index -> (t, h, w) per video token
    # 2) Pad video tokens up to nq_video * block_size with sentinel coords
    # 3) Reshape to [nq_video, block_size, 3], reduce min/max along block dim
    # 4) Pairwise AABB-within-window test -> [nq_video, nq_video] bool
    # 5) Pad to full [n_blocks_total, n_blocks_total]; force trailing rows/cols True
    # 6) Reshape to [1, 1, nq, nk]
```

## 5. Configuration

**Single config object:** extend `AsaConfig` (in `asa.py:21-58`) rather than introduce a separate `StaConfig`. STA is gated under ASA in scope (1) and shares `enable`, `warmup_steps`, `step_scheme_path`, and the Gilbert rearranger; splitting configs would force duplicate plumbing.

**New fields:**

| Field | Type | Default | Env var | Purpose |
|---|---|---|---|---|
| `sta_enable` | `bool` | `False` | `VLLM_MGM_STA_ENABLE` (`"1"`) | Master STA toggle |
| `sta_window` | `tuple[int, int, int]` | `(7, 13, 13)` | `VLLM_MGM_STA_WINDOW` (`"7,13,13"`) | (wT, wH, wW) full-extent |

**Window convention:** "full-extent" means a token at coord `c` attends to neighbors in `[c - w//2, c + w//2]`. So `wT=7` ≈ ±3 frames, `wH=13` ≈ ±6 latent units H, `wW=13` ≈ ±6 latent units W. Asymmetric on purpose — temporal axis is much shorter than spatial in the latent grid (T=16 vs H=45, W=80 for 81-frame 480p).

**Validation at config load:**
- Window must be length-3 tuple of odd positive ints. Even values round up to next odd at first forward, with a warning. Malformed env string falls back to default + warning, no crash.
- Window-vs-latent comparison deferred to first forward (`(T, H, W)` unknown at config time). On any axis where `w > axis_dim`, clamp to `axis_dim` (or `axis_dim - 1` to preserve oddness) and warn.

## 6. Step Gating, CP, Variant Edge Cases

**Step gating:** STA reuses `_should_use_dense_for_step` (`mmdit_blocks_inference.py:63-95`). When the gate routes to dense, neither ASA nor STA runs. When it routes to ASA, both run and OR. No new gate logic, no new env vars beyond §5.

**Mask is shape-only:** STA mask depends purely on `(T, H, W, block_size, window, gilbert_index)` — no Q/K, no `step_idx`, no head dim. Cached on first ASA-active forward, reused across steps and MMDiT blocks.

**Context parallelism (CP):** ASA's mask is built before head-shard all-to-all (`mmdit_blocks_inference.py:214-241`); the sequence is not sharded, only heads are. STA's block mask is built at the same point and is identical across CP ranks. No CP-aware rebalancing needed.

**Variant interaction:**
- `variant="dense_probe"` — ASA produces all-True; OR is a no-op. Skip the STA cache build and OR; emit one informational log on first forward.
- `variant="asa"` — normal hybrid OR. Primary target.
- `variant="asa_g"` — LSE/global path; not in scope (1). If `sta_enable=True`, log a one-line warning on first forward and skip OR.

**Out-of-range step_idx (`None`, negative, beyond scheme length):** existing ASA fallback rule applies — fall through to ASA path. STA OR runs as designed.

**Bit-exact guarantee:** `sta_enable=False` AND `VLLM_MGM_STA_ENABLE` unset → no new code paths execute. `m_block` flows untouched into `expand_block_to_token_mask`. Acceptance test asserts byte-identical `m_token` against a snapshot from the pre-STA path.

## 7. Testing Strategy

New file `tests/mgm_video/test_sta.py`, mirroring `test_asa.py`. ~18-22 cases, all `@pytest.mark.cpu @pytest.mark.core_model` (L1).

**(a) Config & env (5 cases):** defaults; `VLLM_MGM_STA_ENABLE=1` flips bool; `VLLM_MGM_STA_WINDOW="7,13,13"` parses; malformed string falls back + warns; even window value rounds up at first forward.

**(b) Bbox-AABB correctness (6 cases):** tiny grid `(T=2,H=4,W=4)` with hand-computed expected adjacency; identity (i==j) always True; symmetry; window=(1,1,1) with block_size=1 reduces to identity; window spanning entire grid → all-True; padding tokens never enable spurious edges.

**(c) Gilbert ordering compatibility (3 cases):** for every token pair within window in original coords, their Gilbert blocks are connected; conversely, two blocks with no within-window token pair are disconnected; text rows/cols all-True regardless of window.

**(d) OR-combine + integration (3 cases):** `m_asa | m_sta` with diagonal-only ASA; both all-True → all-True; after OR, `expand_block_to_token_mask` produces video×video region equal to expanded OR with text rows/cols all-True.

**(e) End-to-end gate + bit-exact (3 cases):** `sta_enable=False` → byte-identical `m_token` vs pre-STA snapshot; `warmup_steps=2, step_idx=1` → STA does not contribute (gate dense); `warmup_steps=0, step_idx=0` → STA contributes; result differs from pure-ASA exactly in entries enabled by STA.

**(f) Variant edge cases (2 cases):** `sta_enable=True, variant="dense_probe"` → STA skipped + warning; `sta_enable=True, variant="asa_g"` → STA skipped + warning.

**Quality validation (out of unit-test scope):** reuse `scripts/probe_mgm_asa.py`. Add a flag (or env switch) to compare pure-ASA, ASA+STA, and dense reference on the same prompts. Manual NPU runs; not gated by unit tests.

## 8. Documentation Deliverables

- **This spec:** `docs/superpowers/specs/2026-06-08-mgm-video-sta-hybrid-design.md`
- **Plan:** to be produced by the `writing-plans` skill in the next step — phased steps with acceptance gates
- **Probe report (post-impl):** new section in `docs/analysis/2026-06-02-mgm-video-asa-quality-probe.md` (existing ASA report) comparing ASA-only vs ASA+STA on the same prompt set; reuse infra rather than fork the report

## 9. Risks

- **R-STA-1 — Window default may be wrong.** `(7, 13, 13)` is a starting guess on a `(16, 45, 80)` latent. Mitigation: `VLLM_MGM_STA_WINDOW` env var; sweep windows on the probe harness as part of validation.
- **R-STA-2 — Block-bbox AABB reduction.** Claimed equivalent to per-token STA semantics under "any-pair within window" reduction; verified by unit tests in 7(c). Add a property test if a counterexample appears in probe runs.
- **R-STA-3 — Adding edges may not fix noise points.** Hypothesis: ASA over-prunes critical local-neighborhood blocks. If hybrid OR doesn't move the noise needle, the issue is elsewhere (energy threshold, Gilbert reorder boundaries, etc.). Probe harness reveals this in one validation run.
- **R-STA-4 — Memory.** Block-mask OR adds ~200 KB intermediate; token mask is unchanged from today (~3.3 GB at S=57.6k for the dense `[1,1,S,S]` bool). No regression vs today's ASA path.
- **R-STA-5 — Caching key correctness.** Cache key must include all shape inputs and gilbert identity; a stale cache across two generations with different `(T, H, W)` would silently produce wrong masks. Mitigation: include `id(gilbert2original)` and all numeric params in the key; assert cache miss on shape change.

## 10. Acceptance Criteria

- `sta_enable=False` and `VLLM_MGM_STA_ENABLE` unset reproduce the pre-STA path bit-exact (snapshot test passes).
- All ~20 unit tests pass on CPU/L1.
- Generation with `VLLM_MGM_ASA_ENABLE=1 VLLM_MGM_STA_ENABLE=1 VLLM_MGM_STA_WINDOW="7,13,13"` runs to completion on NPU without OOM, with end-to-end latency within 10% of pure-ASA at the same step schedule.
- Probe-harness comparison report includes pure-ASA vs ASA+STA on at least one prompt set.
