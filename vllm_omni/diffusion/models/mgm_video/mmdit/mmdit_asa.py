# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BLADE Adaptive Block-Sparse Attention (ASA) — inference-time quality probe.

Pure-torch reference implementation that builds a binary block-sparse
attention mask over the video<->video sub-block of MGM-Video MMDiT attention.

This is a QUALITY PROBE: it does NOT skip blocks and yields NO wall-clock
speedup (dense FA + mask recomputes masked positions). Its sole purpose is to
let us eyeball whether block sparsity degrades the already-distilled 8-step
model's video quality before investing in a fused block-sparse kernel.

Faithful-but-simplified vs the paper:
- raster block partition (NO Gilbert reorder) -> this is a quality LOWER bound.
- exact mean-pooled block importance (NO k=16 sampling approximation).
"""

import os

import torch
import torch.nn.functional as F


def _parse_range(spec):
    """Parse 'all' or 'lo-hi' (inclusive, 0-based) or 'n' into a set of ints.

    Returns None for 'all' (means "every index").
    """
    spec = (spec or "all").strip().lower()
    if spec == "all":
        return None
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return set(range(int(lo), int(hi) + 1))
    return {int(spec)}


class AsaConfig:
    """ASA probe configuration, sourced from VLLM_MGM_ASA_* env vars."""

    def __init__(self, enable, tau, block, layers, steps, per_head, log):
        self.enable = enable
        self.tau = tau
        self.block = block
        self.layers = layers  # raw range string, parsed lazily
        self.steps = steps    # raw range string, parsed lazily
        self.per_head = per_head
        self.log = log

    @classmethod
    def from_env(cls):
        return cls(
            enable=os.environ.get("VLLM_MGM_ASA_ENABLE", "0") == "1",
            tau=float(os.environ.get("VLLM_MGM_ASA_TAU", "0.95")),
            block=int(os.environ.get("VLLM_MGM_ASA_BLOCK", "128")),
            layers=os.environ.get("VLLM_MGM_ASA_LAYERS", "all"),
            steps=os.environ.get("VLLM_MGM_ASA_STEPS", "all"),
            per_head=os.environ.get("VLLM_MGM_ASA_PER_HEAD", "1") == "1",
            log=os.environ.get("VLLM_MGM_ASA_LOG", "1") == "1",
        )


def should_apply_asa(cfg, layer_idx, step_idx, downscale):
    """Decide whether ASA masking applies at this (layer, step).

    Note: TDM-cache skip is handled UPSTREAM (skip_compute short-circuits
    before infer() is called), so no cache-table lookup is needed here.
    """
    if not cfg.enable:
        return False
    if downscale != 1:
        # skiparse static sparsity is mutually exclusive with ASA
        return False
    layers = _parse_range(cfg.layers)
    if layers is not None and layer_idx not in layers:
        return False
    if step_idx is not None:
        steps = _parse_range(cfg.steps)
        if steps is not None and step_idx not in steps:
            return False
    return True


def _block_importance_keep(q_v, k_v, block, tau, scale):
    """Exact mean-pooled block importance -> binary block keep-mask.

    Args:
        q_v, k_v: [Sv, D] single-head video query/key.
        block: block size b.
        tau: cumulative threshold (nucleus selection).
        scale: softmax scale (head_dim ** -0.5).

    Returns:
        keep: [nB, nB] bool, True = KEEP (block participates).
        nB: number of blocks.
        pad: zero-padding added to reach nB*block.
    """
    Sv, D = q_v.shape
    nB = (Sv + block - 1) // block
    pad = nB * block - Sv
    if pad:
        q_v = F.pad(q_v, (0, 0, 0, pad))
        k_v = F.pad(k_v, (0, 0, 0, pad))

    # 1. mean-pool each KV block (query-independent)
    k_blk = k_v.reshape(nB, block, D).mean(dim=1)            # [nB, D]
    # 2. token-row x block-col approximate scores
    s_imp = (q_v @ k_blk.transpose(0, 1)) * scale            # [nB*block, nB]
    # 3. max-pool over query block rows (shared mask per query block)
    s_blk = s_imp.reshape(nB, block, nB).amax(dim=1)         # [nB, nB]
    # 4. row softmax + nucleus (cumulative >= tau) selection
    p = torch.softmax(s_blk.float(), dim=-1)                 # [nB, nB]
    sorted_p, sorted_idx = torch.sort(p, dim=-1, descending=True)
    cum = torch.cumsum(sorted_p, dim=-1)
    # keep a block if cumulative-before-it < tau (so the block crossing tau is kept)
    keep_sorted = (cum - sorted_p) < tau
    keep = torch.zeros_like(p, dtype=torch.bool)
    keep.scatter_(-1, sorted_idx, keep_sorted)
    # 5. diagonal self-select (locality safety net)
    diag = torch.arange(nB, device=q_v.device)
    keep[diag, diag] = True
    return keep, nB, pad


def build_asa_block_mask(q, k, T, L, block, tau, scale, log=False,  # noqa: N803
                         layer=-1, step=None):
    """Build a video<->video block-sparse attention mask.

    Args:
        q, k: [bs, 1, S, D] single-head (CP processes one head per fa call),
              S = T + L, video tokens [0:T], text tokens [T:T+L].
        T, L: video / text sequence lengths.
        block, tau, scale: ASA hyper-parameters.
        log: print per-call sparsity stat.
        layer, step: for logging only.

    Returns:
        atten_mask: [bs, 1, S, S] bool, True = MASKED (npu_fusion convention).
                    text rows and text cols are always dense (never masked).
    """
    bs, n, S, D = q.shape
    assert n == 1, "ASA probe expects single-head fa calls (CP head-major loop)"
    device = q.device

    q_v = q[0, 0, :T]                                        # [T, D]
    k_v = k[0, 0, :T]
    keep_blk, nB, pad = _block_importance_keep(q_v, k_v, block, tau, scale)

    # atten_mask: start all-keep (False), then mask the dropped video token blocks.
    atten = torch.zeros(S, S, dtype=torch.bool, device=device)
    mask_blk = ~keep_blk                                     # [nB, nB] True = masked
    mask_tok = mask_blk.repeat_interleave(block, 0).repeat_interleave(block, 1)
    atten[:T, :T] = mask_tok[:T, :T]
    # text rows/cols stay False (dense) by construction.

    if log:
        sparsity = mask_blk.float().mean().item()
        print(f"[ASA][raster, no-gilbert] layer={layer} step={step} "
              f"nB={nB} tau={tau} block={block} block_sparsity={sparsity:.1%}")

    return atten.unsqueeze(0).unsqueeze(0).expand(bs, 1, S, S)
