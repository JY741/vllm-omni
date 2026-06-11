# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""BLADE ASA (Adaptive Sparse Attention) adaptation for mgm_video MMDiT.

A-phase: precision feasibility probe using dense atten_mask via existing
npu_fusion_attention. No new NPU kernel introduced.

See: docs/superpowers/specs/2026-06-02-mgm-video-asa-adaptation-design.md
"""

import logging
import math
import os
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AsaConfig:
    enable: bool = False
    variant: Literal["dense_probe", "asa", "asa_g"] = "asa"
    max_retain_ratio: float = 0.20
    min_retain_ratio: float = 0.05
    energy_threshold: float = 0.95
    block_size: int = 128
    num_keep: int = 32
    sample_gap: int = 30
    use_gilbert: bool = True
    text_length: int = 256
    video_shape: tuple[int, int, int] | None = None  # (W, H, T)
    collect_stats: bool = False
    warmup_steps: int = 0
    step_scheme_path: str | None = None
    # STA hybrid (NABLA-style OR). See docs/superpowers/specs/2026-06-08-mgm-video-sta-hybrid-design.md
    sta_enable: bool = False
    sta_window: tuple[int, int, int] = (7, 13, 13)  # (wT, wH, wW), full extent

    @classmethod
    def from_env(cls) -> "AsaConfig":
        def _f(name, default, cast):
            v = os.environ.get(name)
            return cast(v) if v is not None else default

        def _parse_window(s: str) -> tuple[int, int, int]:
            parts = s.split(",")
            if len(parts) != 3:
                raise ValueError(f"expected 3 comma-separated ints, got {s!r}")
            return (int(parts[0]), int(parts[1]), int(parts[2]))

        sta_window = cls.__dataclass_fields__["sta_window"].default
        raw_window = os.environ.get("VLLM_MGM_STA_WINDOW")
        if raw_window is not None:
            try:
                sta_window = _parse_window(raw_window)
            except ValueError as exc:
                log.warning(
                    "VLLM_MGM_STA_WINDOW=%r malformed (%s); falling back to default %r",
                    raw_window, exc, sta_window,
                )

        return cls(
            enable=_f("VLLM_MGM_ASA_ENABLE", False, lambda v: v == "1"),
            variant=_f("VLLM_MGM_ASA_VARIANT", "asa", str),
            max_retain_ratio=_f("VLLM_MGM_ASA_MAX_RETAIN", 0.20, float),
            min_retain_ratio=_f("VLLM_MGM_ASA_MIN_RETAIN", 0.05, float),
            energy_threshold=_f("VLLM_MGM_ASA_ENERGY", 0.95, float),
            block_size=_f("VLLM_MGM_ASA_BLOCK_SIZE", 128, int),
            num_keep=_f("VLLM_MGM_ASA_NUM_KEEP", 32, int),
            sample_gap=_f("VLLM_MGM_ASA_SAMPLE_GAP", 30, int),
            use_gilbert=_f("VLLM_MGM_ASA_USE_GILBERT", True, lambda v: v == "1"),
            text_length=_f("VLLM_MGM_ASA_TEXT_LEN", 256, int),
            collect_stats=_f("VLLM_MGM_ASA_COLLECT_STATS", False, lambda v: v == "1"),
            warmup_steps=_f("VLLM_MGM_ASA_WARMUP_STEPS", 0, int),
            step_scheme_path=_f("VLLM_MGM_ASA_STEP_SCHEME", None, str),
            sta_enable=_f("VLLM_MGM_STA_ENABLE", False, lambda v: v == "1"),
            sta_window=sta_window,
        )


def parse_asa_step_scheme(path: str) -> tuple[int, ...]:
    """Parse ASA step scheme: one int (0 or 1) per non-comment, non-blank line.

    1 = ASA on this step, 0 = dense full attention on this step.
    Lines starting with '#' and blank lines are ignored.
    Whitespace around tokens is stripped.

    Raises ValueError on any malformed line, with file path, line number,
    and offending token in the message.
    """
    values: list[int] = []
    with open(path) as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line not in ("0", "1"):
                raise ValueError(
                    f"{path}:{lineno}: expected '0' or '1', got {raw!r}"
                )
            values.append(int(line))
    return tuple(values)


# Ported from BLADE cogvideox/train/special_attentions_local/utils/gilbert3d.py
# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2018 Jakub Červený


def sgn(x: int) -> int:
    return -1 if x < 0 else (1 if x > 0 else 0)


def generate3d(
    x: int, y: int, z: int,
    ax: int, ay: int, az: int,
    bx: int, by: int, bz: int,
    cx: int, cy: int, cz: int,
):
    w = abs(ax + ay + az)
    h = abs(bx + by + bz)
    d = abs(cx + cy + cz)

    (dax, day, daz) = (sgn(ax), sgn(ay), sgn(az))  # unit major direction ("right")
    (dbx, dby, dbz) = (sgn(bx), sgn(by), sgn(bz))  # unit ortho direction ("forward")
    (dcx, dcy, dcz) = (sgn(cx), sgn(cy), sgn(cz))  # unit ortho direction ("up")

    # trivial row/column fills
    if h == 1 and d == 1:
        for i in range(0, w):
            yield (x, y, z)
            (x, y, z) = (x + dax, y + day, z + daz)
        return

    if w == 1 and d == 1:
        for i in range(0, h):
            yield (x, y, z)
            (x, y, z) = (x + dbx, y + dby, z + dbz)
        return

    if w == 1 and h == 1:
        for i in range(0, d):
            yield (x, y, z)
            (x, y, z) = (x + dcx, y + dcy, z + dcz)
        return

    (ax2, ay2, az2) = (ax // 2, ay // 2, az // 2)
    (bx2, by2, bz2) = (bx // 2, by // 2, bz // 2)
    (cx2, cy2, cz2) = (cx // 2, cy // 2, cz // 2)

    w2 = abs(ax2 + ay2 + az2)
    h2 = abs(bx2 + by2 + bz2)
    d2 = abs(cx2 + cy2 + cz2)

    # prefer even steps
    if (w2 % 2) and (w > 2):
        (ax2, ay2, az2) = (ax2 + dax, ay2 + day, az2 + daz)

    if (h2 % 2) and (h > 2):
        (bx2, by2, bz2) = (bx2 + dbx, by2 + dby, bz2 + dbz)

    if (d2 % 2) and (d > 2):
        (cx2, cy2, cz2) = (cx2 + dcx, cy2 + dcy, cz2 + dcz)

    # wide case, split in w only
    if (2 * w > 3 * h) and (2 * w > 3 * d):
        yield from generate3d(x, y, z, ax2, ay2, az2, bx, by, bz, cx, cy, cz)
        yield from generate3d(x + ax2, y + ay2, z + az2, ax - ax2, ay - ay2, az - az2, bx, by, bz, cx, cy, cz)

    # do not split in d
    elif 3 * h > 4 * d:
        yield from generate3d(x, y, z, bx2, by2, bz2, cx, cy, cz, ax2, ay2, az2)
        yield from generate3d(x + bx2, y + by2, z + bz2, ax, ay, az, bx - bx2, by - by2, bz - bz2, cx, cy, cz)
        yield from generate3d(
            x + (ax - dax) + (bx2 - dbx),
            y + (ay - day) + (by2 - dby),
            z + (az - daz) + (bz2 - dbz),
            -bx2, -by2, -bz2,
            cx, cy, cz,
            -(ax - ax2), -(ay - ay2), -(az - az2),
        )

    # do not split in h
    elif 3 * d > 4 * h:
        yield from generate3d(x, y, z, cx2, cy2, cz2, ax2, ay2, az2, bx, by, bz)
        yield from generate3d(x + cx2, y + cy2, z + cz2, ax, ay, az, bx, by, bz, cx - cx2, cy - cy2, cz - cz2)
        yield from generate3d(
            x + (ax - dax) + (cx2 - dcx),
            y + (ay - day) + (cy2 - dcy),
            z + (az - daz) + (cz2 - dcz),
            -cx2, -cy2, -cz2,
            -(ax - ax2), -(ay - ay2), -(az - az2),
            bx, by, bz,
        )

    # regular case, split in all w/h/d
    else:
        yield from generate3d(x, y, z, bx2, by2, bz2, cx2, cy2, cz2, ax2, ay2, az2)
        yield from generate3d(x + bx2, y + by2, z + bz2, cx, cy, cz, ax2, ay2, az2, bx - bx2, by - by2, bz - bz2)
        yield from generate3d(
            x + (bx2 - dbx) + (cx - dcx),
            y + (by2 - dby) + (cy - dcy),
            z + (bz2 - dbz) + (cz - dcz),
            ax, ay, az,
            -bx2, -by2, -bz2,
            -(cx - cx2), -(cy - cy2), -(cz - cz2),
        )
        yield from generate3d(
            x + (ax - dax) + bx2 + (cx - dcx),
            y + (ay - day) + by2 + (cy - dcy),
            z + (az - daz) + bz2 + (cz - dcz),
            -cx, -cy, -cz,
            -(ax - ax2), -(ay - ay2), -(az - az2),
            bx - bx2, by - by2, bz - bz2,
        )
        yield from generate3d(
            x + (ax - dax) + (bx2 - dbx),
            y + (ay - day) + (by2 - dby),
            z + (az - daz) + (bz2 - dbz),
            -bx2, -by2, -bz2,
            cx2, cy2, cz2,
            -(ax - ax2), -(ay - ay2), -(az - az2),
        )


def gilbert3d(width: int, height: int, depth: int):
    """Yield (x, y, z) tuples covering width*height*depth in Gilbert order."""
    if width >= height and width >= depth:
        yield from generate3d(0, 0, 0, width, 0, 0, 0, height, 0, 0, 0, depth)
    elif height >= width and height >= depth:
        yield from generate3d(0, 0, 0, 0, height, 0, width, 0, 0, 0, 0, depth)
    else:  # depth >= width and depth >= height
        yield from generate3d(0, 0, 0, 0, 0, depth, width, 0, 0, 0, height, 0)


class GilbertRearranger(nn.Module):
    """Reorder video tokens by 3D Gilbert space-filling curve.

    Text tokens (last `text_length` of seq) are kept in place at the tail.
    Indices are pre-computed once and registered as buffers.

    Args:
        width, height, depth: video latent (W, H, T)
        text_length: number of text tokens at the tail (kept unchanged)
    """

    def __init__(self, width: int, height: int, depth: int, text_length: int):
        super().__init__()
        self.width = width
        self.height = height
        self.depth = depth
        self.text_length = text_length
        self.total_video = width * height * depth

        coord_to_index: dict[int, int] = {}
        gilbert_order = 0
        for x, y, z in gilbert3d(width, height, depth):
            flat = x + width * (y + height * z)
            coord_to_index[flat] = gilbert_order
            gilbert_order += 1

        original2gilbert = torch.empty(self.total_video, dtype=torch.long)
        gilbert2original = torch.empty(self.total_video, dtype=torch.long)
        for orig_flat, gil_idx in coord_to_index.items():
            original2gilbert[orig_flat] = gil_idx
            gilbert2original[gil_idx] = orig_flat

        self.register_buffer("original2gilbert", original2gilbert, persistent=False)
        self.register_buffer("gilbert2original", gilbert2original, persistent=False)

    def rearrange(self, x: torch.Tensor) -> torch.Tensor:
        """Reorder video segment of x along seq_dim=-2.

        Args:
            x: [..., T+L, D] where T = total_video, L = text_length
        Returns:
            same shape, video segment Gilbert-ordered, text segment unchanged
        """
        assert x.shape[-2] == self.total_video + self.text_length, (
            f"expect seq={self.total_video + self.text_length}, got {x.shape[-2]}"
        )
        x_v = x[..., : self.total_video, :]
        x_t = x[..., self.total_video :, :]
        x_v_g = x_v.index_select(-2, self.original2gilbert)
        return torch.cat([x_v_g, x_t], dim=-2)

    def reversed_rearrange(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-2] == self.total_video + self.text_length
        x_v_g = x[..., : self.total_video, :]
        x_t = x[..., self.total_video :, :]
        x_v = x_v_g.index_select(-2, self.gilbert2original)
        return torch.cat([x_v, x_t], dim=-2)


def pad_to_multiple(x: torch.Tensor, multiple: int, dim: int = -2) -> torch.Tensor:
    """Pad tensor along `dim` so its size is a multiple of `multiple`. Pads with zeros."""
    size = x.shape[dim]
    pad_len = (multiple - size % multiple) % multiple
    if pad_len == 0:
        return x
    pad_shape = list(x.shape)
    pad_shape[dim] = pad_len
    pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)


def random_sample_tokens(
    x: torch.Tensor,
    block_size: int,
    num_keep: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample `num_keep` random tokens from each `block_size` block.

    Args:
        x: [B, N, L, D] where L must be a multiple of block_size
        block_size: block length along seq dim
        num_keep: tokens kept per block
    Returns:
        [B, N, (L // block_size) * num_keep, D]
    """
    B, N, L, D = x.shape
    assert L % block_size == 0, f"L={L} not multiple of block_size={block_size}"
    num_blocks = L // block_size
    x_blocks = x.view(B, N, num_blocks, block_size, D)

    # Draw a single set of intra-block positions and reuse it across all blocks
    # (shape [B,N,1,block_size] broadcast over num_blocks). This is a deliberate
    # approximation: independent per-block sampling would require shape
    # [B,N,num_blocks,block_size] of RNG draws, which dominates runtime on long
    # sequences. The block-importance estimate downstream is averaged over
    # heads anyway, so per-block correlated draws preserve the head-mean
    # signal we care about while keeping RNG cost O(block_size).
    rand_vals = torch.rand(B, N, 1, block_size, device=x.device, generator=generator)
    _, idx = torch.topk(rand_vals, num_keep, dim=-1)  # [B,N,1,num_keep]
    idx = idx.expand(-1, -1, num_blocks, -1)  # [B,N,num_blocks,num_keep]
    idx_d = idx.unsqueeze(-1).expand(-1, -1, -1, -1, D)  # [B,N,num_blocks,num_keep,D]
    sampled = torch.gather(x_blocks, 3, idx_d)
    return sampled.reshape(B, N, num_blocks * num_keep, D)


def sample_pool_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    block_size: int,
    num_keep: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Estimate per-block attention importance via NABLA-style mean-pool.

    NABLA reference: Wan2.1-NABLA/wan/modules/attention.py::nablaT (lines 260-270):
        qa = q.reshape(B, h, S//bs, bs, D).mean(-2)
        ka = k.reshape(B, h, S//bs, bs, D).mean(-2).transpose(-2, -1)
        map = softmax((qa @ ka) / sqrt(D), dim=-1)

    Compared to the original sampling-based path: replace per-block random
    token sampling + intra-block max-pool with block mean-pool over the seq
    dim, then a single block-level GEMM + softmax. `num_keep` becomes unused
    but stays in the signature so the call site in `asa_attention` and the
    cfg plumbing don't need to change. NABLA's doc-mask is a multi-segment
    construct; here q/k carry a single (video+text) stream and downstream
    `expand_block_to_token_mask` already forces video<->text to True, so we
    drop the doc-mask term.

    Args:
        q, k: [B, N, L, D]   L need NOT be a multiple of block_size; we pad.
        num_keep: unused under NABLA-style path; kept for signature compat.
    Returns:
        P: [B, 1, nq, nk] head-mean shared block importance (fp32)
        nq = nk = ceil(L / block_size)
    """
    del num_keep, generator  # NABLA-style path is deterministic, no sampling
    B, N, L, D = q.shape
    q_pad = pad_to_multiple(q, block_size, dim=-2)
    k_pad = pad_to_multiple(k, block_size, dim=-2)

    nq = q_pad.shape[-2] // block_size
    nk = k_pad.shape[-2] // block_size

    # Block-level mean pooling along seq dim (NABLA: qa/ka)
    q_blk = q_pad.view(B, N, nq, block_size, D).mean(dim=-2)  # [B,N,nq,D]
    k_blk = k_pad.view(B, N, nk, block_size, D).mean(dim=-2)  # [B,N,nk,D]

    # Block-level small attention in fp32 for numerical stability
    q_blk_f = q_blk.float()
    k_blk_f = k_blk.float()
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q_blk_f, k_blk_f.transpose(-1, -2)) * scale  # [B,N,nq,nk]
    P = torch.softmax(scores, dim=-1)

    # head-mean share (matches downstream H=1 assumption in
    # build_asa_block_mask / expand_block_to_token_mask).
    P = P.mean(dim=1, keepdim=True)  # [B, 1, nq, nk]
    return P


# --- Original sampling-based importance estimator (kept for easy revert) ---
# def sample_pool_attn(
#     q: torch.Tensor,
#     k: torch.Tensor,
#     block_size: int,
#     num_keep: int,
#     generator: torch.Generator | None = None,
# ) -> torch.Tensor:
#     """Estimate per-block attention importance via sampling.
#
#     Args:
#         q, k: [B, N, L, D]   L need NOT be a multiple of block_size; we pad.
#     Returns:
#         P: [B, 1, nq, nk] head-mean shared block importance (fp32)
#         nq = nk = ceil(L / block_size)
#     """
#     B, N, L, D = q.shape
#     q_pad = pad_to_multiple(q, block_size, dim=-2)
#     k_pad = pad_to_multiple(k, block_size, dim=-2)
#
#     q_smp = random_sample_tokens(q_pad, block_size, num_keep, generator)  # [B,N,nq*ns,D]
#     k_smp = random_sample_tokens(k_pad, block_size, num_keep, generator)  # [B,N,nk*ns,D]
#
#     nq = q_pad.shape[-2] // block_size
#     nk = k_pad.shape[-2] // block_size
#     ns = num_keep
#
#     # small attention in fp32 for numerical stability
#     q_smp_f = q_smp.float()
#     k_smp_f = k_smp.float()
#     scale = 1.0 / math.sqrt(D)
#     scores = torch.matmul(q_smp_f, k_smp_f.transpose(-1, -2)) * scale  # [B,N,nq*ns,nk*ns]
#     scores = torch.softmax(scores, dim=-1)
#
#     # block max-pool over within-block dims
#     scores = scores.view(B, N, nq, ns, nk, ns)
#     P = scores.amax(dim=(3, 5))  # [B, N, nq, nk]
#     # head-mean share (see spec §3.2 step 4 / §4.2)
#     P = P.mean(dim=1, keepdim=True)  # [B, 1, nq, nk]
#     return P


_PREFIX_SUM_MATRIX_CACHE: dict[tuple[torch.device, torch.dtype, int], torch.Tensor] = {}


def _get_prefix_sum_matrix(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return upper-triangular ones matrix M[i,j] = 1 if i <= j else 0, cached per (device, dtype, n).

    Used to express prefix-sum as a single GEMM `x @ M` so the reduction runs
    on the NPU cube unit instead of the serial cumsum scan kernel.
    """
    key = (device, dtype, n)
    m = _PREFIX_SUM_MATRIX_CACHE.get(key)
    if m is None:
        m = torch.ones((n, n), device=device, dtype=dtype).triu_()
        _PREFIX_SUM_MATRIX_CACHE[key] = m
    return m


def build_asa_block_mask(
    importance: torch.Tensor,
    max_retain_ratio: float,
    min_retain_ratio: float,
    energy_threshold: float,
) -> torch.Tensor:
    """Energy-threshold pruning of block importance matrix.

    Per row: find smallest k s.t. cumsum of top-k values >= threshold * total,
    clamp k to [min_retain * nk, max_retain * nk], emit mask of positions whose
    importance is at least the k-th-largest value.

    Under exact ties on the threshold value the kept count may exceed
    max_retain * nk; this is intentional. Production importance values come
    from fp32 softmax outputs where exact ties have zero probability, and the
    threshold-compare form lets us drop NPU `scatter_` entirely — the most
    expensive op in the pre-mask pipeline.

    Args:
        importance: [B, 1, nq, nk] block importance (any non-negative dtype; fp32 used internally)
    Returns:
        mask: [B, 1, nq, nk] bool, True = keep block
    """
    B, H, nq, nk = importance.shape
    imp_f = importance if importance.dtype == torch.float32 else importance.float()

    min_keep = max(1, int(nk * min_retain_ratio))
    max_keep = max(min_keep, int(nk * max_retain_ratio))

    # Cap at max_keep: max_retain_ratio bounds the kept block count, so topk
    # on max_keep is enough to recover both the energy-threshold cutoff and
    # the per-row threshold value. Total energy is a single reduce instead of
    # reading the tail of a full-length cumsum.
    topk_vals, _ = torch.topk(imp_f, max_keep, dim=-1, sorted=True)
    total = imp_f.sum(dim=-1, keepdim=True).clamp(min=1e-30)

    # cumsum-as-matmul: `cum[..., j] = sum_{i<=j} topk_vals[..., i]` is exactly
    # `topk_vals @ M` where M is upper-triangular ones. NPU cumsum is a serial
    # scan kernel that costs ms even on short rows; matmul runs on the cube
    # unit and is two orders of magnitude faster for max_keep in the hundreds.
    # The triangular matrix is shape-only state, cached across denoise steps.
    prefix_m = _get_prefix_sum_matrix(max_keep, imp_f.device, imp_f.dtype)
    cum = torch.matmul(topk_vals, prefix_m)

    # k = smallest count whose cumulative energy reaches the threshold.
    # `(cum < need).sum() + 1` collapses argmax+any+where into one reduce;
    # if cum never reaches need within max_keep, the count saturates at
    # max_keep+1 and the clamp pulls it back, matching the original
    # `unsatisfied -> nk -> clamp` fallback.
    need = energy_threshold * total
    k_count = (cum < need).sum(dim=-1, keepdim=True) + 1
    k_count = k_count.clamp(min=min_keep, max=max_keep)

    # The k_count-th largest value per row is the keep threshold. Elementwise
    # `imp >= threshold` avoids scatter_ entirely; for fp32 softmax inputs the
    # kept count equals k_count exactly (no ties at threshold).
    threshold = topk_vals.gather(-1, k_count - 1)
    return imp_f >= threshold


def expand_block_to_token_mask(
    m_block: torch.Tensor,
    block_size: int,
    t_len: int,
    l_len: int,
) -> torch.Tensor:
    """Expand block mask to token-level dense mask.

    Args:
        m_block: [B, 1, nq, nk] bool, where nq = nk = ceil((T_pad)/block_size).
                 The mask covers the padded video length; rows/cols beyond t_len are
                 truncated. Only the video x video sub-region is constrained;
                 video x text, text x video, and text x text are forced to True.
        block_size: block length used when building m_block.
        t_len: video sequence length.
        l_len: text sequence length.
    Returns:
        m_token: [B, 1, t_len+l_len, t_len+l_len] bool, True = keep.
    """
    B, H, nq, nk = m_block.shape
    assert H == 1, f"head-mean shared mask expected, got H={H}"

    vv = (
        m_block
        .unsqueeze(-2)
        .unsqueeze(-1)
        .expand(B, H, nq, block_size, nk, block_size)
        .reshape(B, H, nq * block_size, nk * block_size)
    )

    m_token = torch.ones((B, 1, t_len + l_len, t_len + l_len), dtype=torch.bool, device=m_block.device)
    m_token[..., :t_len, :t_len] = vv[..., :t_len, :t_len]

    return m_token


def asa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cfg: "AsaConfig",
    rearranger: GilbertRearranger,
    t_len: int,
    l_len: int,
    fa_full_dense,
    generator: torch.Generator | None = None,
    sta_cache: "StaMaskCache | None" = None,
) -> torch.Tensor:
    """ASA forward (A subphase: variant in {'dense_probe', 'asa'}).

    Args:
        q, k, v: [B, N, T+L, D] (BNSD)
        cfg: AsaConfig with enable=True and variant in {'dense_probe', 'asa'}
        rearranger: pre-built GilbertRearranger matching (W,H,T) and L
        t_len, l_len: video and text lengths
        fa_full_dense: callable (q, k, v, atten_mask) -> out [B,N,T+L,D] using
                       npu_fusion_attention; supplied by caller to keep this
                       module free of torch_npu coupling for unit tests.
        generator: optional torch.Generator for deterministic sampling
        sta_cache: optional StaMaskCache. Required when cfg.sta_enable=True
                   and cfg.variant == 'asa'.
    Returns:
        out: [B, N, T+L, D]
    """
    assert cfg.enable
    assert cfg.variant in ("dense_probe", "asa"), (
        f"asa_attention only supports A-subphase variants, got {cfg.variant}"
    )

    # 1. Gilbert rearrange (video segment only)
    if cfg.use_gilbert:
        q_g = rearranger.rearrange(q)
        k_g = rearranger.rearrange(k)
        v_g = rearranger.rearrange(v)
    else:
        q_g, k_g, v_g = q, k, v

    if cfg.variant == "dense_probe":
        if cfg.sta_enable:
            log.info("sta_enable ignored under variant='dense_probe' (no-op OR)")
        m_token = None  # treat as full attention
    else:
        # 2. block importance estimation (head-mean shared)
        with torch.no_grad():
            imp = sample_pool_attn(q_g, k_g, cfg.block_size, cfg.num_keep, generator)
            # 3. energy-threshold mask
            m_block = build_asa_block_mask(imp, cfg.max_retain_ratio, cfg.min_retain_ratio, cfg.energy_threshold)
            # 4. OR-combine with STA block mask when enabled
            if cfg.sta_enable:
                if sta_cache is None:
                    raise ValueError(
                        "asa_attention: sta_cache must be provided when "
                        "cfg.sta_enable=True"
                    )
                from .sta import validate_and_normalize_window
                T_grid, H_grid, W_grid = rearranger.depth, rearranger.height, rearranger.width
                normalized = validate_and_normalize_window(
                    cfg.sta_window, grid=(T_grid, H_grid, W_grid),
                )
                if normalized != cfg.sta_window:
                    log.warning(
                        "STA window %r normalized to %r (grid=(T=%d,H=%d,W=%d))",
                        cfg.sta_window, normalized, T_grid, H_grid, W_grid,
                    )
                B, _, nq, nk = m_block.shape
                m_sta = sta_cache.get_or_build(
                    T=T_grid, H=H_grid, W=W_grid,
                    block_size=cfg.block_size,
                    n_blocks_total=nq,
                    n_video_tokens=rearranger.total_video,
                    gilbert2original=rearranger.gilbert2original,
                    window=normalized,
                    device=m_block.device,
                )
                m_block = m_block | m_sta
            # 5. expand to dense token mask (video x video only)
            m_token = expand_block_to_token_mask(m_block, cfg.block_size, t_len, l_len)

    # 6. flash attention with dense atten_mask (or None for dense_probe)
    out_g = fa_full_dense(q_g, k_g, v_g, m_token)

    # 7. inverse Gilbert
    if cfg.use_gilbert:
        out = rearranger.reversed_rearrange(out_g)
    else:
        out = out_g
    return out
