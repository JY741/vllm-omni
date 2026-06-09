# SPDX-License-Identifier: Apache-2.0
"""Sliding Tile Attention (STA) block-level mask for mgm_video MMDiT.

Hybrid path: M = M_asa OR M_sta at the block-mask level inside asa_attention.
See docs/superpowers/specs/2026-06-08-mgm-video-sta-hybrid-design.md.
"""

import torch

_SENTINEL_HI = 2**30


def build_sta_block_mask_in_gilbert_order(
    T: int,
    H: int,
    W: int,
    block_size: int,
    n_blocks_total: int,
    n_video_tokens: int,
    gilbert2original: torch.Tensor,
    window: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """Build per-block AABB-within-window boolean mask for STA.

    Args:
        T, H, W: video latent grid (depth, height, width).
        block_size: tokens per block (must match ASA block_size).
        n_blocks_total: total blocks including trailing text blocks.
        n_video_tokens: actual video token count (T*H*W; pre-padding).
        gilbert2original: [n_video_tokens] long tensor mapping Gilbert
            block-flattened position -> original (T,H,W) flat index. The
            same buffer used by GilbertRearranger.
        window: (wT, wH, wW), full-extent window. Half-extent on each
            axis is w // 2.
        device: target device for the output mask.

    Returns:
        mask: [1, 1, n_blocks_total, n_blocks_total] bool. Trailing
              text-block rows/cols are True.
    """
    wT, wH, wW = window
    half_t, half_h, half_w = wT // 2, wH // 2, wW // 2
    nq_video = (n_video_tokens + block_size - 1) // block_size
    n_pad = nq_video * block_size

    g2o = gilbert2original.to(device=device, dtype=torch.long)
    t_coord = (g2o // (H * W)).long()
    h_coord = ((g2o // W) % H).long()
    w_coord = (g2o % W).long()

    # When n_pad == n_video_tokens (block_size divides evenly), no padding
    # needed; the view below over t_coord/h_coord/w_coord is already exact.
    if n_pad > n_video_tokens:
        n_extra = n_pad - n_video_tokens
        pad_hi = torch.full((n_extra,), _SENTINEL_HI, dtype=torch.long, device=device)
        t_coord = torch.cat([t_coord, pad_hi])
        h_coord = torch.cat([h_coord, pad_hi])
        w_coord = torch.cat([w_coord, pad_hi])

    t_blk = t_coord.view(nq_video, block_size)
    h_blk = h_coord.view(nq_video, block_size)
    w_blk = w_coord.view(nq_video, block_size)
    # Real coords are <= max(T,H,W); _SENTINEL_HI dwarfs them, so it does
    # not become min unless every entry is sentinel (fully-padded block,
    # which we want to never connect to real blocks).
    t_min, t_max = t_blk.min(dim=-1).values, t_blk.max(dim=-1).values
    h_min, h_max = h_blk.min(dim=-1).values, h_blk.max(dim=-1).values
    w_min, w_max = w_blk.min(dim=-1).values, w_blk.max(dim=-1).values
    # When a block is fully padded, both min and max are _SENTINEL_HI.
    # The AABB test below produces gap=0 vs another all-sentinel block, but
    # since real blocks have coords << _SENTINEL_HI the gap to any real
    # block is huge, blocking spurious edges.

    def _within(
        a_min: torch.Tensor,
        a_max: torch.Tensor,
        b_min: torch.Tensor,
        b_max: torch.Tensor,
        half: int,
    ) -> torch.Tensor:
        gap = torch.maximum(
            a_min.unsqueeze(1) - b_max.unsqueeze(0),
            b_min.unsqueeze(0) - a_max.unsqueeze(1),
        )
        gap = gap.clamp(min=0)
        return gap <= half

    in_t = _within(t_min, t_max, t_min, t_max, half_t)
    in_h = _within(h_min, h_max, h_min, h_max, half_h)
    in_w = _within(w_min, w_max, w_min, w_max, half_w)
    video_block_mask = in_t & in_h & in_w

    full = torch.ones((n_blocks_total, n_blocks_total), dtype=torch.bool, device=device)
    full[:nq_video, :nq_video] = video_block_mask
    return full.unsqueeze(0).unsqueeze(0)


def validate_and_normalize_window(
    window: tuple[int, int, int],
    grid: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Clamp each window axis to grid dim (odd) and round even values up.

    Args:
        window: (wT, wH, wW) requested.
        grid:   (T, H, W) latent grid.
    Returns:
        normalized window with each axis odd, positive, and <= grid axis.
    """
    out = []
    for w, g in zip(window, grid, strict=True):
        if w <= 0:
            w = 1
        if w % 2 == 0:
            w = w + 1
        # Largest odd <= g
        max_odd = g if g % 2 == 1 else g - 1
        if max_odd < 1:
            max_odd = 1
        if w > max_odd:
            w = max_odd
        out.append(int(w))
    return tuple(out)  # type: ignore[return-value]


class StaMaskCache:
    """Per-shape cache for STA block masks.

    Keyed by (T, H, W, block_size, window, n_blocks_total, n_video_tokens,
    id(gilbert2original)). Holds at most one entry; a shape change evicts.
    """

    def __init__(self):
        self._key = None
        self._mask: torch.Tensor | None = None

    def get_or_build(
        self,
        T: int,
        H: int,
        W: int,
        block_size: int,
        n_blocks_total: int,
        n_video_tokens: int,
        gilbert2original: torch.Tensor,
        window: tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        key = (
            T, H, W, block_size, n_blocks_total, n_video_tokens,
            tuple(window), id(gilbert2original), str(device),
        )
        if self._key == key and self._mask is not None:
            return self._mask
        self._mask = build_sta_block_mask_in_gilbert_order(
            T=T, H=H, W=W, block_size=block_size,
            n_blocks_total=n_blocks_total, n_video_tokens=n_video_tokens,
            gilbert2original=gilbert2original, window=window, device=device,
        )
        self._key = key
        return self._mask
