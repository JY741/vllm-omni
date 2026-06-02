# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for BLADE ASA quality-probe block-sparse mask.

Run: pytest tests/mgm_video/test_mmdit_asa.py -v
"""

import torch  # noqa: F401  # retained for upcoming tensor-based ASA tests

from vllm_omni.diffusion.models.mgm_video.mmdit.mmdit_asa import (
    AsaConfig,
    _parse_range,
    should_apply_asa,
)


def test_parse_range_all():
    assert _parse_range("all") is None


def test_parse_range_span():
    assert _parse_range("8-11") == {8, 9, 10, 11}


def test_parse_range_single():
    assert _parse_range("5") == {5}


def test_should_apply_respects_enable():
    cfg = AsaConfig(enable=False, tau=0.95, block=128, layers="all",
                    steps="all", per_head=True, log=False)
    assert should_apply_asa(cfg, layer_idx=0, step_idx=0, downscale=1) is False


def test_should_apply_skiparse_mutex():
    cfg = AsaConfig(enable=True, tau=0.95, block=128, layers="all",
                    steps="all", per_head=True, log=False)
    assert should_apply_asa(cfg, layer_idx=0, step_idx=0, downscale=2) is False


def test_should_apply_layer_step_scope():
    cfg = AsaConfig(enable=True, tau=0.95, block=128, layers="8-41",
                    steps="2-7", per_head=True, log=False)
    assert should_apply_asa(cfg, layer_idx=0, step_idx=5, downscale=1) is False
    assert should_apply_asa(cfg, layer_idx=10, step_idx=0, downscale=1) is False
    assert should_apply_asa(cfg, layer_idx=10, step_idx=5, downscale=1) is True


def test_should_apply_step_none_passes():
    cfg = AsaConfig(enable=True, tau=0.95, block=128, layers="all",
                    steps="2-7", per_head=True, log=False)
    # step unknown (cache disabled) -> step scope not enforced
    assert should_apply_asa(cfg, layer_idx=0, step_idx=None, downscale=1) is True


from vllm_omni.diffusion.models.mgm_video.mmdit.mmdit_asa import (  # noqa: E402
    _block_importance_keep,
)


def _reference_block_keep(q_v, k_v, block, tau, scale):
    """Naive loop reference of the SAME mean-pool block-importance algorithm."""
    Sv, D = q_v.shape
    nB = (Sv + block - 1) // block
    pad = nB * block - Sv
    if pad:
        q_v = torch.cat([q_v, torch.zeros(pad, D, dtype=q_v.dtype)], 0)
        k_v = torch.cat([k_v, torch.zeros(pad, D, dtype=k_v.dtype)], 0)
    k_blk = torch.stack([k_v[i * block:(i + 1) * block].mean(0) for i in range(nB)])
    keep = torch.zeros(nB, nB, dtype=torch.bool)
    for i in range(nB):
        qi = q_v[i * block:(i + 1) * block]
        s_blk = ((qi @ k_blk.t()) * scale).amax(0)
        p = torch.softmax(s_blk.float(), -1)
        order = torch.argsort(p, descending=True)
        acc = 0.0
        for j in order.tolist():
            keep[i, j] = True
            acc += p[j].item()
            if acc >= tau:
                break
        keep[i, i] = True
    return keep


def test_block_importance_diagonal_self_select():
    torch.manual_seed(0)
    q = torch.randn(256, 16)
    k = torch.randn(256, 16)
    keep, nB, pad = _block_importance_keep(q, k, block=64, tau=0.5, scale=16 ** -0.5)
    assert nB == 4 and pad == 0
    assert bool(keep.diagonal().all())


def test_block_importance_tau_monotonic():
    torch.manual_seed(1)
    q = torch.randn(256, 16)
    k = torch.randn(256, 16)
    keep_low, _, _ = _block_importance_keep(q, k, block=64, tau=0.50, scale=16 ** -0.5)
    keep_high, _, _ = _block_importance_keep(q, k, block=64, tau=0.99, scale=16 ** -0.5)
    # higher tau keeps >= blocks
    assert int(keep_high.sum()) >= int(keep_low.sum())


def test_block_importance_matches_reference():
    torch.manual_seed(2)
    q = torch.randn(320, 16)  # non-divisible -> exercises padding (nB=5, pad=0 for 320/64)
    k = torch.randn(320, 16)
    keep, _, _ = _block_importance_keep(q, k, block=64, tau=0.9, scale=16 ** -0.5)
    ref = _reference_block_keep(q, k, block=64, tau=0.9, scale=16 ** -0.5)
    assert torch.equal(keep, ref)


def test_block_importance_padding():
    torch.manual_seed(3)
    q = torch.randn(200, 16)  # 200/64 -> nB=4, pad=56
    k = torch.randn(200, 16)
    keep, nB, pad = _block_importance_keep(q, k, block=64, tau=0.9, scale=16 ** -0.5)
    assert nB == 4 and pad == 56
    ref = _reference_block_keep(q, k, block=64, tau=0.9, scale=16 ** -0.5)
    assert torch.equal(keep, ref)


from vllm_omni.diffusion.models.mgm_video.mmdit.mmdit_asa import (  # noqa: E402
    build_asa_block_mask,
)


def test_build_mask_shape_and_dtype():
    torch.manual_seed(4)
    T, L, D = 256, 32, 16
    S = T + L
    q = torch.randn(1, 1, S, D)
    k = torch.randn(1, 1, S, D)
    m = build_asa_block_mask(q, k, T, L, block=64, tau=0.9, scale=D ** -0.5)
    assert m.shape == (1, 1, S, S)
    assert m.dtype == torch.bool


def test_build_mask_text_is_dense():
    torch.manual_seed(5)
    T, L, D = 256, 32, 16
    S = T + L
    q = torch.randn(1, 1, S, D)
    k = torch.randn(1, 1, S, D)
    m = build_asa_block_mask(q, k, T, L, block=64, tau=0.9, scale=D ** -0.5)
    # True = masked. Text rows and text cols must be fully dense (never masked).
    assert not bool(m[0, 0, T:, :].any())   # text query rows
    assert not bool(m[0, 0, :, T:].any())   # text key cols


def test_build_mask_diagonal_video_unmasked():
    torch.manual_seed(6)
    T, L, D, block = 256, 32, 16, 64
    S = T + L
    q = torch.randn(1, 1, S, D)
    k = torch.randn(1, 1, S, D)
    m = build_asa_block_mask(q, k, T, L, block=block, tau=0.5, scale=D ** -0.5)
    # diagonal video blocks self-select -> their on-diagonal token region unmasked
    for i in range(T // block):
        s = i * block
        assert not bool(m[0, 0, s:s + block, s:s + block].any())


def test_build_mask_higher_tau_less_masked():
    torch.manual_seed(7)
    T, L, D = 256, 32, 16
    S = T + L
    q = torch.randn(1, 1, S, D)
    k = torch.randn(1, 1, S, D)
    m_low = build_asa_block_mask(q, k, T, L, block=64, tau=0.50, scale=D ** -0.5)
    m_high = build_asa_block_mask(q, k, T, L, block=64, tau=0.99, scale=D ** -0.5)
    assert int(m_high.sum()) <= int(m_low.sum())


def test_build_mask_fullp_sanity():
    """Soft check: ASA-kept blocks should cover most of the true full-P mass."""
    torch.manual_seed(8)
    T, L, D, block = 256, 0, 16, 64
    q = torch.randn(1, 1, T, D)
    k = torch.randn(1, 1, T, D)
    scale = D ** -0.5
    m = build_asa_block_mask(q, k, T, L, block=block, tau=0.9, scale=scale)
    keep = ~m[0, 0, :T, :T]
    # true post-softmax attention probabilities
    p_full = torch.softmax((q[0, 0] @ k[0, 0].t()) * scale, dim=-1)
    covered = (p_full * keep.float()).sum(-1)  # per query row
    # kept blocks should cover a large majority of probability mass on average
    assert covered.mean().item() > 0.7
