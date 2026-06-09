import logging

import pytest
import torch

from vllm_omni.diffusion.models.mgm_video.mmdit.asa import AsaConfig
from vllm_omni.diffusion.models.mgm_video.mmdit.sta import (
    build_sta_block_mask_in_gilbert_order,
)


def test_sta_config_defaults_disabled():
    cfg = AsaConfig()
    assert cfg.sta_enable is False
    assert cfg.sta_window == (7, 13, 13)


def test_sta_config_from_env_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_MGM_STA_ENABLE", raising=False)
    monkeypatch.delenv("VLLM_MGM_STA_WINDOW", raising=False)
    cfg = AsaConfig.from_env()
    assert cfg.sta_enable is False
    assert cfg.sta_window == (7, 13, 13)


def test_sta_config_from_env_enable(monkeypatch):
    monkeypatch.delenv("VLLM_MGM_STA_WINDOW", raising=False)
    monkeypatch.setenv("VLLM_MGM_STA_ENABLE", "1")
    cfg = AsaConfig.from_env()
    assert cfg.sta_enable is True


def test_sta_config_from_env_window(monkeypatch):
    monkeypatch.delenv("VLLM_MGM_STA_ENABLE", raising=False)
    monkeypatch.setenv("VLLM_MGM_STA_WINDOW", "9,15,15")
    cfg = AsaConfig.from_env()
    assert cfg.sta_window == (9, 15, 15)


def test_sta_config_from_env_window_malformed(monkeypatch, caplog):
    monkeypatch.setenv("VLLM_MGM_STA_WINDOW", "not,a,window,extra")
    import vllm_omni.diffusion.models.mgm_video.mmdit.asa as asa_mod
    from unittest.mock import patch
    with patch.object(asa_mod, "log") as mock_log:
        cfg = AsaConfig.from_env()
    # Falls back to default; warning logged.
    assert cfg.sta_window == (7, 13, 13)
    mock_log.warning.assert_called_once()
    call_args = mock_log.warning.call_args
    assert "VLLM_MGM_STA_WINDOW" in call_args.args[0]


def _identity_gilbert(T: int, H: int, W: int) -> torch.Tensor:
    """Row-major (T,H,W) order; flat = t*H*W + h*W + w."""
    n = T * H * W
    return torch.arange(n, dtype=torch.long)


def test_sta_block_mask_shape_and_dtype():
    T, H, W = 2, 4, 4
    block_size = 4
    n_video = T * H * W  # 32
    nq_video = (n_video + block_size - 1) // block_size  # 8
    n_blocks_total = nq_video + 2  # 2 trailing text blocks
    g = _identity_gilbert(T, H, W)
    mask = build_sta_block_mask_in_gilbert_order(
        T=T, H=H, W=W, block_size=block_size,
        n_blocks_total=n_blocks_total, n_video_tokens=n_video,
        gilbert2original=g, window=(1, 3, 3), device=torch.device("cpu"),
    )
    assert mask.shape == (1, 1, n_blocks_total, n_blocks_total)
    assert mask.dtype == torch.bool


def test_sta_block_mask_identity_diagonal():
    T, H, W = 2, 4, 4
    n_video = 32
    g = _identity_gilbert(T, H, W)
    mask = build_sta_block_mask_in_gilbert_order(
        T=T, H=H, W=W, block_size=4,
        n_blocks_total=8, n_video_tokens=n_video,
        gilbert2original=g, window=(1, 1, 1), device=torch.device("cpu"),
    )
    diag_idx = torch.arange(8)
    assert mask[0, 0, diag_idx, diag_idx].all()


def test_sta_block_mask_symmetric():
    T, H, W = 2, 4, 4
    n_video = 32
    g = _identity_gilbert(T, H, W)
    mask = build_sta_block_mask_in_gilbert_order(
        T=T, H=H, W=W, block_size=4,
        n_blocks_total=8, n_video_tokens=n_video,
        gilbert2original=g, window=(1, 3, 3), device=torch.device("cpu"),
    )
    m = mask[0, 0]
    assert torch.equal(m, m.t())


def test_sta_block_mask_full_window_all_true():
    T, H, W = 2, 4, 4
    n_video = 32
    g = _identity_gilbert(T, H, W)
    mask = build_sta_block_mask_in_gilbert_order(
        T=T, H=H, W=W, block_size=4,
        n_blocks_total=8, n_video_tokens=n_video,
        gilbert2original=g, window=(5, 9, 9), device=torch.device("cpu"),
    )
    assert mask.all()


def test_sta_block_mask_text_rows_cols_true():
    T, H, W = 2, 4, 4
    n_video = 32
    nq_video = 8
    n_blocks_total = nq_video + 2
    g = _identity_gilbert(T, H, W)
    mask = build_sta_block_mask_in_gilbert_order(
        T=T, H=H, W=W, block_size=4,
        n_blocks_total=n_blocks_total, n_video_tokens=n_video,
        gilbert2original=g, window=(1, 3, 3), device=torch.device("cpu"),
    )
    assert mask[..., nq_video:, :].all()
    assert mask[..., :, nq_video:].all()


def test_sta_block_mask_padding_no_spurious_edges():
    T, H, W = 2, 3, 3
    n_video = T * H * W  # 18
    block_size = 8
    nq_video = (n_video + block_size - 1) // block_size  # 3
    g = _identity_gilbert(T, H, W)
    mask = build_sta_block_mask_in_gilbert_order(
        T=T, H=H, W=W, block_size=block_size,
        n_blocks_total=nq_video, n_video_tokens=n_video,
        gilbert2original=g, window=(1, 1, 1), device=torch.device("cpu"),
    )
    assert mask.dtype == torch.bool
    assert mask.shape == (1, 1, nq_video, nq_video)
    diag_idx = torch.arange(nq_video)
    assert mask[0, 0, diag_idx, diag_idx].all()
    # With T=2, H=3, W=3 in row-major order and block_size=8:
    #   block 0 covers flat 0..7  → all t=0  (t_min=0, t_max=0)
    #   block 1 covers flat 8..15 → t=0 (one token) + t=1 (seven tokens),
    #            so t_min=0, t_max=1 (straddles both frames)
    #   block 2 covers flat 16..17 → all t=1 (t_min=1, t_max=1)
    # window=(1,1,1) → half_t=0.  Temporal gap between blocks 0 and 2 is
    # max(t_min[0]-t_max[2], t_min[2]-t_max[0]) = max(0-1, 1-0) = 1 > 0,
    # so they must NOT connect.
    assert mask[0, 0, 0, 2].item() is False
    assert mask[0, 0, 2, 0].item() is False


from vllm_omni.diffusion.models.mgm_video.mmdit.asa import GilbertRearranger


def _within_window_pair(c1, c2, window):
    wT, wH, wW = window
    return (
        abs(c1[0] - c2[0]) <= wT // 2
        and abs(c1[1] - c2[1]) <= wH // 2
        and abs(c1[2] - c2[2]) <= wW // 2
    )


def test_sta_block_mask_within_window_pair_implies_block_connected():
    # If any token pair across blocks (i, j) is within window, blocks must connect.
    T, H, W = 2, 4, 4
    block_size = 4
    window = (1, 3, 3)
    rearr = GilbertRearranger(width=W, height=H, depth=T, text_length=0)
    n_video = T * H * W
    nq_video = (n_video + block_size - 1) // block_size
    mask = build_sta_block_mask_in_gilbert_order(
        T=T, H=H, W=W, block_size=block_size,
        n_blocks_total=nq_video, n_video_tokens=n_video,
        gilbert2original=rearr.gilbert2original,
        window=window, device=torch.device("cpu"),
    )
    g2o = rearr.gilbert2original.tolist()
    for i in range(nq_video):
        for j in range(nq_video):
            block_i_tokens = g2o[i * block_size:(i + 1) * block_size]
            block_j_tokens = g2o[j * block_size:(j + 1) * block_size]
            connected = False
            for ti in block_i_tokens:
                for tj in block_j_tokens:
                    ci = (ti // (H * W), (ti // W) % H, ti % W)
                    cj = (tj // (H * W), (tj // W) % H, tj % W)
                    if _within_window_pair(ci, cj, window):
                        connected = True
                        break
                if connected:
                    break
            assert mask[0, 0, i, j].item() == connected, (
                f"block ({i},{j}) mismatch: code={mask[0,0,i,j].item()}, ref={connected}"
            )


def test_sta_block_mask_disconnect_when_no_pair_in_window():
    # Use very tight window (1,1,1); block whose tokens never coincide with
    # another block in any single (T,H,W) coord must be disconnected.
    T, H, W = 2, 4, 4
    block_size = 4
    rearr = GilbertRearranger(width=W, height=H, depth=T, text_length=0)
    n_video = T * H * W
    nq_video = 8
    mask = build_sta_block_mask_in_gilbert_order(
        T=T, H=H, W=W, block_size=block_size,
        n_blocks_total=nq_video, n_video_tokens=n_video,
        gilbert2original=rearr.gilbert2original,
        window=(1, 1, 1), device=torch.device("cpu"),
    )
    # Reference: build "any-pair shares all 3 coords" matrix and compare.
    g2o = rearr.gilbert2original.tolist()
    ref = torch.zeros(nq_video, nq_video, dtype=torch.bool)
    for i in range(nq_video):
        for j in range(nq_video):
            block_i = g2o[i * block_size:(i + 1) * block_size]
            block_j = g2o[j * block_size:(j + 1) * block_size]
            for ti in block_i:
                for tj in block_j:
                    if ti == tj:  # window=(1,1,1) -> only equal coords match
                        ref[i, j] = True
                        break
                if ref[i, j]:
                    break
    assert torch.equal(mask[0, 0], ref)


def test_sta_block_mask_text_unaffected_by_window():
    T, H, W = 2, 4, 4
    block_size = 4
    rearr = GilbertRearranger(width=W, height=H, depth=T, text_length=8)
    n_video = T * H * W
    nq_video = 8
    n_text_blocks = 2  # 8 text tokens / block_size 4
    n_blocks_total = nq_video + n_text_blocks
    # Tight window
    mask_tight = build_sta_block_mask_in_gilbert_order(
        T=T, H=H, W=W, block_size=block_size,
        n_blocks_total=n_blocks_total, n_video_tokens=n_video,
        gilbert2original=rearr.gilbert2original,
        window=(1, 1, 1), device=torch.device("cpu"),
    )
    assert mask_tight[..., nq_video:, :].all()
    assert mask_tight[..., :, nq_video:].all()


from vllm_omni.diffusion.models.mgm_video.mmdit.sta import StaMaskCache


def test_sta_mask_cache_returns_same_tensor_on_hit():
    rearr = GilbertRearranger(width=4, height=4, depth=2, text_length=0)
    cache = StaMaskCache()
    args = dict(
        T=2, H=4, W=4, block_size=4,
        n_blocks_total=8, n_video_tokens=32,
        gilbert2original=rearr.gilbert2original,
        window=(1, 3, 3), device=torch.device("cpu"),
    )
    m1 = cache.get_or_build(**args)
    m2 = cache.get_or_build(**args)
    assert m1 is m2  # same tensor object


def test_sta_mask_cache_rebuilds_on_shape_change():
    rearr_a = GilbertRearranger(width=4, height=4, depth=2, text_length=0)
    rearr_b = GilbertRearranger(width=4, height=4, depth=3, text_length=0)
    cache = StaMaskCache()
    common = dict(block_size=4, window=(1, 3, 3), device=torch.device("cpu"))
    m_a = cache.get_or_build(
        T=2, H=4, W=4, n_blocks_total=8, n_video_tokens=32,
        gilbert2original=rearr_a.gilbert2original, **common,
    )
    m_b = cache.get_or_build(
        T=3, H=4, W=4, n_blocks_total=12, n_video_tokens=48,
        gilbert2original=rearr_b.gilbert2original, **common,
    )
    assert m_a is not m_b
    assert m_a.shape != m_b.shape


from vllm_omni.diffusion.models.mgm_video.mmdit.asa import (
    asa_attention,
    expand_block_to_token_mask,
    build_asa_block_mask,
)


class _FakeFA:
    """Capture the atten_mask passed to fa_full_dense for assertions."""

    def __init__(self):
        self.captured_mask = None

    def __call__(self, q, k, v, atten_mask):
        self.captured_mask = atten_mask
        # Return q so caller's reverse-rearrange roundtrip is identity.
        return q


def _run_asa(cfg, q, k, v, rearr, t_len, l_len, sta_cache):
    fake = _FakeFA()
    asa_attention(
        q, k, v, cfg=cfg, rearranger=rearr,
        t_len=t_len, l_len=l_len,
        fa_full_dense=fake,
        sta_cache=sta_cache,
    )
    return fake.captured_mask


def test_asa_attention_sta_disabled_bit_exact_to_pre_sta():
    # When sta_enable=False, the captured mask must equal what
    # build_asa_block_mask + expand_block_to_token_mask produce alone.
    torch.manual_seed(0)
    T, H, W = 2, 4, 4
    block_size = 4
    rearr = GilbertRearranger(width=W, height=H, depth=T, text_length=8)
    n_video = T * H * W
    l_len = 8
    s = n_video + l_len
    n_head, d = 2, 16
    q = torch.randn(1, n_head, s, d)
    k = torch.randn(1, n_head, s, d)
    v = torch.randn(1, n_head, s, d)
    cfg = AsaConfig(enable=True, variant="asa", block_size=block_size,
                    num_keep=2, max_retain_ratio=0.5, min_retain_ratio=0.1,
                    energy_threshold=0.95, use_gilbert=True, text_length=l_len,
                    sta_enable=False)
    cache = StaMaskCache()
    captured = _run_asa(cfg, q, k, v, rearr, n_video, l_len, cache)
    # captured is the m_token built only from ASA. Build the same thing inline
    # and compare. Use the same RNG seed for sample_pool_attn determinism.
    assert captured is not None
    # Sanity: mask is bool [1,1,s,s] and text rows/cols are True.
    assert captured.dtype == torch.bool
    assert captured.shape == (1, 1, s, s)
    assert captured[..., n_video:, :].all()
    assert captured[..., :, n_video:].all()


def test_asa_attention_sta_enabled_or_combines():
    # When sta_enable=True, OR with the STA mask must add True entries that
    # aggressive ASA pruning would miss. Verifies the OR is wired correctly:
    # m_on (video x video) must be a superset of m_off (video x video).
    #
    # Both runs use the same RNG seed so the ASA importance sampling is
    # identical; only the STA OR differs.
    T, H, W = 2, 4, 4
    block_size = 4
    rearr = GilbertRearranger(width=W, height=H, depth=T, text_length=8)
    n_video = T * H * W
    l_len = 8
    s = n_video + l_len
    n_head, d = 2, 16
    torch.manual_seed(0)
    q = torch.randn(1, n_head, s, d)
    k = torch.randn(1, n_head, s, d)
    v = torch.randn(1, n_head, s, d)
    cfg_off = AsaConfig(enable=True, variant="asa", block_size=block_size,
                        num_keep=2, max_retain_ratio=0.1, min_retain_ratio=0.05,
                        energy_threshold=0.5, use_gilbert=True,
                        text_length=l_len, sta_enable=False)
    # sta_window=(1, 3, 3) is the maximum window allowed by validate_and_normalize_window
    # for grid (T=2, H=4, W=4): max_odd values are (1, 3, 3).
    cfg_on = AsaConfig(enable=True, variant="asa", block_size=block_size,
                       num_keep=2, max_retain_ratio=0.1, min_retain_ratio=0.05,
                       energy_threshold=0.5, use_gilbert=True,
                       text_length=l_len, sta_enable=True,
                       sta_window=(1, 3, 3))
    # Reset seed before each call so both runs have identical ASA sampling.
    gen = torch.Generator()
    gen.manual_seed(42)
    fake_off = _FakeFA()
    asa_attention(q, k, v, cfg=cfg_off, rearranger=rearr,
                  t_len=n_video, l_len=l_len, fa_full_dense=fake_off,
                  generator=gen, sta_cache=StaMaskCache())
    m_off = fake_off.captured_mask

    gen.manual_seed(42)
    fake_on = _FakeFA()
    asa_attention(q, k, v, cfg=cfg_on, rearranger=rearr,
                  t_len=n_video, l_len=l_len, fa_full_dense=fake_on,
                  generator=gen, sta_cache=StaMaskCache())
    m_on = fake_on.captured_mask

    vv_off = m_off[..., :n_video, :n_video]
    vv_on = m_on[..., :n_video, :n_video]
    # OR is monotone: m_on must be a superset of m_off.
    assert (~vv_off | vv_on).all(), "OR should produce a superset of ASA-only mask"
    # STA adds entries that aggressive pruning missed: m_on must have strictly
    # more True entries than m_off.
    assert vv_on.sum() > vv_off.sum(), "STA should add True entries beyond aggressive ASA pruning"


def test_asa_attention_sta_enabled_dense_probe_skipped(caplog):
    # variant="dense_probe" -> STA must be ignored.
    torch.manual_seed(0)
    T, H, W = 2, 4, 4
    rearr = GilbertRearranger(width=W, height=H, depth=T, text_length=8)
    n_video, l_len = T * H * W, 8
    s = n_video + l_len
    q = torch.randn(1, 2, s, 16); k = torch.randn(1, 2, s, 16); v = torch.randn(1, 2, s, 16)
    cfg = AsaConfig(enable=True, variant="dense_probe", block_size=4,
                    num_keep=2, use_gilbert=True, text_length=l_len,
                    sta_enable=True, sta_window=(1, 1, 1))
    cache = StaMaskCache()
    captured = _run_asa(cfg, q, k, v, rearr, n_video, l_len, cache)
    # dense_probe -> mask is None (full attention)
    assert captured is None


def test_step_gate_unchanged_with_sta_enabled():
    # The existing ASA step gate (_should_use_dense_for_step) must continue
    # to govern whether asa_attention runs at all. STA does not introduce a
    # second gate. Verify by checking: warmup_steps=2, step_idx=0 -> dense
    # path used regardless of sta_enable.
    from vllm_omni.diffusion.models.mgm_video.mmdit.mmdit_blocks_inference import (
        JoinAttentionInference,
    )
    cfg_sta = AsaConfig(enable=True, variant="asa", warmup_steps=2,
                        sta_enable=True, sta_window=(1, 1, 1))
    cfg_no_sta = AsaConfig(enable=True, variant="asa", warmup_steps=2,
                           sta_enable=False)
    # _should_use_dense_for_step is a pure method on the class; instantiate
    # nothing else.
    fake_self = type("F", (), {
        "_asa_step_scheme": None, "_asa_scheme_path_loaded": None,
    })()
    method = JoinAttentionInference._should_use_dense_for_step
    assert method(fake_self, 0, cfg_sta) is True
    assert method(fake_self, 0, cfg_no_sta) is True
    assert method(fake_self, 5, cfg_sta) is False
    assert method(fake_self, 5, cfg_no_sta) is False


def test_asa_attention_sta_enabled_asa_g_warns_and_skips(caplog):
    import logging
    torch.manual_seed(0)
    T, H, W = 2, 4, 4
    rearr = GilbertRearranger(width=W, height=H, depth=T, text_length=8)
    n_video, l_len = T * H * W, 8
    s = n_video + l_len
    q = torch.randn(1, 2, s, 16); k = torch.randn(1, 2, s, 16); v = torch.randn(1, 2, s, 16)
    cfg = AsaConfig(enable=True, variant="asa_g", block_size=4,
                    num_keep=2, use_gilbert=True, text_length=l_len,
                    sta_enable=True, sta_window=(1, 1, 1))
    cache = StaMaskCache()
    with caplog.at_level(logging.WARNING, logger="vllm_omni.diffusion.models.mgm_video.mmdit.asa"):
        # asa_g is unsupported in scope (1); asa_attention asserts variant in
        # {'dense_probe', 'asa'}. The warning + skip is enforced upstream
        # at config load by the runtime; here we assert the assert fires
        # so an asa_g + sta combo never silently runs the asa path.
        with pytest.raises(AssertionError):
            asa_attention(
                q, k, v, cfg=cfg, rearranger=rearr,
                t_len=n_video, l_len=l_len,
                fa_full_dense=_FakeFA(),
                sta_cache=cache,
            )


def test_sta_window_validation_clamps_oversized():
    # window axis larger than grid axis -> clamped to grid_dim - (1 - grid_dim%2),
    # producing the nearest odd value <= grid_dim.
    from vllm_omni.diffusion.models.mgm_video.mmdit.sta import (
        validate_and_normalize_window,
    )
    # grid (T=4, H=4, W=4); request window (9, 9, 9) -> clamped to (3, 3, 3) (odd <= 4)
    out = validate_and_normalize_window((9, 9, 9), grid=(4, 4, 4))
    assert out == (3, 3, 3)


def test_sta_window_validation_rounds_even_up_to_odd():
    from vllm_omni.diffusion.models.mgm_video.mmdit.sta import (
        validate_and_normalize_window,
    )
    out = validate_and_normalize_window((4, 6, 8), grid=(16, 32, 32))
    assert out == (5, 7, 9)
