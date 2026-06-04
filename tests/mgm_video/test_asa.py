import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.models.mgm_video.mmdit.asa import (
    AsaConfig,
    GilbertRearranger,
    asa_attention,
    build_asa_block_mask,
    expand_block_to_token_mask,
    gilbert3d,
    pad_to_multiple,
    parse_asa_step_scheme,
    random_sample_tokens,
    sample_pool_attn,
)


def test_asa_config_defaults_disabled():
    cfg = AsaConfig()
    assert cfg.enable is False
    assert cfg.variant == "asa"
    assert cfg.max_retain_ratio == 0.20
    assert cfg.min_retain_ratio == 0.05
    assert cfg.energy_threshold == 0.95
    assert cfg.block_size == 128
    assert cfg.num_keep == 32
    assert cfg.sample_gap == 30
    assert cfg.use_gilbert is True
    assert cfg.text_length == 256
    assert cfg.video_shape is None
    assert cfg.warmup_steps == 0
    assert cfg.step_scheme_path is None


def test_asa_config_from_env_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_MGM_ASA_ENABLE", raising=False)
    cfg = AsaConfig.from_env()
    assert cfg.enable is False


def test_asa_config_from_env_reads_overrides(monkeypatch):
    monkeypatch.setenv("VLLM_MGM_ASA_ENABLE", "1")
    monkeypatch.setenv("VLLM_MGM_ASA_MAX_RETAIN", "0.15")
    monkeypatch.setenv("VLLM_MGM_ASA_VARIANT", "dense_probe")
    monkeypatch.setenv("VLLM_MGM_ASA_WARMUP_STEPS", "2")
    cfg = AsaConfig.from_env()
    assert cfg.enable is True
    assert cfg.max_retain_ratio == 0.15
    assert cfg.variant == "dense_probe"
    assert cfg.warmup_steps == 2


def test_asa_config_is_frozen():
    cfg = AsaConfig()
    with pytest.raises(Exception):
        cfg.enable = True  # type: ignore


def test_gilbert3d_visits_each_cell_once():
    cells = list(gilbert3d(4, 3, 2))
    assert len(cells) == 4 * 3 * 2
    assert len(set(cells)) == 4 * 3 * 2
    for x, y, z in cells:
        assert 0 <= x < 4 and 0 <= y < 3 and 0 <= z < 2


def test_gilbert_rearrange_inverse_is_identity_small():
    rearr = GilbertRearranger(width=4, height=3, depth=2, text_length=5)
    T = 4 * 3 * 2
    L = 5
    x = torch.randn(2, 3, T + L, 8)
    y = rearr.rearrange(x)
    z = rearr.reversed_rearrange(y)
    assert torch.equal(x, z)


def test_gilbert_text_segment_unchanged():
    rearr = GilbertRearranger(width=4, height=3, depth=2, text_length=5)
    T = 4 * 3 * 2
    L = 5
    x = torch.randn(1, 1, T + L, 4)
    y = rearr.rearrange(x)
    assert torch.equal(x[..., T:, :], y[..., T:, :])


def test_gilbert_rearrange_inverse_is_identity_mgm_shape():
    """Match mgm_video shape: W=80, H=45, T=16, L=256."""
    rearr = GilbertRearranger(width=80, height=45, depth=16, text_length=256)
    T = 80 * 45 * 16
    L = 256
    # Use small batch/head/dim to keep memory small
    x = torch.randn(1, 1, T + L, 4)
    y = rearr.rearrange(x)
    z = rearr.reversed_rearrange(y)
    assert torch.equal(x, z)


def test_pad_to_multiple_no_op_when_already_multiple():
    x = torch.randn(1, 1, 8, 4)
    y = pad_to_multiple(x, 4, dim=-2)
    assert torch.equal(x, y)


def test_pad_to_multiple_zero_pads_tail():
    x = torch.randn(1, 1, 5, 4)
    y = pad_to_multiple(x, 4, dim=-2)
    assert y.shape[-2] == 8
    assert torch.equal(x, y[..., :5, :])
    assert torch.all(y[..., 5:, :] == 0)


def test_random_sample_tokens_shape_and_membership():
    g = torch.Generator().manual_seed(42)
    x = torch.arange(2 * 3 * 16 * 4).reshape(2, 3, 16, 4).float()
    s = random_sample_tokens(x, block_size=4, num_keep=2, generator=g)
    assert s.shape == (2, 3, 4 * 2, 4)
    # every sampled row must equal some row from the original
    x_rows = set(map(tuple, x.reshape(-1, 4).tolist()))
    for row in s.reshape(-1, 4).tolist():
        assert tuple(row) in x_rows


def test_sample_pool_attn_output_shape_and_head_shared():
    g = torch.Generator().manual_seed(42)
    B, N, L, D = 1, 4, 256, 8
    q = torch.randn(B, N, L, D)
    k = torch.randn(B, N, L, D)
    P = sample_pool_attn(q, k, block_size=64, num_keep=8, generator=g)
    assert P.shape == (B, 1, 4, 4)  # nq = nk = 256/64 = 4, head-mean shared
    assert P.dtype == torch.float32
    assert (P >= 0).all()


def test_build_asa_block_mask_dense_probe_keeps_all():
    """max_retain=1.0 + threshold=0.95 with uniform P should keep every block."""
    nk = 16
    P = torch.full((1, 1, 4, nk), 1.0 / nk)
    mask = build_asa_block_mask(P, max_retain_ratio=1.0, min_retain_ratio=0.0, energy_threshold=0.95)
    # Uniform P + max_retain=1.0 clamps to nk per row; every position must be True.
    assert (mask.sum(-1) == nk).all()


def test_build_asa_block_mask_energy_threshold_prunes_tail():
    # row [0.5, 0.4, 0.05, 0.05]; cum=[0.5,0.9,0.95,1.0]; threshold=0.95 -> first 3
    P = torch.tensor([[[[0.5, 0.4, 0.05, 0.05]]]])
    mask = build_asa_block_mask(P, max_retain_ratio=1.0, min_retain_ratio=0.0, energy_threshold=0.95)
    assert mask[0, 0, 0].tolist() == [True, True, True, False]


def test_build_asa_block_mask_respects_min_retain():
    # row strongly peaked: [0.99, 0.01/3 each]; threshold=0.95 -> k=1
    # but min_retain=0.5 with nk=4 -> floor to 2 keeps
    P = torch.tensor([[[[0.99, 0.005, 0.0033, 0.0017]]]])
    mask = build_asa_block_mask(P, max_retain_ratio=1.0, min_retain_ratio=0.5, energy_threshold=0.95)
    assert mask.sum().item() == 2


def test_build_asa_block_mask_respects_max_retain():
    # row uniform: every block needed for 0.95; max_retain=0.25, nk=8 -> cap at 2
    P = torch.full((1, 1, 1, 8), 1.0 / 8)
    mask = build_asa_block_mask(P, max_retain_ratio=0.25, min_retain_ratio=0.0, energy_threshold=0.95)
    assert mask.sum().item() == 2


def test_build_asa_block_mask_bounds_random():
    torch.manual_seed(0)
    nq, nk = 32, 64
    P = torch.softmax(torch.randn(1, 1, nq, nk), dim=-1)
    mask = build_asa_block_mask(P, max_retain_ratio=0.20, min_retain_ratio=0.05, energy_threshold=0.95)
    row_sums = mask.sum(-1)
    assert (row_sums >= max(1, int(nk * 0.05))).all()
    assert (row_sums <= max(1, int(nk * 0.20))).all()


def test_expand_block_to_token_mask_layout_basic():
    """2x2 block mask with block_size=2, T=4, L=2 -> manually verifiable."""
    m_block = torch.tensor([[[[True, False], [False, True]]]])  # [1,1,2,2]
    m_tok = expand_block_to_token_mask(m_block, block_size=2, t_len=4, l_len=2)
    expected = torch.tensor([
        [1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ]).bool()
    assert torch.equal(m_tok[0, 0], expected)


def test_expand_block_to_token_mask_text_blocks_always_true():
    """For any random video block mask, the text x * and * x text rows/cols are all True."""
    torch.manual_seed(0)
    nq, nk = 4, 4
    m_block = torch.rand(1, 1, nq, nk) > 0.5
    t_len, l_len = 8, 3
    m_tok = expand_block_to_token_mask(m_block, block_size=2, t_len=t_len, l_len=l_len)
    assert torch.all(m_tok[..., t_len:, :])
    assert torch.all(m_tok[..., :, t_len:])


def test_expand_block_to_token_mask_truncates_padding():
    """T=5 with block_size=4 -> nq=2; expanded vv would be 8x8, truncate to 5x5."""
    m_block = torch.tensor([[[[True, False], [False, True]]]])  # [1,1,2,2]
    t_len, l_len = 5, 1
    m_tok = expand_block_to_token_mask(m_block, block_size=4, t_len=t_len, l_len=l_len)
    assert m_tok.shape == (1, 1, t_len + l_len, t_len + l_len)
    # rows 0..3 are vv block-row 0 -> [True]*4 then [False]*4 truncated to T=5
    assert m_tok[0, 0, 0, :t_len].tolist() == [True, True, True, True, False]



def _sdpa_fa(q, k, v, atten_mask):
    # F.scaled_dot_product_attention: attn_mask True = keep, False = mask out
    return F.scaled_dot_product_attention(q, k, v, attn_mask=atten_mask)


def test_asa_attention_dense_probe_matches_full_attention():
    """variant=dense_probe + use_gilbert=True must match full attention:
    rearrange + full_attn + inverse_rearrange == full_attn."""
    torch.manual_seed(0)
    W, H, Tdepth = 4, 3, 2
    T = W * H * Tdepth  # 24
    L = 5
    B, N, D = 1, 2, 8

    q = torch.randn(B, N, T + L, D)
    k = torch.randn(B, N, T + L, D)
    v = torch.randn(B, N, T + L, D)

    cfg = AsaConfig(enable=True, variant="dense_probe", use_gilbert=True, text_length=L)
    rearr = GilbertRearranger(W, H, Tdepth, text_length=L)

    out_asa = asa_attention(q, k, v, cfg, rearr, t_len=T, l_len=L, fa_full_dense=_sdpa_fa)
    out_ref = _sdpa_fa(q, k, v, atten_mask=None)
    torch.testing.assert_close(out_asa, out_ref, atol=1e-5, rtol=1e-5)


def test_asa_attention_asa_variant_text_segment_unchanged():
    """variant=asa: text rows always attend to all keys -> equal to full attention on text rows."""
    torch.manual_seed(0)
    W, H, Tdepth = 4, 3, 2
    T = W * H * Tdepth
    L = 5
    B, N, D = 1, 2, 8

    q = torch.randn(B, N, T + L, D)
    k = torch.randn(B, N, T + L, D)
    v = torch.randn(B, N, T + L, D)

    cfg = AsaConfig(
        enable=True, variant="asa", max_retain_ratio=0.5, min_retain_ratio=0.5,
        energy_threshold=0.95, block_size=8, num_keep=4, use_gilbert=False, text_length=L,
    )
    rearr = GilbertRearranger(W, H, Tdepth, text_length=L)
    g = torch.Generator().manual_seed(42)

    out = asa_attention(q, k, v, cfg, rearr, t_len=T, l_len=L, fa_full_dense=_sdpa_fa, generator=g)
    out_ref = _sdpa_fa(q, k, v, atten_mask=None)
    torch.testing.assert_close(out[..., T:, :], out_ref[..., T:, :], atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# P4.5 / P4.6: per-step gate tests for JoinAttentionInference
#
# We test the instance method `_should_use_dense_for_step` via a tiny fake
# that binds the unbound method without constructing a real
# JoinAttentionInference (whose __init__ pulls in heavy FA setup).
# ---------------------------------------------------------------------------

from vllm_omni.diffusion.models.mgm_video.mmdit.mmdit_blocks_inference import (
    JoinAttentionInference,
)


class _FakeAttn:
    """Minimal fake of JoinAttentionInference exposing only the gate
    method and the two cache attributes it touches."""
    _should_use_dense_for_step = JoinAttentionInference._should_use_dense_for_step

    def __init__(self):
        self._asa_step_scheme = None
        self._asa_scheme_path_loaded = None


# --- P4.5 warmup_steps gate (no scheme set) --------------------------------


def test_warmup_gate_step_below_threshold_returns_dense():
    cfg = AsaConfig(enable=True, warmup_steps=2)
    a = _FakeAttn()
    assert a._should_use_dense_for_step(0, cfg) is True
    assert a._should_use_dense_for_step(1, cfg) is True


def test_warmup_gate_step_at_or_above_threshold_returns_asa():
    cfg = AsaConfig(enable=True, warmup_steps=2)
    a = _FakeAttn()
    assert a._should_use_dense_for_step(2, cfg) is False
    assert a._should_use_dense_for_step(3, cfg) is False


def test_warmup_gate_warmup_zero_always_returns_asa():
    # warmup_steps=0 (default): no non-negative step_idx is < 0,
    # so we always fall through to ASA — bit-exact pre-P4.5 behavior.
    cfg = AsaConfig(enable=True, warmup_steps=0)
    a = _FakeAttn()
    assert a._should_use_dense_for_step(0, cfg) is False
    assert a._should_use_dense_for_step(5, cfg) is False


def test_warmup_gate_step_idx_none_returns_asa():
    # When pipeline didn't set _current_step_idx (e.g., direct unit-test
    # path), the gate degrades to ASA — never blocks the path.
    a = _FakeAttn()
    assert a._should_use_dense_for_step(None, AsaConfig(enable=True, warmup_steps=5)) is False
    assert a._should_use_dense_for_step(None, AsaConfig(enable=True, warmup_steps=0)) is False


# --- P4.6 step scheme parser ------------------------------------------------


def test_parse_asa_step_scheme_basic(tmp_path):
    p = tmp_path / "scheme.txt"
    p.write_text("1\n0\n1\n")
    assert parse_asa_step_scheme(str(p)) == (1, 0, 1)


def test_parse_asa_step_scheme_with_comments_and_blanks(tmp_path):
    p = tmp_path / "scheme.txt"
    p.write_text("# header\n\n1\n# mid\n0\n\n")
    assert parse_asa_step_scheme(str(p)) == (1, 0)


def test_parse_asa_step_scheme_rejects_malformed(tmp_path):
    p = tmp_path / "scheme.txt"
    p.write_text("1\nfoo\n")
    with pytest.raises(ValueError, match=r":2:"):
        parse_asa_step_scheme(str(p))


def test_parse_asa_step_scheme_strips_whitespace(tmp_path):
    p = tmp_path / "scheme.txt"
    p.write_text("  1  \n\t0\t\n")
    assert parse_asa_step_scheme(str(p)) == (1, 0)


def test_committed_example_scheme_parses():
    """Sanity that the committed sample scheme file parses to the expected tuple."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    sample = (
        repo_root
        / "vllm_omni"
        / "diffusion"
        / "models"
        / "mgm_video"
        / "asa_scheme"
        / "asa_scheme_8step_first_last_dense.txt"
    )
    assert sample.exists(), f"sample scheme file missing: {sample}"
    assert parse_asa_step_scheme(str(sample)) == (0, 1, 1, 1, 1, 1, 1, 0)


# --- P4.6 step scheme gate semantics ---------------------------------------


def _write_scheme(tmp_path, values, name="scheme.txt"):
    p = tmp_path / name
    p.write_text("\n".join(str(v) for v in values) + "\n")
    return str(p)


def test_step_scheme_gate_basic(tmp_path):
    scheme = [0, 1, 1, 1, 1, 1, 1, 0]
    cfg = AsaConfig(
        enable=True,
        warmup_steps=0,
        step_scheme_path=_write_scheme(tmp_path, scheme),
    )
    a = _FakeAttn()
    # step 0 -> dense
    assert a._should_use_dense_for_step(0, cfg) is True
    # step 1 -> ASA
    assert a._should_use_dense_for_step(1, cfg) is False
    # last step (7) -> dense
    assert a._should_use_dense_for_step(7, cfg) is True
    # out-of-range -> fall through to ASA
    assert a._should_use_dense_for_step(99, cfg) is False
    assert a._should_use_dense_for_step(-1, cfg) is False


def test_step_scheme_falls_back_to_asa_for_none_step(tmp_path):
    cfg = AsaConfig(
        enable=True,
        step_scheme_path=_write_scheme(tmp_path, [0, 1, 0]),
    )
    a = _FakeAttn()
    assert a._should_use_dense_for_step(None, cfg) is False


def test_step_scheme_overrides_warmup_steps(tmp_path):
    """When a scheme is set, warmup_steps is ignored (P4.6 over P4.5)."""
    scheme = [0, 1, 1, 1, 1, 1, 1, 0]
    cfg = AsaConfig(
        enable=True,
        warmup_steps=2,  # would say steps 0,1 dense; but scheme overrides
        step_scheme_path=_write_scheme(tmp_path, scheme),
    )
    a = _FakeAttn()
    # step 0 -> dense (scheme & warmup agree, but scheme is the source of truth)
    assert a._should_use_dense_for_step(0, cfg) is True
    # step 1 -> ASA per scheme, even though warmup_steps=2 would've said dense
    assert a._should_use_dense_for_step(1, cfg) is False
    # step 7 -> dense per scheme, warmup_steps=2 would've said ASA
    assert a._should_use_dense_for_step(7, cfg) is True


def test_step_scheme_caches_load(tmp_path):
    """Scheme is parsed once per attention instance per path."""
    path = _write_scheme(tmp_path, [1, 0, 1])
    cfg = AsaConfig(enable=True, step_scheme_path=path)
    a = _FakeAttn()
    a._should_use_dense_for_step(0, cfg)
    assert a._asa_step_scheme == (1, 0, 1)
    assert a._asa_scheme_path_loaded == path

    # Mutate the file on disk; cache should hold.
    with open(path, "w") as f:
        f.write("0\n0\n0\n")
    # step 0 was 1 (ASA) per cached scheme; verify cache was used.
    assert a._should_use_dense_for_step(0, cfg) is False
    assert a._asa_step_scheme == (1, 0, 1)


def test_step_scheme_reloads_on_path_change(tmp_path):
    """Switching to a different path triggers a reload."""
    pa = tmp_path / "a.txt"
    pa.write_text("1\n0\n1\n")
    pb = tmp_path / "b.txt"
    pb.write_text("0\n1\n0\n")

    cfg_a = AsaConfig(enable=True, step_scheme_path=str(pa))
    cfg_b = AsaConfig(enable=True, step_scheme_path=str(pb))
    a = _FakeAttn()
    a._should_use_dense_for_step(0, cfg_a)
    assert a._asa_step_scheme == (1, 0, 1)
    a._should_use_dense_for_step(0, cfg_b)
    assert a._asa_step_scheme == (0, 1, 0)


def test_asa_config_from_env_reads_step_scheme(monkeypatch, tmp_path):
    p = tmp_path / "scheme.txt"
    p.write_text("0\n1\n")
    monkeypatch.setenv("VLLM_MGM_ASA_STEP_SCHEME", str(p))
    cfg = AsaConfig.from_env()
    assert cfg.step_scheme_path == str(p)
