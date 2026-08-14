import copy
import io

import pytest
import torch
import torch.nn as nn

from memory_module.causal_memory_adapter import (
    CausalMemoryAdapter,
    CausalMemoryAdapterConfig,
    WanCompatibleRMSNorm,
    WanCausalMemoryAdapterHooks,
    fixed_separable_position_encoding,
    expected_trainable_inventory,
    tensor_module_fingerprint,
)


class FakeSelfAttention(nn.Module):
    def __init__(self, dim=24, heads=3):
        super().__init__()
        self.dim = dim
        self.num_heads = heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanCompatibleRMSNorm(dim)
        self.norm_k = WanCompatibleRMSNorm(dim)


@pytest.fixture
def fixture():
    torch.manual_seed(7)
    cfg = CausalMemoryAdapterConfig(
        latent_channels=4,
        memory_frames=3,
        latent_height=6,
        latent_width=8,
        encoder_channels=8,
        pool_height=2,
        pool_width=2,
        hidden_dim=24,
        num_heads=3,
    )
    base = FakeSelfAttention()
    adapter = CausalMemoryAdapter.from_wan_self_attention(base, cfg)
    return cfg, base, adapter


def test_physical_bypass_returns_exact_object_without_encoder_call(fixture, monkeypatch):
    _, _, adapter = fixture
    h = torch.randn(1, 9, 24)

    def forbidden(*args, **kwargs):
        raise AssertionError("encoder was evaluated during physical bypass")

    monkeypatch.setattr(adapter.memory_encoder, "forward", forbidden)
    assert adapter(h, h, memory_latents=None, route_query_mask=None) is h
    assert adapter(h, h, memory_latents=torch.ones(1), route_query_mask=None, rejected=True) is h
    assert adapter(h, h, memory_latents=torch.ones(1), route_query_mask=None, adapter_enabled=False) is h


def test_enabled_path_only_changes_routed_tokens(fixture):
    cfg, _, adapter = fixture
    h_sa0 = torch.randn(1, 9, cfg.hidden_dim)
    h_base = torch.randn_like(h_sa0)
    memory = torch.randn(1, cfg.latent_channels, cfg.memory_frames, cfg.latent_height, cfg.latent_width)
    route = torch.tensor([[False, True, True, False, False, True, False, False, False]])
    fused, diag = adapter(
        h_sa0,
        h_base,
        memory_latents=memory,
        route_query_mask=route,
        return_diagnostics=True,
    )
    assert fused.shape == h_base.shape
    assert torch.equal(fused[~route], h_base[~route])
    assert torch.count_nonzero(diag["scattered_memory_delta"][~route]) == 0
    assert torch.count_nonzero(diag["fused_delta"][~route]) == 0
    assert diag["memory_tokens"].shape == (1, cfg.memory_tokens, cfg.hidden_dim)
    assert diag["attention_weights"].shape == (1, cfg.num_heads, 3, cfg.memory_tokens)
    assert torch.isfinite(fused).all()


def test_non_support_stays_exact_zero_after_bridge_biases_move(fixture):
    cfg, _, adapter = fixture
    nn.init.constant_(adapter.bridge.w1.bias, 0.25)
    nn.init.constant_(adapter.bridge.w2.bias, 0.5)
    h = torch.randn(1, 6, cfg.hidden_dim)
    memory = torch.randn(1, cfg.latent_channels, cfg.memory_frames, cfg.latent_height, cfg.latent_width)
    route = torch.tensor([[False, True, False, True, False, False]])
    fused = adapter(h, h, memory_latents=memory, route_query_mask=route)
    assert torch.equal(fused[~route], h[~route])


def test_kvo_are_exact_clones_and_base_q_is_not_adapter_state(fixture):
    _, base, adapter = fixture
    assert torch.equal(adapter.wk_mem.weight, base.k.weight)
    assert torch.equal(adapter.wv_mem.weight, base.v.weight)
    assert torch.equal(adapter.wo_mem.weight, base.o.weight)
    assert torch.equal(adapter.norm_k_mem.weight, base.norm_k.weight)
    keys = set(adapter.state_dict())
    assert not any("base_q" in key or "base_norm_q" in key for key in keys)
    assert all(parameter.requires_grad for parameter in base.q.parameters())


def test_state_dict_round_trip_and_reload_parity(fixture):
    cfg, base, adapter = fixture
    h_sa0 = torch.randn(1, 7, cfg.hidden_dim)
    h_base = torch.randn_like(h_sa0)
    memory = torch.randn(1, cfg.latent_channels, cfg.memory_frames, cfg.latent_height, cfg.latent_width)
    route = torch.tensor([[True, False, True, False, True, False, False]])
    before = adapter(h_sa0, h_base, memory_latents=memory, route_query_mask=route)
    buffer = io.BytesIO()
    torch.save({
        "config": cfg.canonical_dict(),
        "config_fingerprint": cfg.fingerprint(),
        "trainable_inventory": adapter.trainable_inventory(),
        "state_dict": adapter.adapter_state_dict(),
    }, buffer)
    buffer.seek(0)
    payload = torch.load(buffer, weights_only=True)
    restored_cfg = CausalMemoryAdapterConfig(**payload["config"])
    assert restored_cfg.fingerprint() == payload["config_fingerprint"]
    restored = CausalMemoryAdapter.from_wan_self_attention(base, restored_cfg)
    restored.load_adapter_state_dict(payload["state_dict"])
    assert restored.trainable_inventory() == payload["trainable_inventory"]
    after = restored(h_sa0, h_base, memory_latents=memory, route_query_mask=route)
    assert torch.equal(before, after)
    assert tensor_module_fingerprint(adapter) == tensor_module_fingerprint(restored)


def test_base_state_is_unchanged_by_enabled_and_bypass_forward(fixture):
    cfg, base, adapter = fixture
    original = copy.deepcopy(base.state_dict())
    h = torch.randn(1, 5, cfg.hidden_dim)
    memory = torch.randn(1, cfg.latent_channels, cfg.memory_frames, cfg.latent_height, cfg.latent_width)
    route = torch.tensor([[False, True, True, False, False]])
    adapter(h, h, memory_latents=None, route_query_mask=None)
    adapter(h, h, memory_latents=memory, route_query_mask=route)
    assert all(torch.equal(original[name], value) for name, value in base.state_dict().items())


def test_position_encoding_frozen_contract_partitions_and_order():
    cfg = CausalMemoryAdapterConfig(hidden_dim=5120)
    pe = fixed_separable_position_encoding(cfg, torch.device("cpu"))
    assert pe.shape == (96, 5120)
    # Same frame/row and adjacent column: only frozen column partition differs.
    assert torch.equal(pe[0, :3414], pe[1, :3414])
    assert not torch.equal(pe[0, 3414:], pe[1, 3414:])
    assert not pe.requires_grad


def test_invalid_route_and_memory_contract_fail_closed(fixture):
    cfg, _, adapter = fixture
    h = torch.randn(1, 5, cfg.hidden_dim)
    memory = torch.randn(1, cfg.latent_channels, cfg.memory_frames, cfg.latent_height, cfg.latent_width)
    with pytest.raises(TypeError, match="boolean"):
        adapter(h, h, memory_latents=memory, route_query_mask=torch.ones(1, 5))
    with pytest.raises(ValueError, match="positive"):
        adapter(h, h, memory_latents=memory, route_query_mask=torch.zeros(1, 5, dtype=torch.bool))
    bad = memory[:, :, :, :-1]
    with pytest.raises(ValueError, match="ordered"):
        adapter(h, h, memory_latents=bad, route_query_mask=torch.ones(1, 5, dtype=torch.bool))


def test_adapter_construction_preserves_all_rng_states():
    cfg = CausalMemoryAdapterConfig(
        latent_channels=4, memory_frames=3, latent_height=6, latent_width=8,
        encoder_channels=8, pool_height=2, pool_width=2, hidden_dim=24, num_heads=3,
    )
    torch.manual_seed(1234)
    base = FakeSelfAttention()
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    CausalMemoryAdapter.from_wan_self_attention(base, cfg)
    assert torch.equal(cpu_before, torch.random.get_rng_state())
    if torch.cuda.is_available():
        assert all(
            torch.equal(before, after)
            for before, after in zip(cuda_before, torch.cuda.get_rng_state_all(), strict=True)
        )


def test_trainable_inventory_exactly_matches_frozen_contract(fixture):
    cfg, _, adapter = fixture
    expected = expected_trainable_inventory(cfg, dtype="torch.float32")
    assert adapter.trainable_inventory() == expected
    assert len(expected) == 17
    assert sum(item["numel"] for item in expected) == sum(
        parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad
    )
    assert not adapter.needs_integration_hooks(
        adapter_enabled=False, memory_latents=None, rejected=False
    )
    assert not adapter.needs_integration_hooks(
        adapter_enabled=True, memory_latents=None, rejected=False
    )
    assert not adapter.needs_integration_hooks(
        adapter_enabled=True, memory_latents=torch.ones(1), rejected=True
    )
    assert adapter.needs_integration_hooks(
        adapter_enabled=True, memory_latents=torch.ones(1), rejected=False
    )


def test_fully_frozen_base_produces_exact_trainable_adapter_inventory_and_gradients():
    torch.manual_seed(19)
    cfg = CausalMemoryAdapterConfig(
        latent_channels=4, memory_frames=3, latent_height=6, latent_width=8,
        encoder_channels=8, pool_height=2, pool_width=2, hidden_dim=24, num_heads=3,
    )
    base = FakeSelfAttention().requires_grad_(False)
    base_state = copy.deepcopy(base.state_dict())
    adapter = CausalMemoryAdapter.from_wan_self_attention(base, cfg)

    assert adapter.trainable_inventory() == expected_trainable_inventory(
        cfg, dtype="torch.float32"
    )
    assert len(adapter.trainable_inventory()) == 17
    assert all(not parameter.requires_grad for parameter in base.parameters())
    assert all(
        torch.equal(base_state[name], value)
        for name, value in base.state_dict().items()
    )

    h = torch.randn(1, 5, cfg.hidden_dim)
    memory = torch.randn(
        1, cfg.latent_channels, cfg.memory_frames, cfg.latent_height, cfg.latent_width
    )
    route = torch.tensor([[False, True, True, False, False]])
    adapter(h, h, memory_latents=memory, route_query_mask=route).float().sum().backward()

    intended = dict(adapter.named_parameters())
    assert set(intended) == {
        item["name"]
        for item in expected_trainable_inventory(cfg, dtype="torch.float32")
    }
    assert all(parameter.requires_grad for parameter in intended.values())
    assert all(parameter.grad is not None for parameter in intended.values())
    assert all(parameter.grad is None for parameter in base.parameters())


def test_ragged_batch_route_masks_are_supported():
    cfg = CausalMemoryAdapterConfig(
        latent_channels=4, memory_frames=3, latent_height=6, latent_width=8,
        encoder_channels=8, pool_height=2, pool_width=2, hidden_dim=24, num_heads=3,
    )
    adapter = CausalMemoryAdapter.from_wan_self_attention(FakeSelfAttention(), cfg)
    h = torch.randn(2, 7, cfg.hidden_dim)
    memory = torch.randn(2, cfg.latent_channels, cfg.memory_frames, cfg.latent_height, cfg.latent_width)
    route = torch.tensor([
        [True, False, True, False, False, False, False],
        [False, True, True, False, True, False, True],
    ])
    fused, diag = adapter(h, h, memory_latents=memory, route_query_mask=route, return_diagnostics=True)
    assert torch.equal(fused[~route], h[~route])
    assert diag["attention_weights"].shape == (2, cfg.num_heads, 4, cfg.memory_tokens)
    assert diag["attention_query_validity"].tolist() == [[True, True, False, False], [True] * 4]


class FakeWanBlock(nn.Module):
    def __init__(self, dim=24):
        super().__init__()
        self.modulation = nn.Parameter(torch.zeros(1, 6, dim))
        self.norm1 = nn.Identity()

    def forward(self, x, e=None):
        return x + 1


class FakeWan(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([FakeWanBlock()])
        self.head = nn.Identity()

    def forward(self, x, e):
        return self.head(self.blocks[0](x, e=e))


class FakeWanFloat32Output(FakeWan):
    def __init__(self):
        super().__init__()
        self.blocks[0].forward = lambda x, e=None: x.float() + 1


def test_real_hook_interface_bypass_and_enabled_route(fixture):
    cfg, _, adapter = fixture
    model = FakeWan()
    x = torch.randn(1, 6, cfg.hidden_dim)
    e = torch.zeros(1, 6, 6, cfg.hidden_dim)
    baseline = model(x, e)
    with WanCausalMemoryAdapterHooks(
        model, adapter, memory_latents=None, route_query_mask=None,
        adapter_enabled=False,
    ) as hooks:
        bypass = model(x, e)
    assert torch.equal(baseline, bypass)
    assert hooks.block0_input is x
    assert hooks.block0_output is not None
    assert torch.equal(hooks.block0_output, x + 1)
    assert hooks.h_sa0 is not None and hooks.pre_head_input is not None
    assert hooks.pre_head_fused is hooks.pre_head_input
    memory = torch.randn(1, cfg.latent_channels, cfg.memory_frames, cfg.latent_height, cfg.latent_width)
    route = torch.tensor([[False, True, True, False, False, False]])
    with WanCausalMemoryAdapterHooks(
        model, adapter, memory_latents=memory, route_query_mask=route,
    ) as hooks:
        enabled = model(x, e)
    assert torch.equal(enabled[~route], baseline[~route])
    assert torch.isfinite(enabled).all()
    assert hooks.block0_input is x
    assert torch.equal(hooks.block0_output, x + 1)
    assert hooks.adapter_diagnostics["physical_bypass"] is False
    assert hooks.pre_head_fused is not hooks.pre_head_input


def test_wan_hook_distinguishes_block_input_output_and_query_dtypes(fixture):
    _, _, adapter = fixture
    model = FakeWanFloat32Output()
    x = torch.randn(1, 6, 24, dtype=torch.bfloat16)
    e = torch.zeros(1, 6, 6, 24, dtype=torch.float32)
    with WanCausalMemoryAdapterHooks(
        model, adapter, memory_latents=None, route_query_mask=None,
        adapter_enabled=False,
    ) as hooks:
        result = model(x, e)
    assert hooks.block0_input is x
    assert hooks.block0_input.dtype == torch.bfloat16
    assert hooks.block0_output.dtype == torch.float32
    assert hooks.h_sa0.dtype == torch.float32
    assert hooks.pre_head_input.dtype == torch.float32
    assert result.dtype == torch.float32


def test_wan_hook_rejects_positional_or_missing_e_and_unloads(fixture):
    _, _, adapter = fixture
    model = FakeWan()
    x = torch.randn(1, 6, 24)
    e = torch.zeros(1, 6, 6, 24, dtype=torch.float32)
    hooks = WanCausalMemoryAdapterHooks(
        model, adapter, memory_latents=None, route_query_mask=None,
        adapter_enabled=False,
    )
    with pytest.raises(RuntimeError, match="exactly one positional"):
        with hooks:
            model.blocks[0](x, e)
    assert hooks.handles == []
    # Hook removal is real: the formerly invalid positional call now reaches
    # the fake block normally.
    assert torch.equal(model.blocks[0](x, e), x + 1)

    with pytest.raises(RuntimeError, match="missing required keyword e"):
        with WanCausalMemoryAdapterHooks(
            model, adapter, memory_latents=None, route_query_mask=None,
            adapter_enabled=False,
        ):
            model.blocks[0](x)


def test_wan_hook_validates_keyword_e_shape_and_dtype(fixture):
    _, _, adapter = fixture
    model = FakeWan()
    x = torch.randn(1, 6, 24)
    for e, message in (
        (torch.zeros(1, 1, 6, 24), r"must be \[B,L,6,D\]"),
        (torch.zeros(1, 6, 6, 24, dtype=torch.bfloat16), "must be float32"),
    ):
        with pytest.raises(RuntimeError, match=message):
            with WanCausalMemoryAdapterHooks(
                model, adapter, memory_latents=None, route_query_mask=None,
                adapter_enabled=False,
            ):
                model.blocks[0](x, e=e)
