"""K-shot support aggregation: prompt averaging shared by Stage 2, HQ training and inference."""
import pytest
import torch
import torch.nn.functional as F

from stubs import StubDino, StubSam

from fs_dino_hq.modeling.backbones import build_sam_prompt_decoder
from fs_dino_hq.modeling.distill_adapter import BottleneckAdapter as DistillationAdapter
from fs_dino_hq.modeling.kshot import aggregate_prompts, compute_support_prompts, forward_kshot
from fs_dino_hq.modeling.model import FSDinoHQModel
from fs_dino_hq.modeling.stage2 import Stage2Model

S = 518


def _masks(n, seed):
    """[n,1,518,518] binary masks with one random rectangle each."""
    g = torch.Generator().manual_seed(seed)
    m = torch.zeros(n, 1, S, S)
    for i in range(n):
        y0, x0 = torch.randint(0, 300, (2,), generator=g).tolist()
        h, w = torch.randint(40, 200, (2,), generator=g).tolist()
        m[i, 0, y0:y0 + h, x0:x0 + w] = 1
    return m


def _images(n, seed):
    return torch.randn(n, 3, S, S, generator=torch.Generator().manual_seed(seed))


@pytest.fixture(scope="module")
def hq_model():
    torch.manual_seed(0)
    model = FSDinoHQModel(StubDino(seed=0), build_sam_prompt_decoder(),
                          adapter_norm_first=True, adapter_dropout=0.0)
    model.early_adapter = DistillationAdapter(in_channels=768, out_channels=256)
    return model.eval()


@pytest.fixture(scope="module")
def stage2_model():
    torch.manual_seed(1)
    return Stage2Model(StubDino(seed=0), StubSam(seed=1)).eval()


@pytest.fixture(scope="module")
def episode():
    K = 3
    return {"support_dino": _images(K, 10)[None], "support_mask": _masks(K, 11)[None],
            "query_dino": _images(1, 12), "query_mask": _masks(1, 13)}


def test_aggregate_prompts_is_mean_over_supports():
    g = torch.Generator().manual_seed(0)
    K = 4
    p = {"semantic_k": torch.randn(K, 1, 256, generator=g),
         "dense_k": torch.randn(K, 256, 64, 64, generator=g),
         "prior_logits_k": torch.randn(K, 1, 37, 37, generator=g),
         "prior_mask_k": torch.rand(K, 1, 37, 37, generator=g),
         "S_pcs_k": torch.randn(K, 1, 37, 37, generator=g)}
    a = aggregate_prompts(p)
    assert set(a) == {"semantic_prompt", "dense_prompt", "prior_logits", "prior_mask"}
    for out_key, in_key in (("semantic_prompt", "semantic_k"), ("dense_prompt", "dense_k"),
                            ("prior_logits", "prior_logits_k"), ("prior_mask", "prior_mask_k")):
        assert a[out_key].shape == (1, *p[in_key].shape[1:])
        assert torch.equal(a[out_key], p[in_key].mean(0, keepdim=True))
        assert torch.allclose(a[out_key], p[in_key].sum(0, keepdim=True) / K, atol=1e-6)
    # the semantic prompt is averaged without renormalisation
    unit = F.normalize(p["semantic_k"], dim=-1)
    a2 = aggregate_prompts({**p, "semantic_k": unit})
    assert a2["semantic_prompt"].norm() < 1.0


@torch.no_grad()
def test_k1_equals_single_support_path(hq_model, episode):
    m = hq_model
    sd, sm = episode["support_dino"][0, :1], episode["support_mask"][0, :1]
    Fq = m.last_extractor(episode["query_dino"])
    agg = aggregate_prompts(compute_support_prompts(m, sd, sm, Fq))
    Fs = m.last_extractor(sd)
    mask = F.interpolate(sm, size=Fs.shape[-2:], mode="nearest")
    sem = m.semantic_branch(Fs, mask)
    _, s_pcs = m.pcs_branch(Fs, mask, Fq)
    _, _, prior_logits, prior_mask, dense = m.corr_branch(Fs, Fq, s_pcs)
    assert torch.equal(agg["semantic_prompt"], sem)
    assert torch.equal(agg["dense_prompt"], dense)
    assert torch.equal(agg["prior_logits"], prior_logits)
    assert torch.equal(agg["prior_mask"], prior_mask)


@torch.no_grad()
def test_permuting_supports_changes_outputs_only_at_float_level(hq_model, episode):
    out = forward_kshot(hq_model, episode, torch.device("cpu"))
    perm = [2, 0, 1]
    permuted = {**episode, "support_dino": episode["support_dino"][:, perm],
                "support_mask": episode["support_mask"][:, perm]}
    out_p = forward_kshot(hq_model, permuted, torch.device("cpu"))
    for k in ("canonical_base_logits", "hq_logits", "corrected_logits", "prior_logits"):
        assert torch.allclose(out[k], out_p[k], rtol=0, atol=1e-4), k
    # sanity: a different support set is not a float-level change
    other = {**episode, "support_mask": torch.flip(episode["support_mask"], dims=[-1])}
    out_o = forward_kshot(hq_model, other, torch.device("cpu"))
    assert (out["corrected_logits"] - out_o["corrected_logits"]).abs().max() > 1e-3


@torch.no_grad()
def test_forward_kshot_uses_aggregated_prompts(hq_model, episode):
    m = hq_model
    out = forward_kshot(m, episode, torch.device("cpu"))
    Fq = m.last_extractor(episode["query_dino"])
    Fq_l3 = m.layer3_extractor(episode["query_dino"])
    p = aggregate_prompts(compute_support_prompts(
        m, episode["support_dino"][0], episode["support_mask"][0], Fq))
    a64 = F.interpolate(m.adapter(Fq_l3), (64, 64), mode="bilinear", align_corners=False)
    base = m.hq_mask_decoder.predict_canonical_stage2_base(
        image_embeddings=a64, image_pe=m.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=p["semantic_prompt"], dense_prompt_embeddings=p["dense_prompt"])
    assert torch.equal(out["canonical_base_logits"], base["canonical_base_logits"])
    assert torch.equal(out["prior_logits"], p["prior_logits"])
    assert out["query_mask_final"].shape == (1, 1, 256, 256)
    assert out["query_mask_prior"].shape == (1, 1, 37, 37)


@torch.no_grad()
def test_stage2_kshot_prompts_equal_kshot_module(stage2_model):
    """Stage2Model.forward (B*K supports batched) aggregates exactly like kshot."""
    B, K = 2, 3
    batch = {"support_dino": _images(B * K, 20).view(B, K, 3, S, S),
             "support_mask": _masks(B * K, 21).view(B, K, 1, S, S),
             "query_dino": _images(B, 22), "query_mask": _masks(B, 23)}
    out = stage2_model(batch)
    assert out["n_shot"] == K
    Fq = stage2_model.last_extractor(batch["query_dino"])
    for b in range(B):
        p = aggregate_prompts(compute_support_prompts(
            stage2_model, batch["support_dino"][b], batch["support_mask"][b], Fq[b:b + 1]))
        for key in ("semantic_prompt", "dense_prompt", "prior_logits", "prior_mask"):
            assert torch.allclose(out[key][b:b + 1], p[key], rtol=1e-5, atol=1e-6), (b, key)


@torch.no_grad()
def test_stage2_k1_paths_agree(stage2_model):
    """4-D supports (one-shot layout) == 5-D with K=1 == kshot with K=1."""
    sd, sm, q = _images(1, 30), _masks(1, 31), _images(1, 32)
    out4 = stage2_model({"support_dino": sd, "support_mask": sm, "query_dino": q})
    out5 = stage2_model({"support_dino": sd[None], "support_mask": sm[None], "query_dino": q})
    Fq = stage2_model.last_extractor(q)
    p = aggregate_prompts(compute_support_prompts(stage2_model, sd, sm, Fq))
    for key in ("semantic_prompt", "dense_prompt", "prior_logits", "prior_mask", "final_logits"):
        assert torch.equal(out4[key], out5[key]), key
    for key in ("semantic_prompt", "dense_prompt", "prior_logits", "prior_mask"):
        assert torch.equal(out4[key], p[key]), key
