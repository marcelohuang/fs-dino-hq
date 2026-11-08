"""Output semantics of FSDinoHQModel: canonical base, HQ residual, corrected output."""
import pytest
import torch

from stubs import StubDino, StubSam

from fs_dino_hq.modeling.backbones import build_sam_prompt_decoder
from fs_dino_hq.modeling.correlation import Correlation4DBranchV2, detect_corr_arch
from fs_dino_hq.modeling.distill_adapter import BottleneckAdapter as DistillationAdapter
from fs_dino_hq.modeling.hq_decoder import HQMaskDecoder
from fs_dino_hq.modeling.kshot import forward_kshot
from fs_dino_hq.modeling.model import Correlation4DBranchV1, FSDinoHQModel
from fs_dino_hq.modeling.stage2 import Stage2Model

S = 518
CPU = torch.device("cpu")


def _episode(K=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    m = torch.zeros(1, K, 1, S, S)
    for k in range(K):
        m[0, k, 0, 50 + 20 * k:250 + 20 * k, 100:300] = 1
    qm = torch.zeros(1, 1, S, S)
    qm[0, 0, 80:200, 60:400] = 1
    return {"support_dino": torch.randn(1, K, 3, S, S, generator=g), "support_mask": m,
            "query_dino": torch.randn(1, 3, S, S, generator=g), "query_mask": qm}


def _build(seed=0, dino=None):
    torch.manual_seed(seed)
    model = FSDinoHQModel(dino if dino is not None else StubDino(seed=0), build_sam_prompt_decoder(),
                          adapter_norm_first=True, adapter_dropout=0.0)
    model.early_adapter = DistillationAdapter(in_channels=768, out_channels=256)
    return model.eval()


@pytest.fixture(scope="module")
def model():
    return _build()


@torch.no_grad()
def test_canonical_base_independent_of_hq_parameters(model):
    batch = _episode()
    out1 = forward_kshot(model, batch, CPU)
    saved = {k: v.clone() for k, v in model.state_dict().items()}
    try:
        g = torch.Generator().manual_seed(5)
        n_hq = 0
        for _, p in model.hq_mask_decoder.hq_named_parameters():
            p.add_(0.5 * torch.randn(p.shape, generator=g))
            n_hq += 1
        assert n_hq > 0
        for p in model.early_adapter.parameters():          # frozen HQ-branch input
            p.add_(0.1 * torch.randn(p.shape, generator=g))
        out2 = forward_kshot(model, batch, CPU)
    finally:
        model.load_state_dict(saved)
    assert torch.equal(out1["canonical_base_logits"], out2["canonical_base_logits"])
    assert torch.equal(out1["iou_pred"], out2["iou_pred"])
    assert not torch.allclose(out1["hq_logits"], out2["hq_logits"])
    # the shared-pass SAM output does depend on the HQ token (diagnostic only)
    assert not torch.equal(out1["shared_pass_sam_logits"], out2["shared_pass_sam_logits"])


@torch.no_grad()
def test_corrected_is_base_plus_hq(model):
    out = forward_kshot(model, _episode(K=3, seed=1), CPU)
    assert model.hq_mask_decoder.residual_scale == 1.0
    for k in ("canonical_base_logits", "hq_logits", "corrected_logits", "final_logits",
              "shared_pass_sam_logits"):
        assert out[k].shape == (1, 1, 256, 256), k
    assert torch.equal(out["corrected_logits"], out["canonical_base_logits"] + out["hq_logits"])
    assert out["final_logits"] is out["corrected_logits"]
    assert out["prior_logits"].shape == (1, 1, 37, 37)
    assert torch.equal(model(_episode(K=3, seed=1))["corrected_logits"], out["corrected_logits"])


def test_corrected_logits_carry_gradient_only_through_hq(model):
    batch = _episode(seed=2)
    hq_params = [p for _, p in model.hq_mask_decoder.hq_named_parameters()]
    for p in hq_params:
        p.requires_grad_(True)
    try:
        out = forward_kshot(model, batch, CPU)
        assert not out["canonical_base_logits"].requires_grad
        out["corrected_logits"].mean().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in hq_params)
    finally:
        for p in hq_params:
            p.grad = None
            p.requires_grad_(False)


def test_hq_keys():
    assert HQMaskDecoder.HQ_MODULE_NAMES == {"hf_token", "hf_mlp", "dino_l3_proj", "adapter_proj",
                                             "early_proj", "decoder_mask_proj"}
    assert HQMaskDecoder.is_hq_key("hf_token.weight")
    assert HQMaskDecoder.is_hq_key("adapter_proj.0.weight")
    assert not HQMaskDecoder.is_hq_key("transformer.layers.0.self_attn.q_proj.weight")
    assert not HQMaskDecoder.is_hq_key("mask_tokens.weight")
    assert not HQMaskDecoder.is_hq_key("hf_token_extra.weight")


def test_hq_state_dict_excludes_stage2_decoder(model):
    hq = model.hq_mask_decoder.hq_state_dict()
    assert hq and all(HQMaskDecoder.is_hq_key(k) for k in hq)
    base_keys = set(build_sam_prompt_decoder().mask_decoder.state_dict())
    assert not base_keys & set(hq)
    assert base_keys | set(hq) == set(model.hq_mask_decoder.state_dict())


@torch.no_grad()
def test_canonical_base_equals_stage2_output():
    """With the Stage 2 weights mapped onto FSDinoHQModel, the canonical base is
    the Stage 2 decoder output (single-mask slot 0)."""
    torch.manual_seed(3)
    dino = StubDino(seed=0)
    st2 = Stage2Model(dino, StubSam(seed=3)).eval()
    hq = _build(seed=4, dino=dino)      # same frozen backbone (StubDino's bias follows the global seed)
    sd = {}
    for k, v in st2.stage2_state_dict().items():
        k = (k.replace("sam_full.mask_decoder.", "hq_mask_decoder.")
              .replace("sam_full.prompt_encoder.", "prompt_encoder."))
        sd[k] = v
    missing, unexpected = hq.load_state_dict(sd, strict=False)
    assert not unexpected
    assert all(k.startswith(("last_extractor.", "layer3_extractor.", "early_adapter."))
               or HQMaskDecoder.is_hq_key(k.split("hq_mask_decoder.", 1)[-1]) for k in missing)
    batch = _episode(K=2, seed=5)
    out_st2 = st2(batch)
    out_hq = forward_kshot(hq, batch, CPU)
    assert torch.allclose(out_hq["canonical_base_logits"], out_st2["final_logits"],
                          rtol=1e-5, atol=1e-5)
    assert torch.allclose(out_hq["prior_logits"], out_st2["prior_logits"], rtol=1e-5, atol=1e-6)


# ── correlation architecture ────────────────────────────────────────────────

def test_detect_corr_arch_v2(model):
    assert detect_corr_arch(model.state_dict()) == ("v2_reduced", 37)
    torch.manual_seed(0)
    st2 = Stage2Model(StubDino(), StubSam())
    assert detect_corr_arch(st2.stage2_state_dict()) == ("v2_reduced", 37)
    small = {f"corr_branch.{k}": v for k, v in Correlation4DBranchV2(feat_hw=5).state_dict().items()}
    assert detect_corr_arch(small) == ("v2_reduced", 5)
    bare = Correlation4DBranchV2(feat_hw=7).state_dict()
    assert detect_corr_arch(bare, prefix="") == ("v2_reduced", 7)


def test_detect_corr_arch_other_layouts():
    v1 = {f"corr_branch.{k}": v for k, v in Correlation4DBranchV1(37, 18).state_dict().items()}
    assert detect_corr_arch(v1) == ("v1_downsampled", 18)
    v3 = {"corr_branch.reduce_support.0.weight": torch.zeros(1),
          "corr_branch.prior_head.0.weight": torch.zeros(1)}
    assert detect_corr_arch(v3)[0] == "v3_geometric"
    with pytest.raises(KeyError):
        detect_corr_arch({"adapter.proj_out.weight": torch.zeros(1)})


@pytest.mark.parametrize("arch", ["v1_downsampled", "v3_geometric", "something"])
def test_models_refuse_other_corr_archs(arch):
    with pytest.raises(ValueError):
        FSDinoHQModel(StubDino(), build_sam_prompt_decoder(), corr_arch=arch)
    with pytest.raises(ValueError):
        Stage2Model(StubDino(), StubSam(), corr_arch=arch)


def test_v1_branch_is_replaced(model):
    assert isinstance(model.corr_branch, Correlation4DBranchV2)
    assert model.corr_arch == "v2_reduced"
    assert not any(k.startswith("corr_branch.") and "reduce_conv1" not in k and "reduce_conv2" not in k
                   and k.split(".")[1] not in ("depthwise", "pointwise", "prior_head", "dense_prompt_head")
                   for k in model.state_dict())
