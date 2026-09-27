"""Quantization-aware training THROUGH the verified units.

This is the piece that was missing: a trainable layer whose FORWARD compute
actually runs on the verified GUDA logic --

    quantize -> NeuralMul (verified INT8 multiply) GEMM -> NeuralRequant16 ->
    NeuralReLU8 -> dequantize

-- while the BACKWARD uses a straight-through estimator (the integer path has no
gradient), so ordinary float weights still learn. With instrument.enable(), each
unit records how many times its neural forward ran, so a training run leaves
hard evidence (call counts) that it computed through the units, not around them.

Honest cost: every forward multiply is a neural forward pass -> this is SLOW
(functional, not fast). It is a correctness/《evidence》demo, not a speed path.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .backends import NeuralBackend


class _VerifiedQGEMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, mul, requant, relu_unit, use_relu, luts):
        ctx.save_for_backward(x, w)
        ctx.device = x.device                       # verified units run on CPU;
        xnp, wnp = x.detach().cpu().numpy(), w.detach().cpu().numpy()
        # A float -> int8 cast of NaN/Inf is undefined; numpy returns 0 with only a
        # RuntimeWarning. One non-finite value anywhere made the whole-tensor scale NaN/Inf
        # and silently zeroed EVERY quantized input of the layer (measured: [0.5, nan, 3.0]
        # -> [0, 0, 0]). Refuse at the source so a diverging run fails where it diverged.
        for nm, a in (("activations", xnp), ("weights", wnp)):
            if not np.isfinite(a).all():
                raise FloatingPointError(
                    "non-finite %s entering the verified GEMM (%d of %d values); the int8 "
                    "quantize would silently turn them all into 0"
                    % (nm, int((~np.isfinite(a)).sum()), a.size))
        sx = max(float(np.abs(xnp).max()) / 127.0, 1e-8)
        sw = max(float(np.abs(wnp).max()) / 127.0, 1e-8)
        xq = np.clip(np.round(xnp / sx), -128, 127).astype(np.int8)
        wq = np.clip(np.round(wnp / sw), -128, 127).astype(np.int8)

        if luts is not None:
            # FAST path: the verified units, materialized as lookup tables
            # (bit-identical to the neural forward, ~500x faster).
            from . import instrument
            acc = luts["backend"].gemm(xq, wq)         # counts LUT products
            acc16 = np.clip(acc, -32768, 32767).astype(np.int64)
            yq = luts["requant"][acc16 & 0xFFFF]
            instrument.bump("VerifiedRequant16(LUT).elements", acc16.size)
            if use_relu:
                yq = luts["relu"][yq & 0xFF]
                instrument.bump("VerifiedReLU8(LUT).elements", yq.size)
        else:
            acc = NeuralBackend(mul).gemm(xq, wq)      # verified multiply fires
            acc16 = np.clip(acc, -32768, 32767).astype(np.int64)
            yq = requant.requant_array(acc16)          # requant16 fires
            if use_relu:
                yq = relu_unit.relu_array(yq)          # relu8 fires
        dequant = sx * sw * 256.0                       # undo requant's >>8
        return torch.from_numpy(yq.astype(np.float32) * dequant).to(ctx.device)

    @staticmethod
    def backward(ctx, gy):
        # straight-through: treat the quantized path as y ≈ x @ w
        x, w = ctx.saved_tensors
        return gy @ w.t(), x.t() @ gy, None, None, None, None, None


def build_luts(mul, requant, relu_unit):
    """Materialize the verified units as lookup tables (one-time). The result is
    bit-identical to the neural forward but ~500x faster to run."""
    from .lut import LUTBackend, build_requant16_lut, build_relu8_lut
    return {"backend": LUTBackend(mul),
            "requant": build_requant16_lut(requant),
            "relu": build_relu8_lut(relu_unit)}


class VerifiedLinear(nn.Module):
    """Linear layer whose forward is computed by the verified units.

    fast=True materializes the units as LUTs (bit-identical, ~500x faster) so
    verified training is practical; fast=False runs the neural forward (proof).
    """

    def __init__(self, in_f, out_f, mul, requant, relu_unit, use_relu=True,
                 fast=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_f, out_f) * 0.3)
        self.bias = nn.Parameter(torch.zeros(out_f))
        self.mul, self.requant, self.relu_unit = mul, requant, relu_unit
        self.use_relu = use_relu
        self.luts = build_luts(mul, requant, relu_unit) if fast else None

    def forward(self, x):
        y = _VerifiedQGEMM.apply(x, self.weight, self.mul, self.requant,
                                 self.relu_unit, self.use_relu, self.luts)
        return y + self.bias


def _weights_dir():
    import os
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")


def load_units(mul_pt=None, requant_pt=None, relu_pt=None):
    """Load the TRAINED, N/N-verified units bundled with DaisyChain."""
    import os
    wd = _weights_dir()
    mul_pt = mul_pt or os.path.join(wd, "mul8.pt")
    requant_pt = requant_pt or os.path.join(wd, "requant16.pt")
    relu_pt = relu_pt or os.path.join(wd, "relu8.pt")
    from .mul8 import NeuralMul8
    from .ops import NeuralReLU8, NeuralRequant16
    mul = NeuralMul8()
    mul.atom.net.load_state_dict(torch.load(mul_pt)["state_dict"]); mul.atom.net.eval()
    relu = NeuralReLU8()
    relu.net.load_state_dict(torch.load(relu_pt)["state_dict"]); relu.net.eval()
    ck = torch.load(requant_pt)
    rq = NeuralRequant16(shift=ck["shift"])
    rq.net.load_state_dict(ck["state_dict"]); rq.net.eval()
    return mul, rq, relu
