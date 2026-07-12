"""Verified units — the trained, N/N bit-exact INT8 building blocks bundled with
DaisyChain (from the neural-gpu-verify project). Build models with VerifiedLinear
and their FORWARD compute runs through these verified units (multiply / requant /
ReLU), materialized as lookup tables for practical speed (fast=True).

These are pre-trained; users do NOT train new units — they train THROUGH them.
"""
from .qat import VerifiedLinear, load_units, build_luts
from .mul8 import NeuralMul8
from .ops import NeuralReLU8, NeuralRequant16
from . import instrument

__all__ = ["VerifiedLinear", "load_units", "build_luts",
           "NeuralMul8", "NeuralReLU8", "NeuralRequant16", "instrument"]
