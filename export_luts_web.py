"""Export the bundled verified units to lookup tables for the browser
(DaisyChain-Web). These are the emulated GPU logic, materialized: the browser
computes through THESE, not plain float.

Writes three little binaries into daisychain-web/public/:
  mul_lut.bin      int16[65536]  signed 8x8 product,  indexed [au*256 + bu]
  requant_lut.bin  int8 [65536]  int16->int8 requant, indexed [acc & 0xFFFF]
  relu_lut.bin     int8 [256]    int8 ReLU,           indexed [byte]
  luts_meta.json   dims + requant shift (for dequant)
"""
import json
import os
import numpy as np

from daisychain.verified.qat import load_units
from daisychain.verified.lut import build_mul8_lut, build_requant16_lut, build_relu8_lut

OUT = os.path.join(os.path.dirname(__file__), "..", "daisychain-web", "public")


def main():
    mul, rq, relu = load_units()
    mul_lut = build_mul8_lut(mul).astype(np.int16)          # (256,256) -> flat 65536
    req_lut = build_requant16_lut(rq).astype(np.int8)       # 65536
    relu_lut = build_relu8_lut(relu).astype(np.int8)        # 256

    os.makedirs(OUT, exist_ok=True)
    mul_lut.reshape(-1).tofile(os.path.join(OUT, "mul_lut.bin"))
    req_lut.tofile(os.path.join(OUT, "requant_lut.bin"))
    relu_lut.tofile(os.path.join(OUT, "relu_lut.bin"))
    meta = {"mul": [256, 256], "requant": 65536, "relu": 256, "shift": rq.shift}
    with open(os.path.join(OUT, "luts_meta.json"), "w") as f:
        json.dump(meta, f)

    # sanity: LUT must equal the true signed product (verified units are exact)
    a, b = 37, -19
    au, bu = a & 0xFF, b & 0xFF
    assert int(mul_lut[au, bu]) == a * b, "mul LUT mismatch"
    print("exported mul_lut(int16 65536), requant_lut(int8 65536), relu_lut(int8 256)")
    print("shift =", rq.shift, "| sanity 37*-19 =", int(mul_lut[au, bu]))


if __name__ == "__main__":
    main()
