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

    # CERTIFY THE EXPORTED TABLES, not a sample of them.
    #
    # These binaries are what the BROWSER computes through, and each is written
    # after a narrowing cast (int64 -> int16 / int8) that numpy performs silently:
    # an out-of-range value wraps rather than raising. The previous check was a
    # single pair (37 * -19), which cannot see a wrap anywhere else in the domain.
    #
    # The domain is finite and tiny, so checking ALL of it is the exhaustive
    # verification rather than a sample -- ~0.5 ms, the same argument
    # `certify_mul8_lut` already makes for the runtime table.
    au = np.repeat(np.arange(256), 256)
    bu = np.tile(np.arange(256), 256)
    sa = np.where(au >= 128, au - 256, au)
    sb = np.where(bu >= 128, bu - 256, bu)
    bad = int((mul_lut[au, bu].astype(np.int64) != sa * sb).sum())
    if bad:
        raise SystemExit("mul LUT: %d/65536 entries wrong after int16 cast" % bad)

    ref_rq = build_requant16_lut(rq).astype(np.int64)
    bad_rq = int((req_lut.astype(np.int64) != ref_rq).sum())
    if bad_rq:
        raise SystemExit("requant LUT: %d/65536 entries wrong after int8 cast" % bad_rq)

    ref_relu = build_relu8_lut(relu).astype(np.int64)
    bad_relu = int((relu_lut.astype(np.int64) != ref_relu).sum())
    if bad_relu:
        raise SystemExit("relu LUT: %d/256 entries wrong after int8 cast" % bad_relu)

    print("exported mul_lut(int16 65536), requant_lut(int8 65536), relu_lut(int8 256)")
    print("certified after cast: mul 65536/65536, requant 65536/65536, relu 256/256")
    print("shift =", rq.shift)


if __name__ == "__main__":
    main()
